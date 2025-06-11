#include "utransmambanet.hpp"

//==============================================================================
// SIMPLE TRANSFORMER BLOCK IMPLEMENTATION
//==============================================================================
SimpleTransformerBlockImpl::SimpleTransformerBlockImpl(int64_t dim, int64_t num_heads)
    : dim_(dim) {
    // Multi‐head attention
    attention = register_module("attention",
        torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(dim, num_heads)));

    // Layer normalization
    norm1 = register_module("norm1", torch::nn::LayerNorm(std::vector<int64_t>{dim}));
    norm2 = register_module("norm2", torch::nn::LayerNorm(std::vector<int64_t>{dim}));

    // MLP with dropout for regularization
    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(dim, dim * 4),  // 4× expansion
        torch::nn::GELU(),
        torch::nn::Dropout(0.1),
        torch::nn::Linear(dim * 4, dim),
        torch::nn::Dropout(0.1)
    ));

    // Positional encoding (learnable)
    pos_encoding = register_parameter("pos_encoding", torch::randn({1, 1, dim}) * 0.02);
}

torch::Tensor SimpleTransformerBlockImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    // Reshape: [B, C, H, W] -> [H*W, B, C]
    auto x_flat = x.view({batch_size, channels, height * width})
                      .permute({2, 0, 1});  // [seq_len, B, C]

    // Add positional encoding via broadcasting: [1,1,dim] -> [seq_len, B, dim]
    x_flat = x_flat + pos_encoding;

    // Self‐attention (query=key=value=x_flat)
    auto attn_out = std::get<0>(attention->forward(x_flat, x_flat, x_flat));
    x_flat = norm1->forward(x_flat + attn_out);

    // MLP with residual
    auto mlp_out = mlp->forward(x_flat);
    x_flat = norm2->forward(x_flat + mlp_out);

    // Reshape back: [seq_len, B, C] -> [B, C, H, W]
    return x_flat.permute({1, 2, 0})
                 .view({batch_size, channels, height, width});
}

//==============================================================================
// MAMBA BLOCK IMPLEMENTATION (Fixed State‐Space)
//==============================================================================
MambaBlockImpl::MambaBlockImpl(int64_t dim, int64_t state_size)
    : dim_(dim), state_size_(state_size) {
    // Projection layers
    proj_in = register_module("proj_in", torch::nn::Linear(dim, dim * 2));
    proj_out = register_module("proj_out", torch::nn::Linear(dim, dim));

    // 1D convolution for local processing (depthwise)
    conv1d = register_module("conv1d", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(dim, dim, 3).padding(1).groups(dim)));

    // FIXED: More stable parameter initialization
    A = register_parameter("A", -torch::ones({dim, state_size}) * 0.1);   // More conservative initialization
    B = register_parameter("B", torch::randn({dim, state_size}) * 0.001); // Much smaller initialization
    C = register_parameter("C", torch::randn({state_size, dim}) * 0.001); // Much smaller initialization
    D = register_parameter("D", torch::ones({dim}) * 0.1);                // Smaller skip connection

    // Delta parameter for selective mechanism
    delta_proj = register_module("delta_proj", torch::nn::Linear(dim, dim));

    // Dropout and LayerNorm
    dropout = register_module("dropout", torch::nn::Dropout(0.1));
    norm = register_module("norm", torch::nn::LayerNorm(std::vector<int64_t>{dim}));
}

torch::Tensor MambaBlockImpl::forward(torch::Tensor x) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    // Keep original for residual
    auto x_orig = x;

    // Flatten spatial dims: [B, C, H, W] -> [B, C, H*W]
    auto x_seq = x.view({batch_size, channels, height * width});

    // Project input: [B, H*W, 2*C]
    auto x_proj = proj_in->forward(x_seq.permute({0, 2, 1}));  // [B, H*W, 2C]
    auto x_split = x_proj.chunk(2, -1);
    auto x_main = x_split[0];  // [B, H*W, C]
    auto x_gate = torch::sigmoid(x_split[1]);  // [B, H*W, C]

    // Depthwise 1D conv on the "main" sequence: [B, C, H*W]
    auto x_conv = conv1d->forward(x_main.permute({0, 2, 1}));  // [B, C, H*W]
    x_conv = torch::silu(x_conv).permute({0, 2, 1});           // [B, H*W, C]

    // FIXED: More stable selective state-space computation
    auto delta = torch::softplus(delta_proj->forward(x_conv));  // [B, H*W, C]
    
    // FIXED: Clamp delta to prevent explosion
    delta = torch::clamp(delta, 1e-6, 10.0);

    // SIMPLIFIED: Use simpler state space computation to avoid instability
    // Instead of complex cumulative products, use a more stable approximation
    auto A_discrete = torch::exp(A.unsqueeze(0) * delta.unsqueeze(-1) * 0.1);  // Scale down
    
    // FIXED: Clamp to prevent overflow
    A_discrete = torch::clamp(A_discrete, 1e-6, 1.0);

    // Simplified state computation without cumulative products
    auto B_expanded = B.unsqueeze(0).unsqueeze(0);    // [1, 1, C, state_size]
    auto input_contrib = x_conv.unsqueeze(-1) * B_expanded;  // [B, H*W, C, state_size]
    
    // FIXED: Use mean instead of cumsum to avoid explosion
    auto states = input_contrib * A_discrete;  // Element-wise instead of cumulative
    
    // Output computation
    auto C_t = C.t().unsqueeze(0).unsqueeze(0);             // [1, 1, C, state_size]
    auto weighted_states = states * C_t;                    // [B, H*W, C, state_size]
    auto output_raw = torch::sum(weighted_states, /*dim=*/3); // [B, H*W, C]

    // FIXED: Scale down the skip connection
    output_raw = output_raw + x_conv * D * 0.1;  // Smaller skip connection

    // Apply gating and dropout
    auto x_ssm = dropout->forward(output_raw * x_gate);  // [B, H*W, C]

    // Project output back to "C" features
    auto output = proj_out->forward(x_ssm);  // [B, H*W, C]
    output = norm->forward(output);          // LayerNorm over last dim

    // FIXED: Smaller residual connection coefficient
    output = output.permute({0, 2, 1}).view({batch_size, channels, height, width});
    return x_orig + output * 0.1;  // Scale down residual connection
}

//==============================================================================
// FEATURE FUSION IMPLEMENTATION (Adaptive Fusion)
//==============================================================================
FeatureFusionImpl::FeatureFusionImpl(int64_t conv_dim, int64_t trans_dim, int64_t mamba_dim, int64_t out_dim) {
    // Project each branch (if present) down to out_dim-channel feature maps
    conv_proj = register_module("conv_proj", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(conv_dim, out_dim, 1)));

    if (trans_dim > 0) {
        trans_proj = register_module("trans_proj", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(trans_dim, out_dim, 1)));
    }

    if (mamba_dim > 0) {
        mamba_proj = register_module("mamba_proj", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(mamba_dim, out_dim, 1)));
    }

    // Attention over concatenated feature‐maps
    int64_t total_channels = out_dim;
    if (trans_dim > 0) total_channels += out_dim;
    if (mamba_dim > 0) total_channels += out_dim;

    attention_conv = register_module("attention_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(total_channels, total_channels, 1)));
    fusion_conv = register_module("fusion_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(total_channels, out_dim, 3).padding(1)));

    norm = register_module("norm", torch::nn::BatchNorm2d(out_dim));
    activation = register_module("activation", torch::nn::ReLU());
    dropout = register_module("dropout", torch::nn::Dropout2d(0.1));
}

torch::Tensor FeatureFusionImpl::forward(torch::Tensor conv_feat,
                                         torch::Tensor trans_feat,
                                         torch::Tensor mamba_feat) {
    std::vector<torch::Tensor> features;

    // Always project the convolutional‐branch features
    features.push_back(conv_proj->forward(conv_feat));

    // Optionally project transformer features
    if (trans_feat.defined()) {
        features.push_back(trans_proj->forward(trans_feat));
    }

    // Optionally project mamba features
    if (mamba_feat.defined()) {
        features.push_back(mamba_proj->forward(mamba_feat));
    }

    // Concatenate along channel‐dimension: [B, total_channels, H, W]
    auto fused = torch::cat(features, /*dim=*/1);

    // Attention weighting
    auto attention_weights = torch::sigmoid(attention_conv->forward(fused));
    fused = fused * attention_weights;

    // Final 3×3 fusion + BN + ReLU + Dropout
    auto output = fusion_conv->forward(fused);
    output = norm->forward(output);
    output = activation->forward(output);
    output = dropout->forward(output);

    return output;
}

//==============================================================================
// ASPP IMPLEMENTATION
//==============================================================================
ASPPImpl::ASPPImpl(int64_t in_channels, int64_t out_channels) {
    // 1×1 conv branch
    conv_1x1 = register_module("conv_1x1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 1)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));

    // Three 3×3 convs with dilations 6, 12, 18
    conv_3x3_r6 = register_module("conv_3x3_r6", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(6).dilation(6)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

    conv_3x3_r12 = register_module("conv_3x3_r12", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(12).dilation(12)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));

    conv_3x3_r18 = register_module("conv_3x3_r18", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(18).dilation(18)));
    bn4 = register_module("bn4", torch::nn::BatchNorm2d(out_channels));

    // Global avg‐pool branch
    global_pool = register_module("global_pool",
        torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    global_conv = register_module("global_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 1)));
    bn5 = register_module("bn5", torch::nn::BatchNorm2d(out_channels));

    // Fusion (concatenate 5 branches)
    fusion_conv = register_module("fusion_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels * 5, in_channels, 1)));
    bn_final = register_module("bn_final", torch::nn::BatchNorm2d(in_channels));

    relu = register_module("relu", torch::nn::ReLU());
    dropout = register_module("dropout", torch::nn::Dropout(0.1));
}

torch::Tensor ASPPImpl::forward(torch::Tensor x) {
    auto height = x.size(2);
    auto width = x.size(3);

    // Branch 1: 1×1 conv
    auto branch1 = relu->forward(bn1->forward(conv_1x1->forward(x)));

    // Branch 2: 3×3, dilation=6
    auto branch2 = relu->forward(bn2->forward(conv_3x3_r6->forward(x)));

    // Branch 3: 3×3, dilation=12
    auto branch3 = relu->forward(bn3->forward(conv_3x3_r12->forward(x)));

    // Branch 4: 3×3, dilation=18
    auto branch4 = relu->forward(bn4->forward(conv_3x3_r18->forward(x)));

    // Branch 5: global avg pool -> 1×1 conv -> upsample
    auto branch5 = global_pool->forward(x);
    branch5 = relu->forward(bn5->forward(global_conv->forward(branch5)));
    branch5 = torch::nn::functional::interpolate(branch5,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear).align_corners(false));

    // Concatenate all 5 branches
    auto concat = torch::cat({branch1, branch2, branch3, branch4, branch5}, /*dim=*/1);

    // Fuse back to original channels + residual
    auto output = relu->forward(bn_final->forward(fusion_conv->forward(concat)));
    return dropout->forward(output) + x;
}

//==============================================================================
// UTRANS‐MAMBA‐NET IMPLEMENTATION
//==============================================================================
UTransMambaNetImpl::UTransMambaNetImpl(int64_t in_channels, int64_t out_channels) {
    // === U‐Net ENCODER ===
    down1 = register_module("down1", double_conv(in_channels, 64));
    down2 = register_module("down2", double_conv(64, 128));
    down3 = register_module("down3", double_conv(128, 256));
    down4 = register_module("down4", double_conv(256, 512));
    down5 = register_module("down5", double_conv(512, 1024));
    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

    // === TRANSFORMER BRANCHES ===
    trans_block3 = register_module("trans_block3", SimpleTransformerBlock(256, 8));
    trans_block4 = register_module("trans_block4", SimpleTransformerBlock(512, 8));

    // === MAMBA BRANCHES ===
    mamba_block1 = register_module("mamba_block1", MambaBlock(64, 16));
    mamba_block2 = register_module("mamba_block2", MambaBlock(128, 16));

    // === FEATURE FUSION AT EACH LEVEL ===
    fusion1 = register_module("fusion1", FeatureFusion(64, 0, 64, 64));
    fusion2 = register_module("fusion2", FeatureFusion(128, 0, 128, 128));
    fusion3 = register_module("fusion3", FeatureFusion(256, 256, 0, 256));
    fusion4 = register_module("fusion4", FeatureFusion(512, 512, 0, 512));

    // === ASPP BOTTLENECK ===
    bottleneck_aspp = register_module("bottleneck_aspp", ASPP(1024, 256));

    // === DECODER (UP‐CONVOLUTIONS) ===
    upconv0 = register_module("upconv0", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(1024, 512, 2).stride(2)));
    upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
    upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
    upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(128,  64, 2).stride(2)));
    upconv4 = register_module("upconv4", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions( 64,  64, 2).stride(2)));

    up0 = register_module("up0", double_conv(1024, 512));
    up1 = register_module("up1", double_conv(512, 256));
    up2 = register_module("up2", double_conv(256, 128));
    up3 = register_module("up3", double_conv(128,  64));
    up4 = register_module("up4", double_conv(64 + in_channels, 64));

    // === FINAL OUTPUT LAYER ===
    final_conv = register_module("final_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, out_channels, 1)));
}

torch::Tensor UTransMambaNetImpl::forward(torch::Tensor x) {
    // === ENCODER WITH MULTI‐BRANCH PROCESSING ===
    // Level 1: down1 → Mamba → fuse → pool
    auto d1_conv = down1->forward(x);                  // [B, 64, H, W]
    auto d1_mamba = mamba_block1->forward(d1_conv);    // [B, 64, H, W]
    auto d1_fused = fusion1->forward(d1_conv, {}, d1_mamba);
    auto p1 = pool->forward(d1_conv);

    // Level 2: down2 → Mamba → fuse → pool
    auto d2_conv = down2->forward(p1);                 // [B,128, H/2, W/2]
    auto d2_mamba = mamba_block2->forward(d2_conv);    // [B,128, H/2, W/2]
    auto d2_fused = fusion2->forward(d2_conv, {}, d2_mamba);
    auto p2 = pool->forward(d2_conv);

    // Level 3: down3 → Transformer → fuse → pool
    auto d3_conv = down3->forward(p2);                 // [B,256, H/4, W/4]
    auto d3_trans = trans_block3->forward(d3_conv);    // [B,256, H/4, W/4]
    auto d3_fused = fusion3->forward(d3_conv, d3_trans, {});
    auto p3 = pool->forward(d3_conv);

    // Level 4: down4 → Transformer → fuse → pool
    auto d4_conv = down4->forward(p3);                 // [B,512, H/8, W/8]
    auto d4_trans = trans_block4->forward(d4_conv);    // [B,512, H/8, W/8]
    auto d4_fused = fusion4->forward(d4_conv, d4_trans, {});
    auto p4 = pool->forward(d4_conv);

    // Level 5: down5 → ASPP bottleneck
    auto d5_conv = down5->forward(p4);                 // [B,1024, H/16, W/16]
    auto d5_enhanced = bottleneck_aspp->forward(d5_conv); // [B,1024, H/16, W/16]

    // === DECODER: FULL SKIP CONNECTIONS ===
    // Decoder Level 0: Bottleneck → Level 4
    auto u0 = upconv0->forward(d5_enhanced);  // [B,512, H/8, W/8]
    if (u0.size(2) != d4_fused.size(2) || u0.size(3) != d4_fused.size(3)) {
        u0 = torch::nn::functional::interpolate(u0,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d4_fused.size(2), d4_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u0 = torch::cat({u0, d4_fused}, 1);  // [B,1024, H/8, W/8]
    u0 = up0->forward(u0);               // [B,512, H/8, W/8]

    // Decoder Level 1: → Level 3
    auto u1 = upconv1->forward(u0);      // [B,256, H/4, W/4]
    if (u1.size(2) != d3_fused.size(2) || u1.size(3) != d3_fused.size(3)) {
        u1 = torch::nn::functional::interpolate(u1,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d3_fused.size(2), d3_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u1 = torch::cat({u1, d3_fused}, 1);  // [B,512, H/4, W/4]
    u1 = up1->forward(u1);               // [B,256, H/4, W/4]

    // Decoder Level 2: → Level 2
    auto u2 = upconv2->forward(u1);      // [B,128, H/2, W/2]
    if (u2.size(2) != d2_fused.size(2) || u2.size(3) != d2_fused.size(3)) {
        u2 = torch::nn::functional::interpolate(u2,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d2_fused.size(2), d2_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u2 = torch::cat({u2, d2_fused}, 1);  // [B,256, H/2, W/2]
    u2 = up2->forward(u2);               // [B,128, H/2, W/2]

    // Decoder Level 3: → Level 1
    auto u3 = upconv3->forward(u2);      // [B, 64, H, W]
    if (u3.size(2) != d1_fused.size(2) || u3.size(3) != d1_fused.size(3)) {
        u3 = torch::nn::functional::interpolate(u3,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d1_fused.size(2), d1_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u3 = torch::cat({u3, d1_fused}, 1);  // [B,128, H, W]
    u3 = up3->forward(u3);               // [B, 64, H, W]

    // Decoder Level 4: → Final output size
    auto u4 = upconv4->forward(u3);      // [B, 64, H, W]
    if (u4.size(2) != x.size(2) || u4.size(3) != x.size(3)) {
        u4 = torch::nn::functional::interpolate(u4,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{x.size(2), x.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u4 = torch::cat({u4, x}, 1);          // [B, 64 + in_channels, H, W]
    u4 = up4->forward(u4);                // [B, 64, H, W]

    return final_conv->forward(u4);       // [B, out_channels, H, W]
}

torch::nn::Sequential UTransMambaNetImpl::double_conv(int64_t in_ch, int64_t out_ch) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout2d(0.1),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}

void UTransMambaNetImpl::reset() {
    // Get original filters from final_conv to determine input‐channels
    auto weight = final_conv->named_parameters()["weight"];
    auto in_channels = weight.size(1) == 64 ? 1 : weight.size(1) - 64;
    auto out_channels = weight.size(0);

    // Re‐register every submodule exactly as in constructor
    down1 = register_module("down1", double_conv(in_channels, 64));
    down2 = register_module("down2", double_conv(64, 128));
    down3 = register_module("down3", double_conv(128, 256));
    down4 = register_module("down4", double_conv(256, 512));
    down5 = register_module("down5", double_conv(512, 1024));
    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

    trans_block3 = register_module("trans_block3", SimpleTransformerBlock(256, 8));
    trans_block4 = register_module("trans_block4", SimpleTransformerBlock(512, 8));

    mamba_block1 = register_module("mamba_block1", MambaBlock(64, 16));
    mamba_block2 = register_module("mamba_block2", MambaBlock(128, 16));

    fusion1 = register_module("fusion1", FeatureFusion(64, 0, 64, 64));
    fusion2 = register_module("fusion2", FeatureFusion(128, 0, 128, 128));
    fusion3 = register_module("fusion3", FeatureFusion(256, 256, 0, 256));
    fusion4 = register_module("fusion4", FeatureFusion(512, 512, 0, 512));

    bottleneck_aspp = register_module("bottleneck_aspp", ASPP(1024, 256));

    upconv0 = register_module("upconv0", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(1024, 512, 2).stride(2)));
    upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
    upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
    upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(128,  64, 2).stride(2)));
    upconv4 = register_module("upconv4", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions( 64,  64, 2).stride(2)));

    up0 = register_module("up0", double_conv(1024, 512));
    up1 = register_module("up1", double_conv(512, 256));
    up2 = register_module("up2", double_conv(256, 128));
    up3 = register_module("up3", double_conv(128,  64));
    up4 = register_module("up4", double_conv(64 + in_channels, 64));

    final_conv = register_module("final_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, out_channels, 1)));
}
