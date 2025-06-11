//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
// UTransMambaNet: Hybrid U-Net + Transformer + Mamba Implementation
//

#include "utransmambanet.hpp"

//==============================================================================
// TRANSFORMER BLOCK IMPLEMENTATION
//==============================================================================
SimpleTransformerBlockImpl::SimpleTransformerBlockImpl(int64_t dim, int64_t num_heads)
    : dim_(dim) {

    // Multi-head attention
    attention = register_module("attention",
        torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(dim, num_heads)));

    // Layer normalization - FIXED: Use vector constructor
    norm1 = register_module("norm1", torch::nn::LayerNorm(std::vector<int64_t>{dim}));
    norm2 = register_module("norm2", torch::nn::LayerNorm(std::vector<int64_t>{dim}));

    // MLP (Feed Forward Network)
    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(dim, dim * 4),
        torch::nn::GELU(),
        torch::nn::Linear(dim * 4, dim)
    ));
}

torch::Tensor SimpleTransformerBlockImpl::forward(torch::Tensor x) {
    // Input: [B, C, H, W] -> Need to convert to [H*W, B, C] for attention
    auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

    // Reshape for attention: [B, C, H, W] -> [H*W, B, C]
    auto x_flat = x.view({B, C, H * W}).permute({2, 0, 1}); // [H*W, B, C]

    // Self-attention with residual connection
    auto attn_out = std::get<0>(attention->forward(x_flat, x_flat, x_flat));
    x_flat = norm1->forward(x_flat + attn_out);

    // MLP with residual connection
    auto mlp_out = mlp->forward(x_flat);
    x_flat = norm2->forward(x_flat + mlp_out);

    // Reshape back: [H*W, B, C] -> [B, C, H, W]
    return x_flat.permute({1, 2, 0}).view({B, C, H, W});
}

//==============================================================================
// MAMBA BLOCK IMPLEMENTATION (Simplified State Space Model)
//==============================================================================
MambaBlockImpl::MambaBlockImpl(int64_t dim, int64_t state_size)
    : dim_(dim), state_size_(state_size) {

    // Projection layers - FIXED: Proper register_module syntax
    proj_in = register_module("proj_in", torch::nn::Linear(dim, dim * 2));
    proj_out = register_module("proj_out", torch::nn::Linear(dim, dim));

    // 1D convolution for local processing
    conv1d = register_module("conv1d", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(dim, dim, 3).padding(1).groups(dim)));

    // State space parameters - FIXED: Use register_parameter with torch::Tensor
    A = register_parameter("A", torch::randn({dim, state_size}));
    B = register_parameter("B", torch::randn({dim, state_size}));
    C = register_parameter("C", torch::randn({dim, state_size}));
    D = register_parameter("D", torch::randn({dim}));

    norm = register_module("norm", torch::nn::LayerNorm(std::vector<int64_t>{dim}));
}

torch::Tensor MambaBlockImpl::forward(torch::Tensor x) {
    // Input: [B, C, H, W] -> Process as sequences
    auto B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    auto x_orig = x;

    // Flatten spatial dimensions: [B, C, H*W]
    auto x_seq = x.view({B, C, H * W});

    // Project input
    auto x_proj = proj_in->forward(x_seq.permute({0, 2, 1})); // [B, H*W, 2*C]
    auto x_split = x_proj.chunk(2, -1);
    auto x_main = x_split[0];  // [B, H*W, C]
    auto x_gate = torch::sigmoid(x_split[1]);  // [B, H*W, C]

    // Apply 1D convolution (treat spatial sequence as 1D)
    auto x_conv = conv1d->forward(x_main.permute({0, 2, 1})); // [B, C, H*W]
    x_conv = torch::silu(x_conv).permute({0, 2, 1}); // [B, H*W, C]

    // Simplified state space computation (for hackathon speed)
    // In full Mamba, this would be more complex selective state space
    auto x_ssm = x_conv * x_gate;  // Simplified gating

    // Project output
    auto output = proj_out->forward(x_ssm); // [B, H*W, C]
    output = norm->forward(output);

    // Reshape back and residual connection
    output = output.permute({0, 2, 1}).view({B, C, H, W});
    return x_orig + output;
}

//==============================================================================
// FEATURE FUSION IMPLEMENTATION
//==============================================================================
FeatureFusionImpl::FeatureFusionImpl(int64_t conv_dim, int64_t trans_dim, int64_t mamba_dim, int64_t out_dim) {

    // Projection layers to match dimensions
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

    // Fusion convolution
    int64_t total_channels = out_dim;
    if (trans_dim > 0) total_channels += out_dim;
    if (mamba_dim > 0) total_channels += out_dim;

    fusion_conv = register_module("fusion_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(total_channels, out_dim, 3).padding(1)));

    norm = register_module("norm", torch::nn::BatchNorm2d(out_dim));
    activation = register_module("activation", torch::nn::ReLU());
}

torch::Tensor FeatureFusionImpl::forward(torch::Tensor conv_feat,
                                       torch::Tensor trans_feat,
                                       torch::Tensor mamba_feat) {

    std::vector<torch::Tensor> features;

    // Project conv features
    features.push_back(conv_proj->forward(conv_feat));

    // Project and add transformer features if available
    if (trans_feat.defined()) {
        features.push_back(trans_proj->forward(trans_feat));
    }

    // Project and add mamba features if available
    if (mamba_feat.defined()) {
        features.push_back(mamba_proj->forward(mamba_feat));
    }

    // Concatenate all features
    auto fused = torch::cat(features, 1);

    // Apply fusion convolution
    auto output = fusion_conv->forward(fused);
    output = norm->forward(output);
    output = activation->forward(output);

    return output;
}

//==============================================================================
// MAIN HYBRID ARCHITECTURE
//==============================================================================
UTransMambaNetImpl::UTransMambaNetImpl(int64_t in_channels, int64_t out_channels) {

    // === U-NET BACKBONE ===
    down1 = register_module("down1", double_conv(in_channels, 64));
    down2 = register_module("down2", double_conv(64, 128));
    down3 = register_module("down3", double_conv(128, 256));
    down4 = register_module("down4", double_conv(256, 512));
    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

    // === TRANSFORMER BRANCHES (Global context at coarse resolutions) ===
    trans_block3 = register_module("trans_block3", SimpleTransformerBlock(256, 8)); // 64x64
    trans_block4 = register_module("trans_block4", SimpleTransformerBlock(512, 8)); // 32x32

    // === MAMBA BRANCHES (Fine details at high resolutions) ===
    mamba_block1 = register_module("mamba_block1", MambaBlock(64, 16));   // 256x256
    mamba_block2 = register_module("mamba_block2", MambaBlock(128, 16));  // 128x128

    // === FEATURE FUSION ===
    fusion1 = register_module("fusion1", FeatureFusion(64, 0, 64, 64));     // Conv + Mamba
    fusion2 = register_module("fusion2", FeatureFusion(128, 0, 128, 128));  // Conv + Mamba
    fusion3 = register_module("fusion3", FeatureFusion(256, 256, 0, 256));  // Conv + Transformer
    fusion4 = register_module("fusion4", FeatureFusion(512, 512, 0, 512));  // Conv + Transformer

    // === DECODER ===
    upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
    upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
    upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
    upconv4 = register_module("upconv4", torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(64, 64, 2).stride(2)));

    up1 = register_module("up1", double_conv(512, 256));
    up2 = register_module("up2", double_conv(256, 128));
    up3 = register_module("up3", double_conv(128, 64));
    up4 = register_module("up4", double_conv(64 + in_channels, 64));

    // === FINAL OUTPUT ===
    final_conv = register_module("final_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, out_channels, 1)));
}

torch::Tensor UTransMambaNetImpl::forward(torch::Tensor x) {
    // === ENCODER WITH MULTI-BRANCH PROCESSING ===

    // Level 1: 256x256 -> Conv + Mamba for fine details
    auto d1_conv = down1->forward(x);                    // [B, 64, 256, 256]
    auto d1_mamba = mamba_block1->forward(d1_conv);      // [B, 64, 256, 256]
    auto d1_fused = fusion1->forward(d1_conv, {}, d1_mamba); // [B, 64, 256, 256]
    auto p1 = pool->forward(d1_conv);                   // [B, 64, 128, 128]

    // Level 2: 128x128 -> Conv + Mamba for fine details
    auto d2_conv = down2->forward(p1);                   // [B, 128, 128, 128]
    auto d2_mamba = mamba_block2->forward(d2_conv);      // [B, 128, 128, 128]
    auto d2_fused = fusion2->forward(d2_conv, {}, d2_mamba); // [B, 128, 128, 128]
    auto p2 = pool->forward(d2_conv);                   // [B, 128, 64, 64]

    // Level 3: 64x64 -> Conv + Transformer for global context
    auto d3_conv = down3->forward(p2);                   // [B, 256, 64, 64]
    auto d3_trans = trans_block3->forward(d3_conv);      // [B, 256, 64, 64]
    auto d3_fused = fusion3->forward(d3_conv, d3_trans, {}); // [B, 256, 64, 64]
    auto p3 = pool->forward(d3_conv);                   // [B, 256, 32, 32]

    // Level 4: 32x32 -> Conv + Transformer for global understanding
    auto d4_conv = down4->forward(p3);                   // [B, 512, 32, 32]
    // auto d4_trans = trans_block4->forward(d4_conv);      // [B, 512, 32, 32]
    // auto d4_fused = fusion4->forward(d4_conv, d4_trans, {}); // [B, 512, 32, 32]

    // Here the reason why d4_-fused is commented out is same as why we are only forwarding the convolution layers, can't risk the transformer or mamba losing spatial info

    // === DECODER WITH SKIP CONNECTIONS ===

    // Decoder Level 1
    auto u1 = upconv1->forward(d4_conv);                // [B, 256, 64, 64]
    if (u1.size(2) != d3_fused.size(2) || u1.size(3) != d3_fused.size(3)) {
        u1 = torch::nn::functional::interpolate(u1,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d3_fused.size(2), d3_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u1 = torch::cat({u1, d3_fused}, 1);                  // [B, 512, 64, 64]
    u1 = up1->forward(u1);                               // [B, 256, 64, 64]

    // Decoder Level 2
    auto u2 = upconv2->forward(u1);                      // [B, 128, 128, 128]
    if (u2.size(2) != d2_fused.size(2) || u2.size(3) != d2_fused.size(3)) {
        u2 = torch::nn::functional::interpolate(u2,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d2_fused.size(2), d2_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u2 = torch::cat({u2, d2_fused}, 1);                  // [B, 256, 128, 128]
    u2 = up2->forward(u2);                               // [B, 128, 128, 128]

    // Decoder Level 3
    auto u3 = upconv3->forward(u2);                      // [B, 64, 256, 256]
    if (u3.size(2) != d1_fused.size(2) || u3.size(3) != d1_fused.size(3)) {
        u3 = torch::nn::functional::interpolate(u3,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d1_fused.size(2), d1_fused.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u3 = torch::cat({u3, d1_fused}, 1);                  // [B, 128, 256, 256]
    u3 = up3->forward(u3);                               // [B, 64, 256, 256]

    // Decoder Level 4 (Final)
    auto u4 = upconv4->forward(u3);                      // [B, 64, 256, 256]
    if (u4.size(2) != x.size(2) || u4.size(3) != x.size(3)) {
        u4 = torch::nn::functional::interpolate(u4,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{x.size(2), x.size(3)})
                .mode(torch::kBilinear).align_corners(false));
    }
    u4 = torch::cat({u4, x}, 1);                         // [B, 65, 256, 256]
    u4 = up4->forward(u4);                               // [B, 64, 256, 256]

    return final_conv->forward(u4);                      // [B, 4, 256, 256]
}

torch::nn::Sequential UTransMambaNetImpl::double_conv(int64_t in_ch, int64_t out_ch) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}