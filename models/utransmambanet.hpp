#pragma once

#include "models/model_stub.hpp"
#include <torch/torch.h>

//==============================================================================
// Enhanced Transformer Block with Positional Encoding
//==============================================================================
class SimpleTransformerBlockImpl : public torch::nn::Module {
public:
    SimpleTransformerBlockImpl() = default;
    SimpleTransformerBlockImpl(int64_t dim, int64_t num_heads = 8);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::MultiheadAttention attention{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Sequential mlp{nullptr};
    torch::Tensor pos_encoding;  // Learnable positional encoding
    int64_t dim_;
};
TORCH_MODULE(SimpleTransformerBlock);

//==============================================================================
// Enhanced Mamba Block with Proper State Space Model
//==============================================================================
class MambaBlockImpl : public torch::nn::Module {
public:
    MambaBlockImpl() = default;
    MambaBlockImpl(int64_t dim, int64_t state_size = 16);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear proj_in{nullptr}, proj_out{nullptr};
    torch::nn::Linear delta_proj{nullptr};  // For selective mechanism
    torch::nn::Conv1d conv1d{nullptr};
    torch::Tensor A, B, C, D;  // State-space parameters
    torch::nn::Dropout dropout{nullptr};  // Added dropout
    torch::nn::LayerNorm norm{nullptr};
    int64_t dim_, state_size_;
};
TORCH_MODULE(MambaBlock);

//==============================================================================
// Enhanced Feature Fusion with Attention
//==============================================================================
class FeatureFusionImpl : public torch::nn::Module {
public:
    FeatureFusionImpl() = default;
    FeatureFusionImpl(int64_t conv_dim, int64_t trans_dim, int64_t mamba_dim, int64_t out_dim);
    torch::Tensor forward(torch::Tensor conv_feat,
                         torch::Tensor trans_feat = {},
                         torch::Tensor mamba_feat = {});

private:
    torch::nn::Conv2d conv_proj{nullptr}, trans_proj{nullptr}, mamba_proj{nullptr};
    torch::nn::Conv2d attention_conv{nullptr};  // For adaptive fusion
    torch::nn::Conv2d fusion_conv{nullptr};
    torch::nn::BatchNorm2d norm{nullptr};
    torch::nn::ReLU activation{nullptr};
    torch::nn::Dropout2d dropout{nullptr};  // Added dropout
};
TORCH_MODULE(FeatureFusion);

//==============================================================================
// ASPP Module for Multi-Scale Processing
//==============================================================================
class ASPPImpl : public torch::nn::Module {
private:
    torch::nn::Conv2d conv_1x1{nullptr};
    torch::nn::Conv2d conv_3x3_r6{nullptr};
    torch::nn::Conv2d conv_3x3_r12{nullptr};
    torch::nn::Conv2d conv_3x3_r18{nullptr};
    torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
    torch::nn::Conv2d global_conv{nullptr};
    torch::nn::Conv2d fusion_conv{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr}, bn_final{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Dropout dropout{nullptr};

public:
    ASPPImpl() = default;
    ASPPImpl(int64_t in_channels, int64_t out_channels = 256);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ASPP);

//==============================================================================
// Main UTransMambaNet Architecture - Complete Version
//==============================================================================
class UTransMambaNetImpl : public torch::nn::Cloneable<UTransMambaNetImpl> {
public:
    UTransMambaNetImpl(int64_t in_channels = 1, int64_t out_channels = 4);
    torch::Tensor forward(torch::Tensor x);
    void reset() override;

private:
    // === ENCODER BRANCHES ===
    torch::nn::Sequential down1{nullptr}, down2{nullptr}, down3{nullptr}, down4{nullptr}, down5{nullptr};
    torch::nn::MaxPool2d pool{nullptr};

    // Transformer branches (global context at coarse levels)
    SimpleTransformerBlock trans_block3{nullptr}, trans_block4{nullptr};

    // Mamba branches (efficient fine detail processing)
    MambaBlock mamba_block1{nullptr}, mamba_block2{nullptr};

    // ASPP bottleneck for multi-scale processing
    ASPP bottleneck_aspp{nullptr};

    // === DECODER ===
    torch::nn::ConvTranspose2d upconv0{nullptr}, upconv1{nullptr}, upconv2{nullptr}, upconv3{nullptr}, upconv4{nullptr};
    torch::nn::Sequential up0{nullptr}, up1{nullptr}, up2{nullptr}, up3{nullptr}, up4{nullptr};

    // === FEATURE FUSION ===
    FeatureFusion fusion1{nullptr}, fusion2{nullptr}, fusion3{nullptr}, fusion4{nullptr};

    // === FINAL OUTPUT ===
    torch::nn::Conv2d final_conv{nullptr};

    // Helper function to build two‐layer conv blocks
    torch::nn::Sequential double_conv(int64_t in_ch, int64_t out_ch);
};

TORCH_MODULE(UTransMambaNet);
