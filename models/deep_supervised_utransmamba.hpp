//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#ifndef DEEP_SUPERVISED_UTRANSMAMBA_HPP
#define DEEP_SUPERVISED_UTRANSMAMBA_HPP

#include "utransmambanet.hpp"

// Enhanced UTransMambaNet with deep supervision
class EnhancedUTransMambaNetImpl : public UTransMambaNetImpl {
public:
    EnhancedUTransMambaNetImpl(int64_t in_channels, int64_t out_channels)
        : UTransMambaNetImpl(in_channels, out_channels) {

        // Replace bottleneck with ASPP
        aspp_bottleneck = register_module("aspp_bottleneck", ASPPModule(1024, 1024));

        // Deep supervision heads
        ds_conv4 = register_module("ds_conv4", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(512, out_channels, 1)));
        ds_conv3 = register_module("ds_conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256, out_channels, 1)));
        ds_conv2 = register_module("ds_conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, out_channels, 1)));

        // Dropout for regularization
        dropout = register_module("dropout", torch::nn::Dropout2d(0.1));
    }

    std::vector<torch::Tensor> forward_with_deep_supervision(torch::Tensor x) {
        std::vector<torch::Tensor> outputs;
        auto input_size = x.sizes();

        // Encoder (same as before but save intermediate features)
        auto d1_conv = down1->forward(x);
        auto d1_mamba = mamba_block1->forward(d1_conv);
        auto d1_fused = fusion1->forward(d1_conv, {}, d1_mamba);
        auto p1 = pool->forward(d1_conv);

        auto d2_conv = down2->forward(p1);
        auto d2_mamba = mamba_block2->forward(d2_conv);
        auto d2_fused = fusion2->forward(d2_conv, {}, d2_mamba);
        auto p2 = pool->forward(d2_conv);

        auto d3_conv = down3->forward(p2);
        auto d3_trans = trans_block3->forward(d3_conv);
        auto d3_fused = fusion3->forward(d3_conv, d3_trans, {});
        auto p3 = pool->forward(d3_conv);

        auto d4_conv = down4->forward(p3);
        auto d4_trans = trans_block4->forward(d4_conv);
        auto d4_fused = fusion4->forward(d4_conv, d4_trans, {});
        auto p4 = pool->forward(d4_conv);

        // Enhanced bottleneck with ASPP
        auto d5_conv = down5->forward(p4);
        d5_conv = aspp_bottleneck->forward(d5_conv);  // Multi-scale processing
        d5_conv = dropout->forward(d5_conv);  // Regularization

        // Decoder with deep supervision outputs
        auto u0 = upconv0->forward(d5_conv);
        if (u0.size(2) != d4_fused.size(2) || u0.size(3) != d4_fused.size(3)) {
            u0 = torch::nn::functional::interpolate(u0,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{d4_fused.size(2), d4_fused.size(3)})
                    .mode(torch::kBilinear).align_corners(false));
        }
        u0 = torch::cat({u0, d4_fused}, 1);
        u0 = up0->forward(u0);

        // Deep supervision output 1
        auto ds_out4 = ds_conv4->forward(u0);
        ds_out4 = torch::nn::functional::interpolate(ds_out4,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{input_size[2], input_size[3]})
                .mode(torch::kBilinear).align_corners(false));
        outputs.push_back(ds_out4);

        // Continue decoder
        auto u1 = upconv1->forward(u0);
        if (u1.size(2) != d3_fused.size(2) || u1.size(3) != d3_fused.size(3)) {
            u1 = torch::nn::functional::interpolate(u1,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{d3_fused.size(2), d3_fused.size(3)})
                    .mode(torch::kBilinear).align_corners(false));
        }
        u1 = torch::cat({u1, d3_fused}, 1);
        u1 = up1->forward(u1);

        // Deep supervision output 2
        auto ds_out3 = ds_conv3->forward(u1);
        ds_out3 = torch::nn::functional::interpolate(ds_out3,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{input_size[2], input_size[3]})
                .mode(torch::kBilinear).align_corners(false));
        outputs.push_back(ds_out3);

        // Continue to final output...
        auto u2 = upconv2->forward(u1);
        if (u2.size(2) != d2_fused.size(2) || u2.size(3) != d2_fused.size(3)) {
            u2 = torch::nn::functional::interpolate(u2,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{d2_fused.size(2), d2_fused.size(3)})
                    .mode(torch::kBilinear).align_corners(false));
        }
        u2 = torch::cat({u2, d2_fused}, 1);
        u2 = up2->forward(u2);

        auto u3 = upconv3->forward(u2);
        if (u3.size(2) != d1_fused.size(2) || u3.size(3) != d1_fused.size(3)) {
            u3 = torch::nn::functional::interpolate(u3,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{d1_fused.size(2), d1_fused.size(3)})
                    .mode(torch::kBilinear).align_corners(false));
        }
        u3 = torch::cat({u3, d1_fused}, 1);
        u3 = up3->forward(u3);

        auto u4 = upconv4->forward(u3);
        if (u4.size(2) != x.size(2) || u4.size(3) != x.size(3)) {
            u4 = torch::nn::functional::interpolate(u4,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{x.size(2), x.size(3)})
                    .mode(torch::kBilinear).align_corners(false));
        }
        u4 = torch::cat({u4, x}, 1);
        u4 = up4->forward(u4);

        // Final output
        auto final_out = final_conv->forward(u4);
        outputs.push_back(final_out);  // Main output is last

        return outputs;
    }

    // Override forward for standard use
    torch::Tensor forward(torch::Tensor x) override {
        auto outputs = forward_with_deep_supervision(x);
        return outputs.back();  // Return main output only
    }

private:
    ASPPModule aspp_bottleneck{nullptr};
    torch::nn::Conv2d ds_conv4{nullptr}, ds_conv3{nullptr}, ds_conv2{nullptr};
    torch::nn::Dropout2d dropout{nullptr};
};
TORCH_MODULE(EnhancedUTransMambaNet);

#endif //DEEP_SUPERVISED_UTRANSMAMBA_HPP
