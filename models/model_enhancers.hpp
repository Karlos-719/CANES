//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#ifndef MODEL_ENHANCERS_HPP
#define MODEL_ENHANCERS_HPP

#include <torch/torch.h>

// Squeeze-and-Excitation block for channel attention
class SEBlockImpl : public torch::nn::Module {
public:
    SEBlockImpl(int64_t channels, int64_t reduction = 16) {
        squeeze = torch::nn::AdaptiveAvgPool2d(1);
        excitation = torch::nn::Sequential(
            torch::nn::Linear(channels, channels / reduction),
            torch::nn::ReLU(),
            torch::nn::Linear(channels / reduction, channels),
            torch::nn::Sigmoid()
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        auto b = x.size(0), c = x.size(1);
        auto y = squeeze(x).view({b, c});
        y = excitation->forward(y).view({b, c, 1, 1});
        return x * y;
    }

private:
    torch::nn::AdaptiveAvgPool2d squeeze{nullptr};
    torch::nn::Sequential excitation{nullptr};
};
TORCH_MODULE(SEBlock);

torch::nn::Sequential enhanced_double_conv(int64_t in_ch, int64_t out_ch) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        SEBlock(out_ch),  // Add channel attention
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}

// ASPP (Atrous Spatial Pyramid Pooling) module for multi-scale features
class ASPPModuleImpl : public torch::nn::Module {
public:
    ASPPModuleImpl(int64_t in_channels, int64_t out_channels) {
        // Multiple parallel convolutions with different dilation rates
        conv1x1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1));

        conv3x3_1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                .padding(6).dilation(6));

        conv3x3_2 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                .padding(12).dilation(12));

        conv3x3_3 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                .padding(18).dilation(18));

        // Global pooling branch
        global_pool = torch::nn::AdaptiveAvgPool2d(1);
        conv_pool = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1));

        // Fusion
        fusion = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels * 5, out_channels, 1)),
            torch::nn::BatchNorm2d(out_channels),
            torch::nn::ReLU()
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        auto size = x.sizes();

        // Parallel branches
        auto feat1 = conv1x1->forward(x);
        auto feat2 = conv3x3_1->forward(x);
        auto feat3 = conv3x3_2->forward(x);
        auto feat4 = conv3x3_3->forward(x);

        // Global pooling branch
        auto feat5 = global_pool->forward(x);
        feat5 = conv_pool->forward(feat5);
        feat5 = torch::nn::functional::interpolate(feat5,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{size[2], size[3]})
                .mode(torch::kBilinear).align_corners(false));

        // Concatenate all features
        auto out = torch::cat({feat1, feat2, feat3, feat4, feat5}, 1);
        return fusion->forward(out);
    }

private:
    torch::nn::Conv2d conv1x1{nullptr}, conv3x3_1{nullptr}, conv3x3_2{nullptr}, conv3x3_3{nullptr};
    torch::nn::Conv2d conv_pool{nullptr};
    torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
    torch::nn::Sequential fusion{nullptr};
};
TORCH_MODULE(ASPPModule);


#endif //MODEL_ENHANCERS_HPP
