//
// Created by Emin Tunc Kirimlioglu on 5/29/25.
// FIXED: Channel mismatch in final skip connection
//

#include "unet.hpp"

UNetImpl::UNetImpl(int64_t in_channels, int64_t out_channels)
    : down1(double_conv(in_channels, 64)),
      down2(double_conv(64, 128)),
      down3(double_conv(128, 256)),
      down4(double_conv(256, 512)),
      pool(torch::nn::MaxPool2dOptions(2)),
      upconv1(torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)),
      upconv2(torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)),
      upconv3(torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)),
      upconv4(torch::nn::ConvTranspose2dOptions(64, 64, 2).stride(2)),
      up1(double_conv(512, 256)),  // 256 + 256 = 512 input channels ✓
      up2(double_conv(256, 128)),  // 128 + 128 = 256 input channels ✓
      up3(double_conv(128, 64)),   // 64 + 64 = 128 input channels ✓
      up4(double_conv(64 + in_channels, 64)),  // 🔧 FIXED: 64 + 1 = 65 input channels
      final_conv(torch::nn::Conv2dOptions(64, out_channels, 1)) {

    register_module("down1", down1);
    register_module("down2", down2);
    register_module("down3", down3);
    register_module("down4", down4);
    register_module("pool", pool);
    register_module("upconv1", upconv1);
    register_module("upconv2", upconv2);
    register_module("upconv3", upconv3);
    register_module("upconv4", upconv4);
    register_module("up1", up1);
    register_module("up2", up2);
    register_module("up3", up3);
    register_module("up4", up4);
    register_module("final_conv", final_conv);
}

::torch::Tensor UNetImpl::forward(::torch::Tensor x) {
    // Encoder
    auto d1 = down1->forward(x);     // 1 → 64 channels
    auto p1 = pool->forward(d1);

    auto d2 = down2->forward(p1);    // 64 → 128 channels
    auto p2 = pool->forward(d2);

    auto d3 = down3->forward(p2);    // 128 → 256 channels
    auto p3 = pool->forward(d3);

    auto d4 = down4->forward(p3);    // 256 → 512 channels

    // Decoder with skip connections
    auto u1 = upconv1->forward(d4);  // 512 → 256 channels
    // Ensure tensor sizes match for concatenation
    if (u1.size(2) != d3.size(2) || u1.size(3) != d3.size(3)) {
        u1 = torch::nn::functional::interpolate(u1,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d3.size(2), d3.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    u1 = torch::cat({u1, d3}, 1);    // 256 + 256 = 512 channels
    u1 = up1->forward(u1);           // 512 → 256 channels

    auto u2 = upconv2->forward(u1);  // 256 → 128 channels
    if (u2.size(2) != d2.size(2) || u2.size(3) != d2.size(3)) {
        u2 = torch::nn::functional::interpolate(u2,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d2.size(2), d2.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    u2 = torch::cat({u2, d2}, 1);    // 128 + 128 = 256 channels
    u2 = up2->forward(u2);           // 256 → 128 channels

    auto u3 = upconv3->forward(u2);  // 128 → 64 channels
    if (u3.size(2) != d1.size(2) || u3.size(3) != d1.size(3)) {
        u3 = torch::nn::functional::interpolate(u3,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{d1.size(2), d1.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    u3 = torch::cat({u3, d1}, 1);    // 64 + 64 = 128 channels
    u3 = up3->forward(u3);           // 128 → 64 channels

    auto u4 = upconv4->forward(u3);  // 64 → 64 channels
    if (u4.size(2) != x.size(2) || u4.size(3) != x.size(3)) {
        u4 = torch::nn::functional::interpolate(u4,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{x.size(2), x.size(3)})
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    u4 = torch::cat({u4, x}, 1);     // 64 + 1 = 65 channels ✅
    u4 = up4->forward(u4);           // 65 → 64 channels ✅

    return final_conv->forward(u4);  // 64 → 4 channels (classes)
}

torch::nn::Sequential UNetImpl::double_conv(int64_t in_ch, int64_t out_ch) {
    return torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
        torch::nn::BatchNorm2d(out_ch),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
    );
}

void UNetImpl::reset() {
    // Store the original parameters (assuming 1 input, 4 output channels)
    // If you need dynamic channels, store them as member variables
    int64_t in_channels = 1;
    int64_t out_channels = 4;

    // Re-initialize all modules exactly as in constructor
    down1 = register_module("down1", double_conv(in_channels, 64));
    down2 = register_module("down2", double_conv(64, 128));
    down3 = register_module("down3", double_conv(128, 256));
    down4 = register_module("down4", double_conv(256, 512));

    pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));

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

    final_conv = register_module("final_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, out_channels, 1)));
}