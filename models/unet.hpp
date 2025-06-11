//
// Created by Emin Tunc Kirimlioglu on 5/29/25.
//

#pragma once

#include "models/model_stub.hpp"



class UNetImpl: public torch::nn::Cloneable<UNetImpl> {
 public:
    UNetImpl(int64_t in_channels, int64_t out_channels);

    ::torch::Tensor forward(::torch::Tensor x);

    void reset() override;

 private:
    // Encoder
    torch::nn::Sequential down1, down2, down3, down4;
    torch::nn::MaxPool2d pool;
    
    // Decoder
    torch::nn::ConvTranspose2d upconv1, upconv2, upconv3, upconv4;
    torch::nn::Sequential up1, up2, up3, up4;
    
    // Final layer
    torch::nn::Conv2d final_conv;

    torch::nn::Sequential double_conv(int64_t in_ch, int64_t out_ch);
};

TORCH_MODULE(UNet);
