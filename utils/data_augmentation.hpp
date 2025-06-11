//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#ifndef DATA_AUGMENTATION_HPP
#define DATA_AUGMENTATION_HPP

#include <torch/torch.h>
#include <torch/data/transforms.h>
#include <random>

// Data augmentation transforms for medical images
struct RandomFlip : public torch::data::transforms::TensorTransform<torch::Tensor> {
    torch::Tensor operator()(torch::Tensor input) override {
        if (torch::rand({1}).item<float>() > 0.5) {
            // Horizontal flip
            input = torch::flip(input, {-1});
        }
        return input;
    }
};

struct RandomRotate : public torch::data::transforms::TensorTransform<torch::Tensor> {
    float max_angle;

    RandomRotate(float max_angle_degrees = 15.0) : max_angle(max_angle_degrees) {}

    torch::Tensor operator()(torch::Tensor input) override {
        float angle = (torch::rand({1}).item<float>() - 0.5) * 2.0 * max_angle; // ±max_angle degrees

        // Convert angle to radians
        float angle_rad = angle * M_PI / 180.0;

        // Get input dimensions
        auto sizes = input.sizes();
        int height = sizes[sizes.size() - 2];
        int width = sizes[sizes.size() - 1];

        // Create rotation matrix
        auto cos_val = std::cos(angle_rad);
        auto sin_val = std::sin(angle_rad);

        // 2x3 affine transformation matrix for rotation around center
        auto theta = torch::tensor({
            {cos_val, -sin_val, 0.0},
            {sin_val, cos_val, 0.0}
        }, torch::kFloat32);

        // Add batch dimension if needed
        if (input.dim() == 3) {
            // Add batch dimension
            input = input.unsqueeze(0);
            theta = theta.unsqueeze(0);

            // Create grid and apply transformation
            auto grid = torch::nn::functional::affine_grid(theta,
                {1, input.size(1), height, width},
                torch::nn::functional::AffineGridFuncOptions().align_corners(false));

            auto rotated = torch::nn::functional::grid_sample(input, grid,
                torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false));

            // Remove batch dimension
            return rotated.squeeze(0);
        } else if (input.dim() == 4) {
            // Batch processing
            int batch_size = input.size(0);
            theta = theta.unsqueeze(0).expand({batch_size, -1, -1});

            auto grid = torch::nn::functional::affine_grid(theta, input.sizes(),
                torch::nn::functional::AffineGridFuncOptions().align_corners(false));

            return torch::nn::functional::grid_sample(input, grid,
                torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false));
        }

        return input;
    }
};

struct ElasticDeformation : public torch::data::transforms::TensorTransform<torch::Tensor> {
    float alpha;      // Deformation strength
    float sigma;      // Smoothness of deformation
    int grid_size;    // Grid resolution for displacement field

    ElasticDeformation(float alpha = 1.0, float sigma = 0.1, int grid_size = 3)
        : alpha(alpha), sigma(sigma), grid_size(grid_size) {}

    torch::Tensor operator()(torch::Tensor input) override {
        auto sizes = input.sizes();
        int height = sizes[sizes.size() - 2];
        int width = sizes[sizes.size() - 1];

        // Create random displacement field
        auto dx = torch::randn({grid_size, grid_size}) * alpha;
        auto dy = torch::randn({grid_size, grid_size}) * alpha;

        // Smooth the displacement field using Gaussian filter
        dx = gaussian_smooth(dx, sigma);
        dy = gaussian_smooth(dy, sigma);

        // Resize displacement field to match image size
        dx = torch::nn::functional::interpolate(dx.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(false)).squeeze(0).squeeze(0);

        dy = torch::nn::functional::interpolate(dy.unsqueeze(0).unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{height, width})
            .mode(torch::kBilinear)
            .align_corners(false)).squeeze(0).squeeze(0);

        // Create coordinate grids
        auto y_coords = torch::arange(height, torch::kFloat32);
        auto x_coords = torch::arange(width, torch::kFloat32);
        auto grid_y = y_coords.view({-1, 1}).expand({height, width});
        auto grid_x = x_coords.view({1, -1}).expand({height, width});

        // Apply displacement
        grid_x = grid_x + dx;
        grid_y = grid_y + dy;

        // Normalize coordinates to [-1, 1] range for grid_sample
        grid_x = 2.0 * grid_x / (width - 1) - 1.0;
        grid_y = 2.0 * grid_y / (height - 1) - 1.0;

        // Stack coordinates (note: grid_sample expects [x, y] order)
        auto grid = torch::stack({grid_x, grid_y}, -1).unsqueeze(0);

        // Handle different input dimensions
        if (input.dim() == 3) {
            // Add batch dimension
            input = input.unsqueeze(0);

            auto deformed = torch::nn::functional::grid_sample(input, grid,
                torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false));

            return deformed.squeeze(0);
        } else if (input.dim() == 4) {
            // Batch processing - expand grid for batch
            int batch_size = input.size(0);
            grid = grid.expand({batch_size, -1, -1, -1});

            return torch::nn::functional::grid_sample(input, grid,
                torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .padding_mode(torch::kZeros)
                .align_corners(false));
        }

        return input;
    }

private:
    torch::Tensor gaussian_smooth(torch::Tensor input, float sigma) {
        // Simple Gaussian smoothing using a 3x3 kernel
        int kernel_size = 3;
        float sum = 0.0;

        // Create Gaussian kernel
        auto kernel = torch::zeros({kernel_size, kernel_size});
        int center = kernel_size / 2;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                float distance = std::sqrt((i - center) * (i - center) + (j - center) * (j - center));
                float value = std::exp(-(distance * distance) / (2 * sigma * sigma));
                kernel[i][j] = value;
                sum += value;
            }
        }

        // Normalize kernel
        kernel = kernel / sum;

        // Apply convolution
        auto input_4d = input.unsqueeze(0).unsqueeze(0);
        auto kernel_4d = kernel.unsqueeze(0).unsqueeze(0);

        auto smoothed = torch::nn::functional::conv2d(input_4d, kernel_4d,
            torch::nn::functional::Conv2dFuncOptions().padding(1));

        return smoothed.squeeze(0).squeeze(0);
    }
};

// Intensity augmentation for medical images
struct RandomIntensity : public torch::data::transforms::TensorTransform<torch::Tensor> {
    float brightness_range;
    float contrast_range;

    RandomIntensity(float brightness = 0.1, float contrast = 0.1)
        : brightness_range(brightness), contrast_range(contrast) {}

    torch::Tensor operator()(torch::Tensor input) override {
        // Random brightness adjustment
        float brightness_factor = 1.0 + (torch::rand({1}).item<float>() - 0.5) * 2.0 * brightness_range;

        // Random contrast adjustment
        float contrast_factor = 1.0 + (torch::rand({1}).item<float>() - 0.5) * 2.0 * contrast_range;

        // Apply transformations
        auto mean_val = torch::mean(input);
        input = input * contrast_factor + brightness_factor - contrast_factor * mean_val;

        // Clamp values to valid range (assuming normalized input)
        return torch::clamp(input, 0.0, 1.0);
    }
};

// Noise augmentation for medical images
struct RandomNoise : public torch::data::transforms::TensorTransform<torch::Tensor> {
    float noise_std;

    RandomNoise(float std = 0.01) : noise_std(std) {}

    torch::Tensor operator()(torch::Tensor input) override {
        if (torch::rand({1}).item<float>() > 0.5) {
            auto noise = torch::randn_like(input) * noise_std;
            input = input + noise;
            return torch::clamp(input, 0.0, 1.0);
        }
        return input;
    }
};

// Example usage class for combining transforms
class MedicalImageAugmentation {
public:
    std::vector<std::unique_ptr<torch::data::transforms::TensorTransform<torch::Tensor>>> transforms;

    MedicalImageAugmentation() {
        // Add transforms in desired order
        transforms.push_back(std::make_unique<RandomFlip>());
        transforms.push_back(std::make_unique<RandomRotate>(15.0));
        transforms.push_back(std::make_unique<ElasticDeformation>(1.0, 0.1, 3));
        transforms.push_back(std::make_unique<RandomIntensity>(0.1, 0.1));
        transforms.push_back(std::make_unique<RandomNoise>(0.01));
    }

    torch::Tensor apply(torch::Tensor input) {
        for (auto& transform : transforms) {
            input = (*transform)(input);
        }
        return input;
    }
};
#endif //DATA_AUGMENTATION_HPP
