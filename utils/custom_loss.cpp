//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#include "utils/custom_loss.hpp"
#include <torch/torch.h>

torch::Tensor DiceLoss::forward(torch::Tensor pred, torch::Tensor target) {
    // pred: [B, C, H, W] - logits
    // target: [B, H, W] - class indices

    auto batch_size = pred.size(0);
    auto num_classes = pred.size(1);
    auto pred_soft = torch::softmax(pred, 1);

    // Convert target to one-hot encoding
    auto target_one_hot = torch::zeros_like(pred_soft);
    target_one_hot.scatter_(1, target.unsqueeze(1), 1);

    // Calculate dice per sample, then average
    auto total_dice_loss = torch::zeros({1}, pred.device());

    // Process each sample in the batch separately
    for (int b = 0; b < batch_size; ++b) {
        auto sample_dice_loss = torch::zeros({1}, pred.device());
        int valid_classes = 0;

        // Skip background class (index 0)
        for (int c = 1; c < num_classes; ++c) {
            // Get predictions and targets for this sample and class
            auto pred_c = pred_soft[b][c].contiguous().view({-1});
            auto target_c = target_one_hot[b][c].contiguous().view({-1});

            auto intersection = (pred_c * target_c).sum();
            auto pred_sum = pred_c.sum();
            auto target_sum = target_c.sum();

            // Skip if class not present in ground truth AND predictions
            if (pred_sum.item<float>() == 0 && target_sum.item<float>() == 0) {
                continue;  // Don't penalize for correctly predicting absence
            }

            valid_classes++;

            if (smooth_) {
                auto dice_score = (2.0 * intersection + smooth_val_) /
                                 (pred_sum + target_sum + smooth_val_);
                sample_dice_loss += 1.0 - dice_score;
            } else {
                auto dice_score = (2.0 * intersection) /
                                 (pred_sum + target_sum + 1e-7);
                sample_dice_loss += 1.0 - dice_score;
            }
        }

        // Average over valid classes for this sample
        if (valid_classes > 0) {
            total_dice_loss += sample_dice_loss / valid_classes;
        }
    }

    // Average over batch
    return total_dice_loss / batch_size;
}

// Alternative: More efficient vectorized implementation
torch::Tensor DiceLossVectorized::forward(torch::Tensor pred, torch::Tensor target) {
    // pred: [B, C, H, W] - logits
    // target: [B, H, W] - class indices

    auto num_classes = pred.size(1);
    auto pred_soft = torch::softmax(pred, 1);

    // Convert target to one-hot encoding
    auto target_one_hot = torch::zeros_like(pred_soft);
    target_one_hot.scatter_(1, target.unsqueeze(1), 1);

    // Flatten spatial dimensions but keep batch and class separate
    // [B, C, H, W] -> [B, C, H*W]
    pred_soft = pred_soft.view({pred_soft.size(0), pred_soft.size(1), -1});
    target_one_hot = target_one_hot.view({target_one_hot.size(0), target_one_hot.size(1), -1});

    // Calculate intersection and cardinality per sample per class
    // [B, C]
    auto intersection = (pred_soft * target_one_hot).sum(2);
    auto pred_sum = pred_soft.sum(2);
    auto target_sum = target_one_hot.sum(2);

    // Dice score per sample per class
    torch::Tensor dice_score;
    if (smooth_) {
        dice_score = (2.0 * intersection + smooth_val_) /
                     (pred_sum + target_sum + smooth_val_);
    } else {
        dice_score = (2.0 * intersection + 1e-7) /
                     (pred_sum + target_sum + 1e-7);
    }

    // Dice loss
    auto dice_loss = 1.0 - dice_score;

    // Handle empty classes (where both pred and target sum are 0)
    auto empty_classes = (pred_sum == 0) & (target_sum == 0);
    dice_loss.masked_fill_(empty_classes, 0);  // No loss for correctly absent classes

    // Average over non-background classes (1:C) and batch
    // Count valid classes per sample - Fixed: use .to(torch::kFloat) instead of .float()
    auto valid_classes = (~empty_classes).slice(1, 1).sum(1, true).to(torch::kFloat);
    valid_classes = torch::clamp(valid_classes, 1.0);  // Avoid division by zero

    // Sum losses for non-background classes and divide by valid classes
    auto loss_per_sample = dice_loss.slice(1, 1).sum(1, true) / valid_classes;

    // Average over batch
    return loss_per_sample.mean();
}

// Focal Dice Loss for handling class imbalance better
torch::Tensor FocalDiceLoss::forward(torch::Tensor pred, torch::Tensor target,
                                    float gamma = 2.0) {
    // Similar to dice loss but with focal weighting
    auto batch_size = pred.size(0);
    auto num_classes = pred.size(1);
    auto pred_soft = torch::softmax(pred, 1);

    auto target_one_hot = torch::zeros_like(pred_soft);
    target_one_hot.scatter_(1, target.unsqueeze(1), 1);

    // Flatten spatial dimensions
    pred_soft = pred_soft.view({batch_size, num_classes, -1});
    target_one_hot = target_one_hot.view({batch_size, num_classes, -1});

    // Calculate dice components
    auto intersection = (pred_soft * target_one_hot).sum(2);
    auto pred_sum = pred_soft.sum(2);
    auto target_sum = target_one_hot.sum(2);

    // Dice score
    auto dice_score = (2.0 * intersection + smooth_val_) /
                      (pred_sum + target_sum + smooth_val_);

    // Focal weighting: give more weight to hard examples
    auto focal_weight = torch::pow(1.0 - dice_score, gamma);

    // Weighted dice loss
    auto dice_loss = focal_weight * (1.0 - dice_score);

    // Skip background and average properly
    auto loss_per_sample = dice_loss.slice(1, 1).mean(1);
    return loss_per_sample.mean();
}

// Generalized Dice Loss (handles class imbalance)
torch::Tensor GeneralizedDiceLoss::forward(torch::Tensor pred, torch::Tensor target) {
    auto batch_size = pred.size(0);
    auto num_classes = pred.size(1);
    auto pred_soft = torch::softmax(pred, 1);

    auto target_one_hot = torch::zeros_like(pred_soft);
    target_one_hot.scatter_(1, target.unsqueeze(1), 1);

    // Flatten spatial dimensions
    pred_soft = pred_soft.view({batch_size, num_classes, -1});
    target_one_hot = target_one_hot.view({batch_size, num_classes, -1});

    // Calculate class weights (inverse of frequency)
    // Fixed: use explicit IntArrayRef instead of initializer list
    auto dims = torch::IntArrayRef({0, 2});
    auto target_sum_per_class = target_one_hot.sum(dims);  // Sum over batch and spatial
    auto weights = 1.0 / (target_sum_per_class.square() + 1.0);

    // Normalize weights
    weights = weights / weights.sum();

    // Calculate weighted intersection and sums
    auto intersection = (pred_soft * target_one_hot).sum(2);  // [B, C]
    auto pred_sum = pred_soft.sum(2);
    auto target_sum = target_one_hot.sum(2);

    // Expand weights for broadcasting
    weights = weights.unsqueeze(0);  // [1, C]

    // Weighted dice calculation
    auto numerator = 2.0 * (weights * intersection).sum(1);
    auto denominator = (weights * (pred_sum + target_sum)).sum(1);

    auto gdl = 1.0 - (numerator + smooth_val_) / (denominator + smooth_val_);

    return gdl.mean();
}

// Combined Loss with proper dice
torch::Tensor CombinedLoss::forward(torch::Tensor pred, torch::Tensor target) {
    // Optional: Add class weights for CE loss if there's class imbalance
    torch::Tensor class_weights;
    if (use_class_weights_) {
        // Calculate class frequencies in target
        auto num_classes = pred.size(1);
        class_weights = torch::zeros({num_classes}, pred.device());

        for (int c = 0; c < num_classes; ++c) {
            // Fixed: use .to(torch::kFloat) instead of .float()
            class_weights[c] = (target == c).sum().to(torch::kFloat);
        }

        // Inverse frequency weighting
        class_weights = 1.0 / (class_weights + 1.0);
        class_weights = class_weights / class_weights.sum() * num_classes;
    }

    // Cross entropy with optional class weights
    auto ce_loss = torch::nn::functional::cross_entropy(
        pred, target,
        torch::nn::functional::CrossEntropyFuncOptions().weight(class_weights));

    // Use vectorized dice loss for efficiency
    auto dice_loss = dice_loss_vectorized_.forward(pred, target);

    // Optional: Add boundary loss for better boundary delineation
    // Fixed: initialize as proper tensor
    auto boundary_loss = torch::zeros({1}, pred.device());
    if (use_boundary_loss_) {
        boundary_loss = calculate_boundary_loss(pred, target);
    }

    return ce_weight_ * ce_loss +
           dice_weight_ * dice_loss +
           boundary_weight_ * boundary_loss;
}

// Helper: Boundary loss for better edge detection
torch::Tensor calculate_boundary_loss(torch::Tensor pred, torch::Tensor target) {
    auto batch_size = pred.size(0);
    auto num_classes = pred.size(1);
    auto height = pred.size(2);
    auto width = pred.size(3);

    // Convert predictions to softmax
    auto pred_soft = torch::softmax(pred, 1);

    // Convert target to one-hot encoding
    auto target_one_hot = torch::zeros_like(pred_soft);
    target_one_hot.scatter_(1, target.unsqueeze(1), 1);

    // Calculate boundaries using gradient-based edge detection
    auto boundaries = torch::zeros_like(target_one_hot);

    // Sobel kernels for edge detection
    auto sobel_x = torch::tensor({{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}},
                                torch::TensorOptions().dtype(torch::kFloat32).device(pred.device()))
                   .view({1, 1, 3, 3});
    auto sobel_y = torch::tensor({{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}},
                                torch::TensorOptions().dtype(torch::kFloat32).device(pred.device()))
                   .view({1, 1, 3, 3});

    for (int c = 1; c < num_classes; ++c) { // Skip background
        auto class_mask = target_one_hot.slice(1, c, c+1);

        // Calculate gradients
        auto grad_x = torch::nn::functional::conv2d(
            class_mask, sobel_x,
            torch::nn::functional::Conv2dFuncOptions().padding(1));
        auto grad_y = torch::nn::functional::conv2d(
            class_mask, sobel_y,
            torch::nn::functional::Conv2dFuncOptions().padding(1));

        // Edge magnitude
        auto edges = torch::sqrt(grad_x.pow(2) + grad_y.pow(2));

        // Threshold to get binary boundary mask
        boundaries.slice(1, c, c+1) = (edges > 0.1).to(torch::kFloat32);
    }

    // Calculate boundary loss only on boundary pixels
    auto boundary_pixels = boundaries.sum({1}, true); // Sum over classes, keep dims
    auto valid_boundaries = (boundary_pixels > 0).to(torch::kFloat32);

    // Cross-entropy loss weighted by boundary importance
    auto log_pred = torch::log(pred_soft + 1e-7);
    auto boundary_ce = -(boundaries * target_one_hot * log_pred).sum({1}); // Sum over classes

    // Normalize by number of boundary pixels per sample
    auto num_boundary_pixels = boundaries.sum({1, 2, 3}); // [B]
    num_boundary_pixels = torch::clamp(num_boundary_pixels, 1.0); // Avoid division by zero

    auto normalized_loss = boundary_ce.sum({1, 2}) / num_boundary_pixels; // [B]

    return normalized_loss.mean();
}
