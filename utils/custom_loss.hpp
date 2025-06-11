#ifndef CUSTOM_LOSS_HPP
#define CUSTOM_LOSS_HPP

#include <torch/torch.h>

// Base Dice Loss
class DiceLoss : public torch::nn::Module {
public:
    DiceLoss(bool smooth = true, float smooth_val = 1e-5)
        : smooth_(smooth), smooth_val_(smooth_val) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

protected:
    bool smooth_;
    float smooth_val_;
};

// Vectorized Dice Loss (more efficient)
class DiceLossVectorized : public torch::nn::Module {
public:
    DiceLossVectorized(bool smooth = true, float smooth_val = 1e-5)
        : smooth_(smooth), smooth_val_(smooth_val) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

protected:
    bool smooth_;
    float smooth_val_;
};

// Focal Dice Loss for handling hard examples
class FocalDiceLoss : public torch::nn::Module {
public:
    FocalDiceLoss(bool smooth = true, float smooth_val = 1e-5, float gamma = 2.0)
        : smooth_(smooth), smooth_val_(smooth_val), gamma_(gamma) {}

    // FIXED: Default parameter should only be in implementation, not header
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, float gamma);

protected:
    bool smooth_;
    float smooth_val_;
    float gamma_;
};

// Generalized Dice Loss for handling class imbalance
class GeneralizedDiceLoss : public torch::nn::Module {
public:
    GeneralizedDiceLoss(bool smooth = true, float smooth_val = 1e-5)
        : smooth_(smooth), smooth_val_(smooth_val) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

protected:
    bool smooth_;
    float smooth_val_;
};

// Combined Loss with multiple components
class CombinedLoss : public torch::nn::Module {
public:
    CombinedLoss(float ce_weight = 0.6,
                 float dice_weight = 0.4,
                 float boundary_weight = 0.0,
                 bool use_class_weights = false,
                 bool use_boundary_loss = false)
        : ce_weight_(ce_weight),
          dice_weight_(dice_weight),
          boundary_weight_(boundary_weight),
          use_class_weights_(use_class_weights),
          use_boundary_loss_(use_boundary_loss) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

    // Allow dynamic weight adjustment during training
    void set_weights(float ce_weight, float dice_weight, float boundary_weight = 0.0) {
        ce_weight_ = ce_weight;
        dice_weight_ = dice_weight;
        boundary_weight_ = boundary_weight;
    }

    void enable_class_weights(bool enable) { use_class_weights_ = enable; }
    void enable_boundary_loss(bool enable) { use_boundary_loss_ = enable; }

protected:
    float ce_weight_;
    float dice_weight_;
    float boundary_weight_;
    bool use_class_weights_;
    bool use_boundary_loss_;

    // Use vectorized dice loss for efficiency
    DiceLossVectorized dice_loss_vectorized_;

};

// Simple ready-to-use loss functions
class SimpleDiceCELoss : public torch::nn::Module {
public:
    SimpleDiceCELoss(float dice_weight = 0.4, float ce_weight = 0.6)
        : dice_weight_(dice_weight), ce_weight_(ce_weight) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
        auto ce_loss = torch::nn::functional::cross_entropy(pred, target);
        auto dice_loss = dice_loss_.forward(pred, target);
        return ce_weight_ * ce_loss + dice_weight_ * dice_loss;
    }

    void set_weights(float dice_weight, float ce_weight) {
        dice_weight_ = dice_weight;
        ce_weight_ = ce_weight;
    }

private:
    float dice_weight_;
    float ce_weight_;
    DiceLossVectorized dice_loss_;
};

// Focal Loss for handling class imbalance
class FocalLoss : public torch::nn::Module {
public:
    FocalLoss(float alpha = 0.25, float gamma = 2.0, bool size_average = true)
        : alpha_(alpha), gamma_(gamma), size_average_(size_average) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

protected:
    float alpha_;
    float gamma_;
    bool size_average_;
};

// Tversky Loss (generalization of Dice with adjustable FP/FN weights)
class TverskyLoss : public torch::nn::Module {
public:
    TverskyLoss(float alpha = 0.5, float beta = 0.5, bool smooth = true, float smooth_val = 1e-5)
        : alpha_(alpha), beta_(beta), smooth_(smooth), smooth_val_(smooth_val) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

protected:
    float alpha_;  // Weight for false positives
    float beta_;   // Weight for false negatives
    bool smooth_;
    float smooth_val_;
};

// Multi-component loss with focal enhancement
class AdvancedCombinedLoss : public torch::nn::Module {
public:
    AdvancedCombinedLoss(float ce_weight = 0.3,
                        float dice_weight = 0.4,
                        float focal_weight = 0.3,
                        float focal_alpha = 0.25,
                        float focal_gamma = 2.0)
        : ce_weight_(ce_weight),
          dice_weight_(dice_weight),
          focal_weight_(focal_weight),
          focal_loss_(focal_alpha, focal_gamma) {}

    torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
        auto ce_loss = torch::nn::functional::cross_entropy(pred, target);
        auto dice_loss = dice_loss_.forward(pred, target);
        auto focal_loss = focal_loss_.forward(pred, target);

        return ce_weight_ * ce_loss +
               dice_weight_ * dice_loss +
               focal_weight_ * focal_loss;
    }

    void set_weights(float ce_weight, float dice_weight, float focal_weight) {
        ce_weight_ = ce_weight;
        dice_weight_ = dice_weight;
        focal_weight_ = focal_weight;
    }

private:
    float ce_weight_;
    float dice_weight_;
    float focal_weight_;
    DiceLossVectorized dice_loss_;
    FocalLoss focal_loss_;
};

// Utility functions for loss computation
namespace loss_utils {
    // Convert class indices to one-hot encoding
    torch::Tensor to_one_hot(torch::Tensor target, int num_classes);

    // Calculate class weights based on frequency
    torch::Tensor calculate_class_weights(torch::Tensor target, int num_classes);

    // Compute soft IoU for differentiable IoU loss
    torch::Tensor soft_iou(torch::Tensor pred, torch::Tensor target, float smooth = 1e-5);

    // Helper for stable log computation
    torch::Tensor stable_log(torch::Tensor x, float eps = 1e-7);
}

// Forward declarations for helper functions
torch::Tensor calculate_boundary_loss(torch::Tensor pred, torch::Tensor target);



#endif // CUSTOM_LOSS_HPP