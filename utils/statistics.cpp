//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#include "statistics.hpp"

::torch::Tensor calculate_iou(const ::torch::Tensor &pred, const ::torch::Tensor &target, int num_classes) {
    auto iou_scores = torch::zeros({num_classes});

    for (int c = 0; c < num_classes; ++c) {
        auto pred_mask = (pred == c);
        auto target_mask = (target == c);

        auto intersection = (pred_mask & target_mask).sum();

        if (auto union_area = (pred_mask | target_mask).sum(); union_area.item<float>() > 0) {
            iou_scores[c] = intersection.item<float>() / union_area.item<float>();
        }
    }

    return iou_scores;
}

::torch::Tensor calculate_dice(const ::torch::Tensor &pred, const ::torch::Tensor &target, int num_classes) {
    auto dice_scores = torch::zeros({num_classes});

    for (int c = 0; c < num_classes; ++c) {
        auto pred_mask = (pred == c);
        auto target_mask = (target == c);

        auto intersection = (pred_mask & target_mask).sum();

        if (auto total = pred_mask.sum() + target_mask.sum(); total.item<float>() > 0) {
            dice_scores[c] = (2.0 * intersection.item<float>()) / total.item<float>();
        }
    }

    return dice_scores;
}