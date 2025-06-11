//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <torch/torch.h>

// Utility functions for metrics calculation
::torch::Tensor calculate_iou(const ::torch::Tensor &pred, const ::torch::Tensor &target, int num_classes);

::torch::Tensor calculate_dice(const ::torch::Tensor &pred, const ::torch::Tensor &target, int num_classes);

#endif //STATISTICS_HPP
