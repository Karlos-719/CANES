// =============================================================================
// acdc_dataset.hpp - Updated header to match Python implementation
// =============================================================================

#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <H5Cpp.h>
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <regex>
#include <map>
#include <random>
#include <algorithm>

enum class Mode {
    TRAIN,
    VAL,
    TEST
};

struct Sample {
    std::string file_path;
    int slice_index;  // -1 for 2D data, >= 0 for 3D slice index
    std::string patient_id;
};

class ACDCDataset : public torch::data::datasets::Dataset<ACDCDataset> {
public:
    ACDCDataset(const std::string& root, Mode mode = Mode::TRAIN,
                torch::IntArrayRef image_size = {256, 256});

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
    std::vector<Sample> samples_;
private:
    void load_samples();
    std::pair<torch::Tensor, torch::Tensor> load_h5_slice(const std::string& path, int slice_index);

    // UPDATED: Two normalization methods
    torch::Tensor normalize_image(const torch::Tensor& image);  // Original min-max
    torch::Tensor normalize_image_python_style(const torch::Tensor& image);  // Python style (divide by max)

    torch::Tensor resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& target_size);

    Mode mode_;
    std::vector<int64_t> image_size_;
    std::string root_;
    
    bool is_training_data_;
};