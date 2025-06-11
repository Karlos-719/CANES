// =============================================================================
// acdc_dataset.cpp - Fixed to match Python implementation
// =============================================================================

#include "acdc_dataset.hpp"

std::vector<std::string> patient_keys;

void fill_patient_keys(auto & patient_samples) {
    for (const auto& pair : patient_samples) {
        patient_keys.push_back(pair.first);
    }

    // Shuffle patients (not individual slices!)
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::shuffle(patient_keys.begin(), patient_keys.end(), rng);

}

ACDCDataset::ACDCDataset(const std::string& root, Mode mode, torch::IntArrayRef image_size)
    : mode_(mode), image_size_(image_size.begin(), image_size.end()), root_(root) {

    // Determine if we're loading training or testing data based on directory name
    is_training_data_ = true;

    load_samples();
    std::cout << "Loaded " << samples_.size() << " samples for "
              << (mode == Mode::TRAIN ? "training" : (mode == Mode::VAL ? "validation" : "testing"))
              << std::endl;
}

torch::data::Example<> ACDCDataset::get(size_t index) {
    const auto& sample = samples_[index];

    auto [image, mask] = load_h5_slice(sample.file_path, sample.slice_index);

    // Normalize image to [0, 1] - FIXED: Now matches Python approach
    image = normalize_image_python_style(image);

    // Resize if needed - matches Python target_size=(256, 256)
    if (!image_size_.empty()) {
        image = resize_tensor(image, image_size_);
        if (is_training_data_) {
            mask = resize_tensor(mask, image_size_);
        }
    }

    // Ensure proper dimensions: [C, H, W] for image
    if (image.dim() == 2) {
        image = image.unsqueeze(0); // Add channel dimension
    }

    // For test data without masks, return dummy mask
    if (!is_training_data_) {
        mask = torch::zeros({image.size(1), image.size(2)}, torch::kLong);
    }

    return {image, mask.to(torch::kLong)};
}

torch::optional<size_t> ACDCDataset::size() const {
    return samples_.size();
}

void ACDCDataset::load_samples() {
    samples_.clear();

    if (!std::filesystem::exists(root_)) {
        std::cerr << "Error: ACDC dataset path does not exist: " << root_ << std::endl;
        return;
    }

    std::vector<std::string> h5_files;

    // Scan for H5 files - same as before
    if (std::filesystem::is_directory(root_)) {
        std::cout << "Scanning directory: " << root_ << std::endl;

        for (const auto& entry : std::filesystem::recursive_directory_iterator(root_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".h5") {
                h5_files.push_back(entry.path().string());
                //std::cout << "Found H5 file: " << entry.path().filename() << std::endl;
            }
        }
    } else if (std::filesystem::is_regular_file(root_) && std::filesystem::path(root_).extension() == ".h5") {
        h5_files.push_back(root_);
    } else {
        std::cerr << "Error: " << root_ << " is neither a directory nor an H5 file" << std::endl;
        return;
    }

    // Sort files for reproducibility
    std::sort(h5_files.begin(), h5_files.end());
    std::cout << "Total H5 files found: " << h5_files.size() << std::endl;

    // FIXED: First collect all samples grouped by patient
    std::map<std::string, std::vector<Sample>> patient_samples;

    // Process each H5 file
    for (const auto& file_path : h5_files) {
        try {
            H5::H5File file(file_path, H5F_ACC_RDONLY);

            if (!H5Lexists(file.getId(), "image", H5P_DEFAULT)) {
                std::cerr << "Warning: 'image' dataset not found in " << file_path << std::endl;
                file.close();
                continue;
            }

            H5::DataSet image_dataset = file.openDataSet("image");
            H5::DataSpace dataspace = image_dataset.getSpace();
            int ndims = dataspace.getSimpleExtentNdims();
            std::vector<hsize_t> dims(ndims);
            dataspace.getSimpleExtentDims(dims.data());

            // Extract patient info from filename
            std::string filename = std::filesystem::path(file_path).filename().string();
            std::regex patient_pattern(R"(patient(\d+)_frame(\d+))");
            std::smatch match;
            std::string patient_id = "unknown";
            std::string frame_id = "unknown";
            if (std::regex_search(filename, match, patient_pattern)) {
                patient_id = match[1].str();
                frame_id = match[2].str();
            }

            // FIXED: Use patient+frame as the key (without slice info)
            std::string patient_key = patient_id + "_frame" + frame_id;

            if (ndims == 3) {  // 3D volume (slices, H, W)
                std::cout << "Loading 3D volume: " << filename << " with shape ["
                          << dims[0] << ", " << dims[1] << ", " << dims[2] << "]" << std::endl;

                // Create a sample for each slice, but group by patient
                for (int slice_idx = 0; slice_idx < dims[0]; ++slice_idx) {
                    Sample sample;
                    sample.file_path = file_path;
                    sample.slice_index = slice_idx;
                    sample.patient_id = patient_key + "_slice" + std::to_string(slice_idx);

                    // Group by patient (without slice info)
                    patient_samples[patient_id].push_back(sample);
                }
            } else if (ndims == 2) {  // 2D image (H, W)
                // std::cout << "Loading 2D image: " << filename << " with shape ["
                //           << dims[0] << ", " << dims[1] << "]" << std::endl;

                Sample sample;
                sample.file_path = file_path;
                sample.slice_index = -1;  // Indicate 2D data
                sample.patient_id = patient_key;

                // Group by patient
                patient_samples[patient_id].push_back(sample);
            } else {
                std::cerr << "Unexpected dimensions in " << file_path << ": " << ndims << "D" << std::endl;
            }

            file.close();
        } catch (const H5::Exception& e) {
            std::cerr << "Error loading " << file_path << ": " << e.getCDetailMsg() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading " << file_path << ": " << e.what() << std::endl;
        }
    }

    // FIXED: Patient-level train/val split
    if (is_training_data_ && (mode_ == Mode::TRAIN || mode_ == Mode::VAL)) {
        // Get list of unique patients
        if (patient_keys.empty()) {
            fill_patient_keys(patient_samples);
        }

        size_t total_patients = patient_keys.size();
        size_t train_patients = static_cast<size_t>(total_patients * 0.8);

        std::cout << "Total patients: " << total_patients << std::endl;
        std::cout << "Train patients: " << train_patients
                  << ", Val patients: " << (total_patients - train_patients) << std::endl;

        // Collect samples based on patient split
        std::vector<std::string> selected_patients;
        if (mode_ == Mode::TRAIN) {
            // First 80% of patients for training
            selected_patients.assign(patient_keys.begin(),
                                   patient_keys.begin() + train_patients);
        } else {  // VAL
            // Last 20% of patients for validation
            selected_patients.assign(patient_keys.begin() + train_patients,
                                   patient_keys.end());
        }

        // Add all samples from selected patients
        for (const auto& patient_key : selected_patients) {
            for (const auto& sample : patient_samples[patient_key]) {
                samples_.push_back(sample);
            }
        }

        std::cout << "Split: " << samples_.size() << " samples for "
                  << (mode_ == Mode::TRAIN ? "training" : "validation")
                  << " from " << selected_patients.size() << " patients" << std::endl;

        // Debug: Print patient distribution
        std::cout << "Patients in " << (mode_ == Mode::TRAIN ? "training" : "validation") << ": ";
        for (size_t i = 0; i < std::min(selected_patients.size(), size_t(100)); ++i) {
            std::cout << selected_patients[i] << " ";
        }
        if (selected_patients.size() > 100) {
            std::cout << "... (and " << (selected_patients.size() - 100) << " more)";
        }
        std::cout << std::endl;

    } else {
        // For test mode, use all samples
        for (const auto& pair : patient_samples) {
            for (const auto& sample : pair.second) {
                samples_.push_back(sample);
            }
        }
    }
}
std::pair<torch::Tensor, torch::Tensor> ACDCDataset::load_h5_slice(const std::string& path, int slice_index) {
    try {
        H5::H5File file(path, H5F_ACC_RDONLY);

        // Load image
        H5::DataSet image_dataset = file.openDataSet("image");
        H5::DataSpace image_space = image_dataset.getSpace();
        int ndims = image_space.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        image_space.getSimpleExtentDims(dims.data());

        torch::Tensor image;
        torch::Tensor mask;

        if (ndims == 3 && slice_index >= 0) {  // 3D volume, extract slice - MATCHES Python
            // Read specific slice
            hsize_t offset[3] = {static_cast<hsize_t>(slice_index), 0, 0};
            hsize_t count[3] = {1, dims[1], dims[2]};
            image_space.selectHyperslab(H5S_SELECT_SET, count, offset);

            // Create memory space for slice
            hsize_t mem_dims[2] = {dims[1], dims[2]};
            H5::DataSpace mem_space(2, mem_dims);

            // Read image slice
            std::vector<float> image_data(dims[1] * dims[2]);
            image_dataset.read(image_data.data(), H5::PredType::NATIVE_FLOAT, mem_space, image_space);
            image = torch::from_blob(image_data.data(), {static_cast<int64_t>(dims[1]),
                                                        static_cast<int64_t>(dims[2])}, torch::kFloat32).clone();

            // Read mask if training data and label exists
            if (is_training_data_ && H5Lexists(file.getId(), "label", H5P_DEFAULT)) {
                H5::DataSet mask_dataset = file.openDataSet("label");
                H5::DataSpace mask_space = mask_dataset.getSpace();
                mask_space.selectHyperslab(H5S_SELECT_SET, count, offset);

                std::vector<uint8_t> mask_data(dims[1] * dims[2]);
                mask_dataset.read(mask_data.data(), H5::PredType::NATIVE_UINT8, mem_space, mask_space);
                mask = torch::from_blob(mask_data.data(), {static_cast<int64_t>(dims[1]),
                                                          static_cast<int64_t>(dims[2])}, torch::kUInt8).clone();

                // Print basic tensor info
                // std::cout << "Mask shape: " << mask.sizes() << std::endl;
                // std::cout << "Mask dtype: " << mask.dtype() << std::endl;
                // std::cout << "Mask device: " << mask.device() << std::endl;

                // Using torch::_unique (note the underscore)
                // auto unique_result = torch::_unique(mask);
                // auto unique_values = std::get<0>(unique_result);  // First element is the unique values
                //
                // std::cout << "Unique values in mask: ";
                // for (int i = 0; i < unique_values.size(0); ++i) {
                //     std::cout << static_cast<int>(unique_values[i].item<uint8_t>()) << " ";
                // }
                // std::cout << std::endl;
            }

        } else if (ndims == 2) {  // 2D image - MATCHES Python
            std::vector<float> image_data(dims[0] * dims[1]);
            image_dataset.read(image_data.data(), H5::PredType::NATIVE_FLOAT);
            image = torch::from_blob(image_data.data(), {static_cast<int64_t>(dims[0]),
                                                        static_cast<int64_t>(dims[1])}, torch::kFloat32).clone();

            // Read mask if training data and label exists
            if (is_training_data_ && H5Lexists(file.getId(), "label", H5P_DEFAULT)) {
                H5::DataSet mask_dataset = file.openDataSet("label");
                std::vector<uint8_t> mask_data(dims[0] * dims[1]);
                mask_dataset.read(mask_data.data(), H5::PredType::NATIVE_UINT8);
                mask = torch::from_blob(mask_data.data(), {static_cast<int64_t>(dims[0]),
                                                          static_cast<int64_t>(dims[1])}, torch::kUInt8).clone();
            }
        }

        file.close();

        // Return empty mask for test data or if no label found
        if (!is_training_data_ || mask.numel() == 0) {
            mask = torch::zeros_like(image, torch::kUInt8);
        }

        return {image, mask};

    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to load H5 file " + path + ": " + e.getCDetailMsg());
    }
}

// FIXED: New normalization method to match Python exactly
torch::Tensor ACDCDataset::normalize_image_python_style(const torch::Tensor& image) {
    // Python: train_images = train_images / np.max(train_images)
    // This normalizes by dividing by the maximum value in the image
    auto max_val = image.max();

    if (max_val.item<float>() > 0) {
        return image / max_val;
    }
    return image;
}

// Keep the old normalize method as backup
torch::Tensor ACDCDataset::normalize_image(const torch::Tensor& image) {
    // Original min-max normalization
    auto min_val = image.min();
    auto max_val = image.max();

    if ((max_val - min_val).item<float>() > 0) {
        return (image - min_val) / (max_val - min_val);
    }
    return image;
}

torch::Tensor ACDCDataset::resize_tensor(const torch::Tensor& tensor, const std::vector<int64_t>& target_size) {
    if (target_size.size() != 2) {
        throw std::runtime_error("Target size must be 2D for image resizing");
    }

    auto input = tensor;
    bool is_mask = (tensor.dtype() == torch::kUInt8 || tensor.dtype() == torch::kLong);

    // Convert to float for interpolation
    if (is_mask) {
        input = input.to(torch::kFloat32);
    }

    // Add batch and channel dimensions if needed
    if (input.dim() == 2) {
        input = input.unsqueeze(0).unsqueeze(0); // Add batch and channel dims
    } else if (input.dim() == 3) {
        input = input.unsqueeze(0); // Add batch dim
    }

    // Resize using appropriate interpolation - MATCHES Python tf.image.resize
    torch::Tensor resized;
    if (is_mask) {
        // Use nearest neighbor for masks (matches Python method='nearest')
        resized = torch::nn::functional::interpolate(
            input,
            torch::nn::functional::InterpolateFuncOptions()
                .size(target_size)
                .mode(torch::kNearest)
        );
    } else {
        // Use bilinear for images (matches Python method='bilinear')
        resized = torch::nn::functional::interpolate(
            input,
            torch::nn::functional::InterpolateFuncOptions()
                .size(target_size)
                .mode(torch::kBilinear)
                .align_corners(false)
        );
    }

    // Remove added dimensions
    while (resized.dim() > tensor.dim()) {
        resized = resized.squeeze(0);
    }

    // Convert back to original dtype for masks
    if (is_mask) {
        resized = resized.round().to(tensor.dtype());
    }

    return resized;
}