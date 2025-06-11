//
// Enhanced main.cpp - Optimized for better myocardium segmentation and learning
//

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <utransmambanet.hpp>
#include "utils/acdc_dataset.hpp"
#include "utils/driver.hpp"
#include "models/unet.hpp"
#include <vector>
#include "utils/statistics.hpp"
#include "utils/custom_loss.hpp"

// Configuration constants - OPTIMIZED
namespace Config {
    // FIXED: Better learning rate and training config
    constexpr double LEARNING_RATE = 2e-4;          // Slightly higher initial LR
    constexpr double WEIGHT_DECAY = 1e-5;           // Reduced weight decay for better learning
    constexpr int64_t GRADIENT_ACCUMULATION_STEPS = 2;  // Keep same

    constexpr size_t DEFAULT_NUM_EPOCHS = 100;
    constexpr std::array<double, 1> NORMALIZE_MEAN = {0.485};
    constexpr std::array<double, 1> NORMALIZE_STD = {0.229};
    constexpr std::array<int64_t, 2> IMAGE_SIZE = {256, 256};
    constexpr int64_t INPUT_CHANNELS = 1;
    constexpr int64_t OUTPUT_CHANNELS = 4;

#ifdef __linux__
    constexpr const char* RUN_TITLE = "/root/data/models/one_more_bro";
    constexpr const char* ACDC_DATA_PATH = "/root/ACDC_preprocessed";
    constexpr int64_t BATCH_SIZE = 4;  // Slightly increased for better gradient estimates
#elif defined(__APPLE__) && defined(__MACH__)
    constexpr const char* RUN_TITLE = "../data/models/ENHANCED_UTRANSMAMBA";
    constexpr const char* ACDC_DATA_PATH = "../data/acdc";
    constexpr int64_t BATCH_SIZE = 8;  // Increased for Mac
#endif
}

struct TrainingConfig {
    size_t num_epochs = Config::DEFAULT_NUM_EPOCHS;
    double learning_rate = Config::LEARNING_RATE;
    double weight_decay = Config::WEIGHT_DECAY;
    int64_t gradient_accumulation_steps = Config::GRADIENT_ACCUMULATION_STEPS;
    int64_t batch_size = Config::BATCH_SIZE;
    std::string model_save_path;
    std::string final_model_path;
};

enum class RunMode {
    TRAIN,
    INFER,
    UNKNOWN
};

enum class ModelType {
    UNET,
    RESNET,
    UTRANSMAMBANET,
    UNKNOWN
};

struct CommandLineArgs {
    RunMode mode = RunMode::UNKNOWN;
    ModelType model_type = ModelType::UNKNOWN;
    size_t epochs = Config::DEFAULT_NUM_EPOCHS;
    bool valid = false;
};

// Utility functions
std::string generate_model_filename(const std::string& base_path, const std::string& suffix = "") {
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    return base_path + "_" + std::to_string(timestamp) + suffix + ".pt";
}

torch::Device get_device() {
#ifdef __linux__
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available. Primary GPU: cuda:0" << std::endl;
        return torch::Device(torch::kCUDA, 0);
    } else {
        std::cout << "CUDA not available on Linux, falling back to CPU." << std::endl;
        return torch::kCPU;
    }
#elif defined(__APPLE__) && defined(__MACH__)
    if (torch::mps::is_available()) {
        std::cout << "MPS is available on macOS." << std::endl;
        return torch::Device(torch::kMPS);
    } else {
        std::cout << "MPS not available on macOS, falling back to CPU." << std::endl;
        return torch::kCPU;
    }
#else
    std::cout << "Platform not explicitly supported for GPU, falling back to CPU." << std::endl;
    return torch::kCPU;
#endif
}

CommandLineArgs parse_command_line(int argc, char* argv[]) {
    CommandLineArgs args;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [train|infer] [unet|resnet|utransmambanet] [epochs (train only)]" << std::endl;
        std::cerr << "Defaulting to training mode with UTransMambaNet for " << Config::DEFAULT_NUM_EPOCHS << " epochs." << std::endl;
        args.mode = RunMode::TRAIN;
        args.model_type = ModelType::UTRANSMAMBANET;
        args.valid = true;
        return args;
    }

    // Parse mode
    std::string mode_str = argv[1];
    if (mode_str == "train") {
        args.mode = RunMode::TRAIN;
    } else if (mode_str == "infer") {
        args.mode = RunMode::INFER;
    } else {
        std::cerr << "Invalid mode '" << mode_str << "'. Use 'train' or 'infer'." << std::endl;
        return args;
    }

    // Parse model type
    std::string model_str = argv[2];
    if (model_str == "unet") {
        args.model_type = ModelType::UNET;
    } else if (model_str == "resnet") {
        args.model_type = ModelType::RESNET;
    } else if (model_str == "utransmambanet") {
        args.model_type = ModelType::UTRANSMAMBANET;
    } else {
        std::cerr << "Invalid model type '" << model_str << "'. Use 'unet', 'resnet', or 'utransmambanet'." << std::endl;
        return args;
    }

    // Parse epochs for training mode
    if (argc >= 4 && args.mode == RunMode::TRAIN) {
        try {
            args.epochs = std::stoul(argv[3]);
            if (args.epochs == 0) {
                std::cerr << "Number of epochs must be greater than 0." << std::endl;
                return args;
            }
        } catch (const std::exception& e) {
            std::cerr << "Invalid number of epochs: " << argv[3] << std::endl;
            return args;
        }
    }

    args.valid = true;
    return args;
}

std::optional<std::vector<torch::Device>> setup_multi_gpu(const torch::Device& primary_device) {
    if (!primary_device.is_cuda()) {
        return std::nullopt;
    }

    int num_gpus = torch::cuda::device_count();
    if (num_gpus <= 1) {
        return std::nullopt;
    }

    std::cout << "Found " << num_gpus << " GPUs. Setting up for potential DataParallel usage." << std::endl;
    std::vector<torch::Device> device_list;
    for (int i = 0; i < num_gpus; ++i) {
        device_list.emplace_back(torch::kCUDA, i);
    }
    return device_list;
}

void ensure_model_on_device(std::shared_ptr<UTransMambaNetImpl>& model, const torch::Device& device) {
    model->to(device);

    // Ensure all parameters are on the correct device
    for (auto& param : model->parameters()) {
        if (device.is_cuda() && !param.is_cuda()) {
            param = param.to(device);
        }
    }
}

std::shared_ptr<UTransMambaNetImpl> create_model(ModelType model_type) {
    switch (model_type) {
        case ModelType::UTRANSMAMBANET:
            std::cout << "Creating Enhanced UTransMambaNet model..." << std::endl;
            return std::make_shared<UTransMambaNetImpl>(Config::INPUT_CHANNELS, Config::OUTPUT_CHANNELS);
        case ModelType::UNET:
            throw std::runtime_error("UNet model creation not implemented yet");
        case ModelType::RESNET:
            throw std::runtime_error("ResNet model creation not implemented yet");
        default:
            throw std::runtime_error("Unknown model type");
    }
}

int run_training(const CommandLineArgs &args, torch::Device &device) {
    TrainingConfig config;
    config.num_epochs = args.epochs;
    config.batch_size = Config::BATCH_SIZE;
    config.learning_rate = Config::LEARNING_RATE;
    config.model_save_path = generate_model_filename(Config::RUN_TITLE);
    config.final_model_path = generate_model_filename(Config::RUN_TITLE, "_final");

    std::cout << "\n=== Enhanced Training Configuration ===" << std::endl;
    std::cout << "  Model type: UTransMambaNet (Boundary-Enhanced)" << std::endl;
    std::cout << "  Epochs: " << config.num_epochs << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Effective batch size: " << config.batch_size * config.gradient_accumulation_steps << std::endl;
    std::cout << "  Initial learning rate: " << config.learning_rate << " (Higher for better exploration)" << std::endl;
    std::cout << "  Weight decay: " << config.weight_decay << " (Reduced for better learning)" << std::endl;
    std::cout << "  LR Schedule: ReduceLROnPlateau (factor=0.85, patience=8)" << std::endl;
    std::cout << "  Loss: CE(30%) + Dice(60%) + Boundary(10%) for myocardium continuity" << std::endl;
    std::cout << "  Device: " << device << std::endl;
    std::cout << "  Model save path: " << config.model_save_path << std::endl;
    std::cout << "=========================================\n" << std::endl;

    try {
        // Create datasets
        auto train_dataset = ACDCDataset(Config::ACDC_DATA_PATH, Mode::TRAIN, Config::IMAGE_SIZE)
            .map(torch::data::transforms::Normalize<>(Config::NORMALIZE_MEAN, Config::NORMALIZE_STD))
            .map(torch::data::transforms::Stack<>());

        auto val_dataset = ACDCDataset(Config::ACDC_DATA_PATH, Mode::VAL, Config::IMAGE_SIZE)
            .map(torch::data::transforms::Normalize<>(Config::NORMALIZE_MEAN, Config::NORMALIZE_STD))
            .map(torch::data::transforms::Stack<>());

        std::cout << "Dataset loaded successfully!" << std::endl;

        // Create data loaders with optimal settings
        auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(config.batch_size)
                .workers(4)  // Parallel data loading
        );

        auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_dataset),
            torch::data::DataLoaderOptions()
                .batch_size(config.batch_size)
                .workers(4)
        );

        // Create model
        auto model = create_model(args.model_type);
        ensure_model_on_device(model, device);

        // Count parameters
        size_t total_params = 0;
        for (const auto& p : model->parameters()) {
            total_params += p.numel();
        }
        std::cout << "Total model parameters: " << total_params / 1e6 << "M" << std::endl;

        // Setup multi-GPU if available
        auto multi_gpu_devices = setup_multi_gpu(device);

        // FIXED: Enhanced optimizer configuration
        torch::optim::AdamW optimizer(
            model->parameters(),
            torch::optim::AdamWOptions(config.learning_rate)
                .weight_decay(config.weight_decay)
                .betas({0.9, 0.999})  // Standard betas
                .eps(1e-8)
        );

        std::cout << "\n🚀 Starting enhanced training for better myocardium segmentation..." << std::endl;
        std::cout << "Key improvements:" << std::endl;
        std::cout << "  ✓ Boundary loss for myocardium continuity" << std::endl;
        std::cout << "  ✓ Improved learning rate schedule" << std::endl;
        std::cout << "  ✓ Enhanced class weighting" << std::endl;
        std::cout << "  ✓ Adaptive loss adjustment" << std::endl;

        // Start enhanced training
        train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            config.num_epochs,
            config.gradient_accumulation_steps,
            config.model_save_path,
            config.final_model_path
            //,multi_gpu_devices
        );

        std::cout << "\n🎉 Enhanced training completed successfully!" << std::endl;
        std::cout << "Expected improvements:" << std::endl;
        std::cout << "  • Better myocardium circle continuity" << std::endl;
        std::cout << "  • Higher overall Dice scores (target: >0.90)" << std::endl;
        std::cout << "  • More stable learning curve" << std::endl;
        
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
        return -1;
    }
}

int run_inference(const CommandLineArgs& args, const torch::Device& device) {
    std::cout << "Inference mode - loading best model and evaluating..." << std::endl;
    std::cout << "Inference not fully implemented yet." << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    // Set random seed for reproducibility
    torch::manual_seed(42);

    // Parse command line arguments
    auto args = parse_command_line(argc, argv);
    if (!args.valid) {
        return -1;
    }

    // Get device
    torch::Device device = get_device();

    // Enable cuDNN benchmarking for better performance
    if (device.is_cuda()) {
        torch::globalContext().setBenchmarkCuDNN(true);
    }

    std::cout << "\n🎯 UTransMambaNet Enhanced Training System" << std::endl;
    std::cout << "Optimized for cardiac segmentation with myocardium continuity" << std::endl;

    try {
        switch (args.mode) {
            case RunMode::TRAIN:
                return run_training(args, device);
            case RunMode::INFER:
                return run_inference(args, device);
            default:
                std::cerr << "Unknown mode" << std::endl;
                return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        return -1;
    }
}