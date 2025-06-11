//
// Fixed driver.cpp - Corrected learning rate scheduling and loss configuration
//

#include "driver.hpp"
#include "models/utransmambanet.hpp"

// Helper function to check for NaN in tensors
bool has_nan(const torch::Tensor& tensor) {
    return torch::isnan(tensor).any().item<bool>();
}

// Helper function to calculate gradient norm
float calculate_grad_norm(const std::vector<torch::Tensor>& parameters) {
    float total_norm = 0.0;
    for (const auto& param : parameters) {
        if (param.grad().defined()) {
            auto param_norm = param.grad().norm().item<float>();
            total_norm += param_norm * param_norm;
        }
    }
    return std::sqrt(total_norm);
}

template<typename Model, typename TrainLoader, typename ValLoader, typename Optimizer>
void train_model(Model& model,
                TrainLoader& train_loader,
                ValLoader& val_loader,
                Optimizer& optimizer,
                torch::Device& device,
                size_t num_epochs,
                int64_t gradient_accumulation_steps,
                std::string model_path,
                std::string model_path_final,
                std::optional<std::vector<torch::Device>> devices) {

    // FIXED: Better loss configuration for myocardium continuity
    CombinedLoss criterion(0.3, 0.6, 0.1, true, true);  // Enable boundary loss + class weights
    
    // FIXED: Get initial learning rate from optimizer
    float initial_lr = 0.0;
    for (auto& param_group : optimizer.param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(param_group.options());
        initial_lr = options.lr();
        break;
    }

    // FIXED: Much more conservative and effective LR schedule
    float current_lr = initial_lr;
    float lr_decay_factor = 0.85;        // Gentler decay
    size_t lr_decay_patience = 8;        // Wait longer before decay
    size_t lr_decay_counter = 0;
    float min_lr = initial_lr * 0.01;     // Higher minimum LR (1e-6 instead of 1e-7)
    
    // Alternative: Use ReduceLROnPlateau style scheduling
    bool use_plateau_schedule = true;
    float best_loss_for_lr = std::numeric_limits<float>::max();

    float best_val_dice = 0.0f;
    float best_foreground_dice = 0.0f;
    size_t patience = 0;
    const size_t max_patience = 35;  // Increased patience for better convergence
    size_t grad_accum_counter = 0;  // FIXED: Missing variable declaration


    std::cout << "Training with initial LR: " << initial_lr << std::endl;
    std::cout << "Using Enhanced Combined Loss (CE: 0.3, Dice: 0.6, Boundary: 0.1)" << std::endl;
    std::cout << "Loss components: Cross-Entropy + Vectorized Dice + Boundary Loss" << std::endl;
    std::cout << "LR Schedule: ReduceLROnPlateau with factor=" << lr_decay_factor 
              << ", patience=" << lr_decay_patience << std::endl;

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        model->to(device);

        // Training phase with enhanced monitoring
        model->train();
        double train_loss = 0.0;
        torch::Tensor train_dice_total = torch::zeros({4}).to(device);
        size_t train_batches = 0;
        float max_grad_norm = 0.0;
        bool nan_detected = false;

        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            // Check for NaN in input data
            if (has_nan(data) || has_nan(targets)) {
                std::cerr << "WARNING: NaN detected in input data at batch " << train_batches << std::endl;
                continue;
            }

            torch::Tensor outputs;

            if (devices.has_value() && devices.value().size() > 1) {
                try {
                    #ifdef __linux__
                    outputs = torch::nn::parallel::data_parallel(
                        model, data, devices.value(), device);
                    #endif
                } catch (const std::exception& e) {
                    std::cerr << "DataParallel error: " << e.what() << std::endl;
                    outputs = model->forward(data);
                }
            } else {
                outputs = model->forward(data);
            }

            outputs = outputs.to(device);

            // Check for NaN in outputs
            if (has_nan(outputs)) {
                std::cerr << "ERROR: NaN detected in model outputs at batch " << train_batches << std::endl;
                nan_detected = true;
                break;
            }

            // Calculate loss with all components
            auto actual_loss = criterion.forward(outputs, targets);
            
            // Check for NaN in loss
            if (has_nan(actual_loss)) {
                std::cerr << "ERROR: NaN detected in loss at batch " << train_batches << std::endl;
                nan_detected = true;
                break;
            }

            auto scaled_loss = actual_loss / gradient_accumulation_steps;
            scaled_loss.backward();

            grad_accum_counter++;

            // Update weights after accumulation
            if (grad_accum_counter % gradient_accumulation_steps == 0) {
                // FIXED: More conservative gradient clipping for stability
                float grad_norm = torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                max_grad_norm = std::max(max_grad_norm, grad_norm);
                
                // Check for exploding gradients
                if (grad_norm > 5.0) {
                    std::cerr << "WARNING: Large gradient norm detected: " << grad_norm << std::endl;
                }

                optimizer.step();
                optimizer.zero_grad();
            }

            // Calculate training dice for monitoring
            auto predictions = torch::argmax(outputs, 1);
            auto batch_dice = calculate_dice(predictions, targets, 4).to(device);
            train_dice_total += batch_dice;

            train_loss += actual_loss.template item<double>();

            if (train_batches % 10 == 0) {
                auto foreground_dice = torch::mean(batch_dice.slice(0, 1)).template item<float>();
                std::cout << "  Batch " << train_batches
                          << ", Loss: " << std::fixed << std::setprecision(4) << actual_loss.template item<double>()
                          << ", FG Dice: " << std::fixed << std::setprecision(4) << foreground_dice
                          << ", Grad Norm: " << std::fixed << std::setprecision(3) << max_grad_norm
                          << std::endl;
                max_grad_norm = 0.0;  // Reset for next 10 batches
            }

            train_batches++;
        }

        // Stop training if NaN detected
        if (nan_detected) {
            std::cerr << "Training stopped due to NaN detection." << std::endl;
            break;
        }

        // Validation phase
        model->eval();
        double val_loss = 0.0;
        torch::Tensor total_dice = torch::zeros({4}).to(device);
        size_t val_batches = 0;

        {
            torch::NoGradGuard no_grad;
            for (auto& batch : *val_loader) {
                auto data = batch.data.to(device);
                auto targets = batch.target.to(device);

                auto outputs = model->forward(data);
                auto loss = criterion.forward(outputs, targets);
                val_loss += loss.template item<double>();

                auto predictions = torch::argmax(outputs, 1);
                auto batch_dice = calculate_dice(predictions, targets, 4).to(device);
                total_dice += batch_dice;
                val_batches++;
            }
        }

        // Calculate metrics
        auto avg_train_loss = train_loss / train_batches;
        auto avg_train_dice = (train_dice_total / static_cast<double>(train_batches)).cpu();
        auto avg_val_loss = val_loss / val_batches;
        auto avg_dice = (total_dice / static_cast<double>(val_batches)).cpu();

        auto foreground_dice = torch::mean(avg_dice.slice(0, 1)).item<float>();
        auto mean_dice_all = torch::mean(avg_dice).item<float>();

        // FIXED: Improved learning rate scheduling
        if (use_plateau_schedule) {
            // Use validation loss for LR scheduling
            if (avg_val_loss < best_loss_for_lr) {
                best_loss_for_lr = avg_val_loss;
                lr_decay_counter = 0;
            } else {
                lr_decay_counter++;
                
                if (lr_decay_counter >= lr_decay_patience && current_lr > min_lr) {
                    current_lr = std::max(current_lr * lr_decay_factor, min_lr);
                    lr_decay_counter = 0;
                    
                    // Update optimizer learning rate
                    for (auto& param_group : optimizer.param_groups()) {
                        auto& options = static_cast<torch::optim::AdamWOptions&>(param_group.options());
                        options.lr(current_lr);
                    }
                    
                    std::cout << "📉 Learning rate reduced to: " << std::scientific << current_lr << std::endl;
                }
            }
        }

        // Print results with enhanced metrics
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start).count();

        std::cout << "\n=== Epoch " << epoch << "/" << num_epochs << " Results (Time: " << epoch_duration << "s) ===" << std::endl;
        std::cout << "LR: " << std::scientific << std::setprecision(2) << current_lr << std::endl;
        std::cout << "Train Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                  << " | Val Loss: " << avg_val_loss << std::endl;
        std::cout << "Mean Dice (all): " << mean_dice_all
                  << " | Foreground Dice: " << foreground_dice << std::endl;
        std::cout << "Class Dice: ";
        std::vector<std::string> class_names = {"BG", "RV", "Myo", "LV"};
        for (int i = 0; i < 4; ++i) {
            std::cout << class_names[i] << ":" << std::fixed << std::setprecision(3)
                      << avg_dice[i].item<float>() << " ";
        }
        std::cout << std::endl;

        // Enhanced myocardium-specific monitoring
        auto myo_dice = avg_dice[2].item<float>();
        if (myo_dice < 0.85) {
            std::cout << "⚠️  Myocardium Dice below 0.85 - check for circle breaks" << std::endl;
        }

        // Save best model with better criteria
        bool is_best = false;
        if (foreground_dice > best_foreground_dice || 
            (foreground_dice >= best_foreground_dice - 0.005 && myo_dice > avg_dice[2].item<float>())) {
            
            best_foreground_dice = foreground_dice;
            best_val_dice = mean_dice_all;
            patience = 0;
            is_best = true;
            
            torch::save(model, model_path + "_best.pt");
            std::cout << "✓ New best model saved! FG Dice: " << best_foreground_dice 
                      << ", Myo Dice: " << myo_dice << std::endl;
        } else {
            patience++;
            if (patience > max_patience) {
                std::cout << "Early stopping triggered after " << patience << " epochs without improvement." << std::endl;
                break;
            }
        }

        // Adaptive loss weight adjustment for myocardium
        if (epoch % 5 == 0 && myo_dice < 0.87) {
            // Slightly increase boundary loss weight for better myocardium continuity
            criterion.set_weights(0.25, 0.6, 0.15);  // Reduce CE, increase boundary
            std::cout << "🔧 Adjusted loss weights for better myocardium segmentation" << std::endl;
        }

        // Periodic checkpoint
        if (epoch % 15 == 0) {
            torch::save(model, model_path + "_epoch_" + std::to_string(epoch) + ".pt");
        }

        // Performance analysis
        if (epoch % 10 == 0) {
            std::cout << "\n📊 Performance Analysis:" << std::endl;
            std::cout << "  RV Progress: " << (avg_dice[1].item<float>() > 0.80 ? "✓" : "⚠️") 
                      << " " << avg_dice[1].item<float>() << std::endl;
            std::cout << "  Myo Progress: " << (myo_dice > 0.87 ? "✓" : "⚠️") 
                      << " " << myo_dice << " (target: >0.87)" << std::endl;
            std::cout << "  LV Progress: " << (avg_dice[3].item<float>() > 0.92 ? "✓" : "⚠️") 
                      << " " << avg_dice[3].item<float>() << std::endl;
        }
    }

    // Save final model
    torch::save(model, model_path_final);
    std::cout << "\n🎯 Training completed!" << std::endl;
    std::cout << "Best foreground Dice: " << best_foreground_dice << std::endl;
    std::cout << "Best overall Dice: " << best_val_dice << std::endl;
    std::cout << "Final model saved to: " << model_path_final << std::endl;
    
    std::cout << "\n💡 Tips for further improvement:" << std::endl;
    std::cout << "1. If myocardium still has breaks, increase boundary loss weight to 0.2" << std::endl;
    std::cout << "2. Consider post-processing with morphological closing for myocardium" << std::endl;
    std::cout << "3. Use test-time augmentation for final inference" << std::endl;
}
// Template instantiations remain the same...
#ifdef __linux__
//template void train_model<std::shared_ptr<UTransMambaNetImpl>, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler> > >, torch::optim::AdamW>(std::shared_ptr<UTransMambaNetImpl>&, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >&, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler> > >&, torch::optim::AdamW&, c10::Device&, unsigned long, long, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::optional<std::vector<c10::Device, std::allocator<c10::Device> > >);
template void train_model<std::shared_ptr<UTransMambaNetImpl>, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler> > >, torch::optim::AdamW>(std::shared_ptr<UTransMambaNetImpl>&, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >&, std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor> >, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::SequentialSampler> > >&, torch::optim::AdamW&, c10::Device&, unsigned long, long, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::optional<std::vector<c10::Device, std::allocator<c10::Device> > >);
#elif defined(__APPLE__) && defined(__MACH__)
//template void train_model<std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>>>, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>>>, torch::optim::AdamW>(std::__1::shared_ptr<ModelStub>, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>>>&, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>>>&, torch::optim::AdamW&, c10::Device, unsigned long, long long, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>);
template void train_model<std::__1::shared_ptr<UTransMambaNetImpl>, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>>>, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>>>, torch::optim::AdamW>(std::__1::shared_ptr<UTransMambaNetImpl>&, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::RandomSampler>>>&, std::__1::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>, std::__1::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<torch::data::datasets::MapDataset<ACDCDataset, torch::data::transforms::Normalize<at::Tensor>>, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor>>>, torch::data::samplers::SequentialSampler>>>&, torch::optim::AdamW&, c10::Device&, unsigned long, long long, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>, std::__1::optional<std::__1::vector<c10::Device, std::__1::allocator<c10::Device>>>);
#else
#error "Platform not supported - please add your platform-specific paths"
#endif