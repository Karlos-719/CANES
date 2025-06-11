//
// Created by Emin Tunc Kirimlioglu on 5/31/25.
//

#ifndef DRIVER_HPP
#define DRIVER_HPP

#include <torch/torch.h>
#include <string>
#include <filesystem>
#include "models/model_stub.hpp"
#include "utils/acdc_dataset.hpp"
#ifdef __linux__
#include <torch/nn/parallel/data_parallel.h>
#endif
#include "utils/custom_loss.hpp"
#include "utils/statistics.hpp"

template<typename Model, typename TrainLoader, typename ValLoader, typename Optimizer>
void train_model(Model& model,
                TrainLoader& train_loader,
                ValLoader& val_loader,
                Optimizer& optimizer,
                torch::Device& device,
                size_t num_epochs,
                int64_t gradient_accumulation_steps = 1,
                std::string model_path = "best_model.pt",
                std::string model_path_final = "best_model_final.pt",
                std::optional<std::vector<c10::Device>> devices = std::nullopt);

template<typename Model>
void run_inference(Model model,
                  auto& test_loader,
                  torch::Device device,
                  const std::string& model_path,
                  const std::string& output_dir);



#endif //DRIVER_HPP
