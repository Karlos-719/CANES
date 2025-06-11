//
// Enhanced inference_main.cpp - Professional poster-quality visualizations
//

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <regex>
#include <iomanip>
#include <sstream>
#include "models/utransmambanet.hpp"
#include "utils/acdc_dataset.hpp"
#include "utils/statistics.hpp"

// Professional color palette for cardiac segmentation
const std::vector<cv::Vec3b> PROFESSIONAL_COLORS = {
    {40, 40, 40},      // Background: Dark Gray (not pure black for better printing)
    {220, 50, 47},     // RV: Vermillion Red (colorblind-safe)
    {44, 162, 95},     // Myocardium: Emerald Green (colorblind-safe)
    {33, 102, 172}     // LV: Royal Blue (colorblind-safe)
};

// Alternative high-contrast palette
const std::vector<cv::Vec3b> HIGH_CONTRAST_COLORS = {
    {30, 30, 30},      // Background: Near Black
    {255, 77, 77},     // RV: Bright Red
    {77, 255, 77},     // Myocardium: Bright Green
    {77, 77, 255}      // LV: Bright Blue
};

struct InferenceConfig {
    std::string model_path;
    std::string data_path;
    std::string output_dir;
    int num_samples = 370;  // Reduced default for poster
    bool use_high_contrast = false;
    int poster_dpi = 300;  // For high-quality poster printing
};

struct PatientFrameInfo {
    std::string patient_id;
    std::string frame_id;
    std::string slice_id;
    bool is_valid;
    
    PatientFrameInfo() : is_valid(false) {}
    
    PatientFrameInfo(const std::string& sample_patient_id) : is_valid(false) {
        parse_patient_id(sample_patient_id);
    }
    
    void parse_patient_id(const std::string& sample_patient_id) {
        // Parse patterns like: "001_frame01_slice5" or "001_frame01"
        std::regex pattern1(R"((\d+)_frame(\d+)_slice(\d+))");
        std::regex pattern2(R"((\d+)_frame(\d+))");
        std::smatch match;
        
        if (std::regex_search(sample_patient_id, match, pattern1)) {
            patient_id = match[1].str();
            frame_id = match[2].str();
            slice_id = match[3].str();
            is_valid = true;
        } else if (std::regex_search(sample_patient_id, match, pattern2)) {
            patient_id = match[1].str();
            frame_id = match[2].str();
            slice_id = "";
            is_valid = true;
        }
    }
    
    std::string get_filename_prefix() const {
        if (!is_valid) return "unknown";
        
        std::string prefix = "patient" + patient_id + "_frame" + frame_id;
        if (!slice_id.empty()) {
            prefix += "_slice" + slice_id;
        }
        return prefix;
    }
    
    std::string get_formatted_display() const {
        if (!is_valid) return "Unknown";
        
        // Format with leading zeros for consistency
        std::stringstream ss;
        ss << "Patient " << std::setw(3) << std::setfill('0') << patient_id 
           << " | Frame " << std::setw(2) << std::setfill('0') << frame_id;
        if (!slice_id.empty()) {
            ss << " | Slice " << std::setw(2) << std::setfill('0') << slice_id;
        }
        return ss.str();
    }
};

class ProfessionalMedicalVisualizer {
private:
    std::string output_dir_;
    std::vector<cv::Vec3b> colors_;
    int poster_dpi_;
    
    // Professional font settings
    const int TITLE_FONT = cv::FONT_HERSHEY_DUPLEX;
    const int LABEL_FONT = cv::FONT_HERSHEY_SIMPLEX;
    const double TITLE_SCALE = 1.2;
    const double LABEL_SCALE = 1;
    const double METRIC_SCALE = 0.7;
    const int TITLE_THICKNESS = 3;
    const int LABEL_THICKNESS = 2;
    const int METRIC_THICKNESS = 2;

public:
    ProfessionalMedicalVisualizer(const std::string& output_dir, bool use_high_contrast = false, int dpi = 300) 
        : output_dir_(output_dir), poster_dpi_(dpi) {
        colors_ = use_high_contrast ? HIGH_CONTRAST_COLORS : PROFESSIONAL_COLORS;
        std::filesystem::create_directories(output_dir_);
    }
    
    // Public method to check class presence (static for external use)
    static std::vector<bool> check_class_presence(const torch::Tensor& mask, int num_classes = 4) {
        std::vector<bool> presence(num_classes, false);
        auto mask_flat = mask.flatten();
        
        for (int c = 0; c < num_classes; c++) {
            auto class_mask = (mask_flat == c);
            presence[c] = class_mask.any().item<bool>();
        }
        
        return presence;
    }

    cv::Mat tensor_to_cv(const torch::Tensor& tensor) {
        auto tensor_cpu = tensor.to(torch::kCPU).to(torch::kFloat);
        
        if (tensor_cpu.dim() == 2) {
            auto tensor_normalized = (tensor_cpu * 255).clamp(0, 255).to(torch::kUInt8);
            cv::Mat img(tensor_cpu.size(0), tensor_cpu.size(1), CV_8UC1, tensor_normalized.data_ptr<uint8_t>());
            return img.clone();
        } else if (tensor_cpu.dim() == 3 && tensor_cpu.size(0) == 1) {
            return tensor_to_cv(tensor_cpu.squeeze(0));
        }
        
        throw std::runtime_error("Unsupported tensor dimensions");
    }

    cv::Mat mask_to_colored(const torch::Tensor& mask) {
        auto mask_cpu = mask.to(torch::kCPU).to(torch::kLong);
        int H = mask_cpu.size(0);
        int W = mask_cpu.size(1);
        
        cv::Mat colored_mask(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
        
        auto mask_accessor = mask_cpu.accessor<long, 2>();
        
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int class_id = mask_accessor[y][x];
                if (class_id >= 0 && class_id < colors_.size()) {
                    colored_mask.at<cv::Vec3b>(y, x) = colors_[class_id];
                }
            }
        }
        
        return colored_mask;
    }

    // Add subtle drop shadow effect
    cv::Mat add_shadow(const cv::Mat& img, int shadow_size = 5) {
        cv::Mat shadow_img = cv::Mat::zeros(img.rows + shadow_size*2, img.cols + shadow_size*2, img.type());
        
        // Create shadow
        cv::Mat shadow_region = shadow_img(cv::Rect(shadow_size, shadow_size, img.cols, img.rows));
        cv::Mat gray_shadow;
        cv::cvtColor(img, gray_shadow, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_shadow, gray_shadow, 1, 80, cv::THRESH_BINARY);
        cv::cvtColor(gray_shadow, gray_shadow, cv::COLOR_GRAY2BGR);
        gray_shadow.copyTo(shadow_region);
        
        // Blur shadow
        cv::GaussianBlur(shadow_img, shadow_img, cv::Size(shadow_size*2+1, shadow_size*2+1), shadow_size/2.0);
        
        // Copy original image on top
        img.copyTo(shadow_img(cv::Rect(shadow_size, shadow_size, img.cols, img.rows)));
        
        return shadow_img;
    }

    // Create color bar for Dice score visualization
    cv::Mat create_dice_color_bar(float dice_score, int width = 200, int height = 30) {
        cv::Mat bar(height, width, CV_8UC3, cv::Scalar(240, 240, 240));
        
        // Draw background gradient
        for (int x = 0; x < width; ++x) {
            float ratio = static_cast<float>(x) / width;
            cv::Vec3b color;
            
            if (ratio < 0.5) {
                // Red to Yellow
                color[0] = 0;
                color[1] = static_cast<uchar>(255 * ratio * 2);
                color[2] = 255;
            } else {
                // Yellow to Green
                color[0] = 0;
                color[1] = 255;
                color[2] = static_cast<uchar>(255 * (1 - (ratio - 0.5) * 2));
            }
            
            cv::line(bar, cv::Point(x, 5), cv::Point(x, height-5), color, 1);
        }
        
        // Add border
        cv::rectangle(bar, cv::Point(0, 5), cv::Point(width-1, height-5), cv::Scalar(50, 50, 50), 2);
        
        // Add indicator
        int indicator_pos = static_cast<int>(dice_score * width);
        cv::line(bar, cv::Point(indicator_pos, 0), cv::Point(indicator_pos, height), cv::Scalar(0, 0, 0), 3);
        
        return bar;
    }

    // Check which classes are present in the ground truth
    std::vector<bool> get_class_presence(const torch::Tensor& mask, int num_classes = 4) {
        std::vector<bool> presence(num_classes, false);
        auto mask_flat = mask.flatten();
        
        for (int c = 0; c < num_classes; c++) {
            auto class_mask = (mask_flat == c);
            presence[c] = class_mask.any().item<bool>();
        }
        
        return presence;
    }

    // Professional comparison layout
    void save_professional_comparison(const torch::Tensor& image, 
                                    const torch::Tensor& ground_truth,
                                    const torch::Tensor& prediction,
                                    const PatientFrameInfo& patient_info,
                                    const std::vector<float>& dice_scores) {
        
        // Convert to OpenCV images
        cv::Mat img_gray = tensor_to_cv(image);
        cv::Mat gt_colored = mask_to_colored(ground_truth);
        cv::Mat pred_colored = mask_to_colored(prediction);
        
        // Convert grayscale to RGB
        cv::Mat img_rgb;
        cv::cvtColor(img_gray, img_rgb, cv::COLOR_GRAY2RGB);
        
        // Create overlays with better transparency
        cv::Mat gt_overlay, pred_overlay;
        cv::addWeighted(img_rgb, 0.5, gt_colored, 0.5, 0, gt_overlay);
        cv::addWeighted(img_rgb, 0.5, pred_colored, 0.5, 0, pred_overlay);
        
        // Professional display size
        int display_size = 400;
        int padding = 20;
        int header_height = 80;
        int footer_height = 120;
        
        cv::resize(img_rgb, img_rgb, cv::Size(display_size, display_size));
        cv::resize(gt_overlay, gt_overlay, cv::Size(display_size, display_size));
        cv::resize(pred_overlay, pred_overlay, cv::Size(display_size, display_size));
        
        // Create main layout
        int total_width = display_size * 2 + padding * 4;
        int total_height = header_height + display_size + footer_height;
        cv::Mat composite(total_height, total_width, CV_8UC3, cv::Scalar(250, 250, 250));
        
        // Add subtle background gradient
        for (int y = 0; y < total_height; ++y) {
            float factor = 1.0f - (y / static_cast<float>(total_height)) * 0.1f;
            cv::Mat row = composite.row(y);
            row *= factor;
        }
        
        // Add images with shadows
        int y_offset = header_height;
        
        auto add_image_with_border = [&](const cv::Mat& img, int x_pos) {
            cv::Mat bordered;
            cv::copyMakeBorder(img, bordered, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(100, 100, 100));
            bordered.copyTo(composite(cv::Rect(x_pos, y_offset, bordered.cols, bordered.rows)));
        };
        
        // add_image_with_border(img_rgb, padding);
        add_image_with_border(gt_overlay, padding * 1);
        add_image_with_border(pred_overlay, padding * 2 + display_size * 1);
        
        // Professional headers
        // cv::putText(composite, "Original MRI", 
        //            cv::Point(padding + display_size/2 - 70, header_height - 35), 
        //            LABEL_FONT, LABEL_SCALE, cv::Scalar(40, 40, 40), LABEL_THICKNESS);
        cv::putText(composite, "Ground Truth", 
                   cv::Point(padding * 1 + display_size/2 - 75, header_height - 35), 
                   LABEL_FONT, LABEL_SCALE, cv::Scalar(40, 40, 40), LABEL_THICKNESS);
        cv::putText(composite, "Model Prediction", 
                   cv::Point(padding * 2 + display_size * 1 + display_size/2 - 90, header_height - 35), 
                   LABEL_FONT, LABEL_SCALE, cv::Scalar(40, 40, 40), LABEL_THICKNESS);
        
        // // Patient info with better formatting
        // // 1) Get the full text string
        // std::string patientText = patient_info.get_formatted_display();

        // // 2) Measure its size (width, height + baseline)
        // int baseline = 0;
        // cv::Size textSize = cv::getTextSize(
        //     patientText,
        //     TITLE_FONT,
        //     TITLE_SCALE,
        //     TITLE_THICKNESS,
        //     &baseline
        // );

        // // 3) Compute bottom‐right origin, leaving 'padding' pixels from the edges.
        // //    Note: in OpenCV, y‐coordinate in putText is the baseline of the text.
        // int x = composite.cols - padding - textSize.width;
        // int y = composite.rows - padding - baseline;

        // // 4) Draw the text
        // cv::putText(
        //     composite,
        //     patientText,
        //     cv::Point(x, y),
        //     TITLE_FONT,
        //     TITLE_SCALE,
        //     cv::Scalar(20, 20, 20),
        //     TITLE_THICKNESS
        // );
        
        // Dice scores with visual indicators
        int metric_y = total_height - footer_height + 30;
        std::vector<std::string> class_names = {"BG", "RV", "Myo", "LV"};
        
        // Check which classes are actually present
        auto class_presence = get_class_presence(ground_truth, 4);
        
        // Calculate average foreground Dice
        float fg_dice = (dice_scores[1] + dice_scores[2] + dice_scores[3]) / 3.0f;
        
        // Count actually present classes for more accurate average
        int present_classes = 0;
        float present_dice_sum = 0.0f;
        for (int i = 1; i < 4; ++i) {
            if (class_presence[i]) {
                present_classes++;
                present_dice_sum += dice_scores[i];
            }
        }
        
        // Use adjusted average if some classes are absent
        if (present_classes > 0 && present_classes < 3) {
            fg_dice = present_dice_sum / present_classes;
        }
        
        // Main performance metric
        std::stringstream main_metric;
        main_metric << "Average Dice Score: " << std::fixed << std::setprecision(3) << fg_dice;
        if (present_classes < 3) {
            main_metric << " (" << present_classes << "/3 structures)";
        }
        cv::putText(composite, main_metric.str(), 
                   cv::Point(padding, metric_y), 
                   TITLE_FONT, METRIC_SCALE + 0.1, cv::Scalar(20, 20, 20), METRIC_THICKNESS);
        
        // Individual class scores with color coding
        int score_y = metric_y + 40;
        for (int i = 1; i < 4; ++i) {  // Skip background
            std::stringstream ss;
            ss << class_names[i] << ": " << std::fixed << std::setprecision(3) << dice_scores[i];
            
            // Check if class is absent
            bool is_absent = !class_presence[i];
            if (is_absent) {
                ss << " (absent)";
            }
            
            // Color code based on performance
            cv::Scalar text_color;
            if (is_absent) text_color = cv::Scalar(128, 128, 128);                // Gray for absent
            else if (dice_scores[i] >= 0.9) text_color = cv::Scalar(0, 150, 0);   // Green
            else if (dice_scores[i] >= 0.8) text_color = cv::Scalar(0, 100, 200); // Blue
            else if (dice_scores[i] >= 0.7) text_color = cv::Scalar(0, 140, 255); // Orange
            else text_color = cv::Scalar(0, 0, 200);                              // Red
            
            int x_pos = padding + (i-1) * 150;
            cv::putText(composite, ss.str(), 
                       cv::Point(x_pos, score_y), 
                       LABEL_FONT, METRIC_SCALE, text_color, METRIC_THICKNESS);
            
            // Add small color bar (skip for absent classes)
            if (!is_absent) {
                auto color_bar = create_dice_color_bar(dice_scores[i], 120, 15);
                color_bar.copyTo(composite(cv::Rect(x_pos, score_y + 5, color_bar.cols, color_bar.rows)));
            }
        }
        
        // Save with high quality
        std::string filename = patient_info.get_filename_prefix() + "_poster.png";
        std::vector<int> compression_params = {cv::IMWRITE_PNG_COMPRESSION, 0};
        cv::imwrite(output_dir_ + "/" + filename, composite, compression_params);
        
        std::cout << "Created poster visualization: " << filename 
                  << " (Dice: " << std::fixed << std::setprecision(3) << fg_dice << ")" << std::endl;
    }

    // Create professional legend
    void create_professional_legend() {
        int legend_width = 500;
        int legend_height = 300;
        cv::Mat legend(legend_height, legend_width, CV_8UC3, cv::Scalar(250, 250, 250));
        
        // Add title
        cv::putText(legend, "Cardiac Structure Segmentation Legend", 
                   cv::Point(50, 50), 
                   TITLE_FONT, TITLE_SCALE, cv::Scalar(20, 20, 20), TITLE_THICKNESS);
        
        std::vector<std::string> class_names = {
            "Background", 
            "Right Ventricle (RV)", 
            "Myocardium", 
            "Left Ventricle (LV)"
        };
        
        std::vector<std::string> descriptions = {
            "Non-cardiac tissue",
            "Pumps blood to lungs",
            "Heart muscle wall",
            "Main pumping chamber"
        };
        
        for (int i = 0; i < colors_.size(); ++i) {
            int y_pos = 100 + i * 50;
            
            // Draw color square with border
            cv::rectangle(legend, cv::Point(50, y_pos - 20), cv::Point(90, y_pos + 20), 
                         cv::Scalar(colors_[i][0], colors_[i][1], colors_[i][2]), -1);
            cv::rectangle(legend, cv::Point(50, y_pos - 20), cv::Point(90, y_pos + 20), 
                         cv::Scalar(50, 50, 50), 2);
            
            // Add class name
            cv::putText(legend, class_names[i], cv::Point(110, y_pos), 
                       LABEL_FONT, LABEL_SCALE, cv::Scalar(20, 20, 20), LABEL_THICKNESS);
            
            // Add description in lighter text
            cv::putText(legend, descriptions[i], cv::Point(300, y_pos), 
                       LABEL_FONT, METRIC_SCALE, cv::Scalar(100, 100, 100), 1);
        }
        
        // Add performance scale
        cv::putText(legend, "Dice Score Scale:", cv::Point(50, 250), 
                   LABEL_FONT, LABEL_SCALE, cv::Scalar(20, 20, 20), LABEL_THICKNESS);
        
        auto scale_bar = create_dice_color_bar(0.5, 300, 25);
        scale_bar.copyTo(legend(cv::Rect(180, 235, scale_bar.cols, scale_bar.rows)));
        
        cv::putText(legend, "0.0", cv::Point(180, 275), 
                   LABEL_FONT, METRIC_SCALE, cv::Scalar(100, 100, 100), 1);
        cv::putText(legend, "0.5", cv::Point(320, 275), 
                   LABEL_FONT, METRIC_SCALE, cv::Scalar(100, 100, 100), 1);
        cv::putText(legend, "1.0", cv::Point(460, 275), 
                   LABEL_FONT, METRIC_SCALE, cv::Scalar(100, 100, 100), 1);
        
        cv::imwrite(output_dir_ + "/legend_professional.png", legend);
        std::cout << "Created professional legend" << std::endl;
    }
};

torch::Device get_device() {
#ifdef __linux__
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
#elif defined(__APPLE__) && defined(__MACH__)
    if (torch::mps::is_available()) {
        return torch::Device(torch::kMPS);
    }
#endif
    return torch::kCPU;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <data_path> [output_dir] [num_samples]" << std::endl;
        std::cout << "Example: ./inference best_model.pt /path/to/acdc/data ./poster_results 10" << std::endl;
        return -1;
    }

    InferenceConfig config;
    config.model_path = argv[1];
    config.data_path = argv[2];
    config.output_dir = argc > 3 ? argv[3] : "./poster_results";
    config.num_samples = argc > 4 ? std::stoi(argv[4]) : 370;

    std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║    ACDC Cardiac Segmentation - Poster Export     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "├─ Model: " << config.model_path << std::endl;
    std::cout << "├─ Data: " << config.data_path << std::endl;
    std::cout << "├─ Output: " << config.output_dir << std::endl;
    std::cout << "└─ Samples: " << config.num_samples << std::endl;

    auto device = get_device();
    std::cout << "\nDevice: " << device << std::endl;

    try {
        // Load model
        std::cout << "\n▶ Loading model..." << std::endl;
        auto model = std::make_shared<UTransMambaNetImpl>(1, 4);
        torch::load(model, config.model_path);
        model->to(device);
        model->eval();
        std::cout << "✓ Model ready" << std::endl;

        // Load dataset
        std::cout << "\n▶ Loading test data..." << std::endl;
        ACDCDataset test_dataset(config.data_path, Mode::VAL, {256, 256});
        auto dataset_size = test_dataset.size().value();
        std::cout << "✓ Dataset loaded (" << dataset_size << " samples)" << std::endl;

        // Create visualizer
        ProfessionalMedicalVisualizer visualizer(config.output_dir, false, 300);
        visualizer.create_professional_legend();

        // Process samples
        std::cout << "\n▶ Generating poster visualizations..." << std::endl;
        
        torch::Tensor total_dice = torch::zeros({4});
        std::vector<float> best_scores(config.num_samples, 0.0f);
        std::vector<size_t> best_indices(config.num_samples, 0);
        
        {
            torch::NoGradGuard no_grad;
            
            // First pass: find best performing samples for poster
            std::cout << "  Finding best samples..." << std::endl;
            for (size_t idx = 0; idx < dataset_size; ++idx) {
                auto example = test_dataset.get(idx);
                auto data = example.data.unsqueeze(0).to(device);
                auto targets = example.target.unsqueeze(0).to(device);
                
                data = (data - 0.485) / 0.229;
                
                auto outputs = model->forward(data);
                auto predictions = torch::argmax(outputs, 1);
                auto batch_dice = calculate_dice(predictions.cpu(), targets.cpu(), 4);
                
                // Check class presence for accurate average
                auto gt_for_check = targets.squeeze(0).cpu();
                auto class_presence_check = ProfessionalMedicalVisualizer::check_class_presence(gt_for_check, 4);
                int present_fg_classes = 0;
                float present_fg_dice = 0.0f;
                
                for (int i = 1; i < 4; ++i) {
                    if (class_presence_check[i]) {
                        present_fg_classes++;
                        present_fg_dice += batch_dice[i].item<float>();
                    }
                }
                
                float fg_dice = present_fg_classes > 0 ? 
                    present_fg_dice / present_fg_classes : 
                    (batch_dice[1].item<float>() + batch_dice[2].item<float>() + batch_dice[3].item<float>()) / 3.0f;
                
                // Keep track of best samples
                for (int i = 0; i < config.num_samples; ++i) {
                    if (fg_dice > best_scores[i]) {
                        // Shift others down
                        for (int j = config.num_samples - 1; j > i; --j) {
                            best_scores[j] = best_scores[j-1];
                            best_indices[j] = best_indices[j-1];
                        }
                        best_scores[i] = fg_dice;
                        best_indices[i] = idx;
                        break;
                    }
                }
                
                if (idx % 50 == 0) {
                    std::cout << "  Processed " << idx << "/" << dataset_size << " samples\r" << std::flush;
                }
            }
            std::cout << std::endl;
            
            // Second pass: create visualizations for best samples
            std::cout << "  Creating visualizations for top " << config.num_samples << " samples..." << std::endl;
            
            for (int i = 0; i < config.num_samples; ++i) {
                size_t idx = best_indices[i];
                
                auto example = test_dataset.get(idx);
                auto image = example.data;
                auto target = example.target;
                
                const auto& sample = test_dataset.samples_[idx];
                PatientFrameInfo patient_info(sample.patient_id);
                
                auto data = image.unsqueeze(0).to(device);
                auto targets = target.unsqueeze(0).to(device);
                
                data = (data - 0.485) / 0.229;
                
                auto outputs = model->forward(data);
                auto predictions = torch::argmax(outputs, 1);
                
                auto batch_dice = calculate_dice(predictions.cpu(), targets.cpu(), 4);
                total_dice += batch_dice;
                
                auto image_viz = data.squeeze(0).squeeze(0).cpu();
                auto gt = targets.squeeze(0).cpu();
                auto pred = predictions.squeeze(0).cpu();
                
                image_viz = (image_viz * 0.229) + 0.485;
                image_viz = torch::clamp(image_viz, 0.0, 1.0);
                
                std::vector<float> dice_scores = {
                    batch_dice[0].item<float>(),
                    batch_dice[1].item<float>(),
                    batch_dice[2].item<float>(),
                    batch_dice[3].item<float>()
                };
                
                visualizer.save_professional_comparison(image_viz, gt, pred, patient_info, dice_scores);
            }
        }

        // Summary statistics
        auto avg_dice = total_dice / config.num_samples;
        auto fg_dice = (avg_dice[1] + avg_dice[2] + avg_dice[3]) / 3.0;

        std::cout << "\n╔═══════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                 Final Results                     ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Average Dice Scores (Top " << std::setw(2) << config.num_samples << " samples):        ║" << std::endl;
        std::cout << "║   • Right Ventricle: " << std::fixed << std::setprecision(3) << avg_dice[1].item<float>() << "                      ║" << std::endl;
        std::cout << "║   • Myocardium:      " << avg_dice[2].item<float>() << "                      ║" << std::endl;
        std::cout << "║   • Left Ventricle:  " << avg_dice[3].item<float>() << "                      ║" << std::endl;
        std::cout << "║   • Mean Foreground: " << fg_dice.item<float>() << "                      ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════╝" << std::endl;

        std::cout << "\n✓ Poster visualizations ready in: " << config.output_dir << std::endl;
        std::cout << "  Files created:" << std::endl;
        std::cout << "  • legend_professional.png" << std::endl;
        std::cout << "  • patient*_frame*_poster.png (best " << config.num_samples << " results)" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return -1;
    }
}