# CANES-Net: Convolutional Attention-Based Network with Encoder and State Space Model for Medical Image Segmentation (ACDC)

This project implements state-of-the-art cardiac image segmentation using PyTorch C++ API for the ACDC (Automatic Cardiac Disease Challenge) dataset. The implementation features the novel CANES-Net architecture combining convolutional encoders, attention mechanisms, and state space models for superior segmentation performance.

## Features

- **CANES-Net Architecture**: Advanced segmentation model combining:
  - Convolutional encoder-decoder structure with skip connections
  - Attention-based mechanisms for global context understanding
  - State space models (Mamba) for efficient sequence modeling
  - Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
- **Advanced Training Pipeline**: 
  - Combined loss function (Cross-Entropy + Dice Loss)
  - Gradient accumulation for effective large batch training
  - Learning rate scheduling with ReduceLROnPlateau
  - Model checkpointing and resuming
- **Cross-platform Support**: Works on macOS (MPS), Linux (CUDA), and Windows
- **Medical Image Optimized**: Specifically designed for cardiac MRI segmentation

## Prerequisites

### System Requirements
- **CMake** (>= 3.10)
- **LibTorch** (PyTorch C++ API)
- **C++17 compatible compiler**
- **CUDA** (optional, for GPU acceleration)

### Installing LibTorch

1. Download LibTorch from the [PyTorch website](https://pytorch.org/get-started/locally/)
2. Extract to a suitable location (e.g., `/usr/local/libtorch`)
3. Update the `CMAKE_PREFIX_PATH` in `CMakeLists.txt` to point to your LibTorch installation

### Dataset Setup

The project uses the ACDC dataset. You can obtain it from:
- [ACDC Challenge Website](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- Kaggle (if available)

Configure your dataset path in the code accordingly.

## Project Structure

```
├── models/
│   ├── canesnet.hpp          # CANES-Net architecture header
│   ├── canesnet.cpp          # CANES-Net implementation
│   └── combined_loss.hpp     # Combined loss function
├── utils/
│   ├── trainer.hpp           # Training utilities header
│   ├── trainer.cpp           # Training implementation
│   ├── inference.hpp         # Inference utilities header
│   └── inference.cpp         # Inference implementation
├── data/                     # Dataset directory
├── models_saved/             # Saved model checkpoints
├── main.cpp                  # Main application
├── CMakeLists.txt           # Build configuration
└── README.md               # This file
```

## Building the Project

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd CANES-Net
   ```

2. **Create build directory:**
   ```bash
   mkdir build && cd build
   ```

3. **Configure and build:**
   ```bash
   cmake ..
   make -j$(nproc)
   ```

## Usage

### Training

Train the CANES-Net model:
```bash
./cardiac_segmentation train --epochs 100 --batch-size 8 --lr 1e-4
```

### Inference

Run inference on test data:
```bash
./cardiac_segmentation infer --model-path models_saved/canesnet_best.pt --input-dir data/test
```

### Resume Training

Resume from a checkpoint:
```bash
./cardiac_segmentation train --resume models_saved/checkpoint_epoch_50.pt
```

## Architecture Details

### CANES-Net Architecture

The CANES-Net (Convolutional Attention-Based Network with Encoder and State Space Model) combines four key components:

1. **Convolutional Encoder-Decoder**: 
   - Multi-scale feature extraction with skip connections
   - Hierarchical representation learning
   - Efficient spatial feature processing

2. **Attention Mechanisms**:
   - Self-attention for global context understanding
   - Channel and spatial attention modules
   - Adaptive feature weighting and selection

3. **State Space Models (Mamba)**:
   - Efficient long-range dependency modeling
   - Selective state space mechanism
   - Linear computational complexity for sequence processing

4. **Feature Fusion Module**:
   - Attention-based adaptive fusion of multi-scale features
   - Learnable combination weights
   - Residual connections for stable gradient flow

### Loss Function

**Combined Loss** (30% Cross-Entropy + 70% Dice Loss):
- Cross-Entropy: Provides stable gradients and handles class imbalance
- Dice Loss: Directly optimizes the evaluation metric
- Class weighting for handling imbalanced datasets

## Configuration

### Training Parameters

Key training parameters that can be tuned:

```cpp
// CANES-Net Training Configuration
const int64_t batch_size = 8;
const int64_t gradient_accumulation_steps = 4;  // Effective batch size: 32
const double learning_rate = 1e-4;
const int64_t num_epochs = 100;

// Loss Configuration for CANES-Net
CombinedLoss criterion(0.3, 0.7, true);  // 30% CE, 70% Dice, class weights enabled
```

### Hardware Acceleration

The project automatically detects and uses:
- **Apple Silicon (M1/M2/M3)**: Metal Performance Shaders (MPS)
- **NVIDIA GPUs**: CUDA (if LibTorch built with CUDA support)
- **CPU**: Fallback option with OpenMP optimization

## Performance

Expected performance on ACDC dataset:
- **Mean Dice Score**: ~0.90+
- **Hausdorff Distance**: <10mm
- **Training Time**: ~2-4 hours (GPU dependent)

### Class-wise Performance
- Background: >0.98 Dice
- Right Ventricle: >0.88 Dice  
- Myocardium: >0.87 Dice
- Left Ventricle: >0.92 Dice

## Troubleshooting

### Common Issues

1. **LibTorch not found**: 
   - Update `CMAKE_PREFIX_PATH` in `CMakeLists.txt`
   - Ensure LibTorch version compatibility

2. **CUDA/MPS not available**: 
   - Code automatically falls back to CPU
   - Check GPU drivers and CUDA installation

3. **Out of memory errors**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use mixed precision training

4. **Poor segmentation results**:
   - Ensure proper data preprocessing
   - Check class distribution in dataset
   - Verify ground truth mask format

### Debug Build

For debugging, build with debug symbols:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Model Optimization Tips

1. **Data Augmentation**: Add random flips, rotations, and elastic deformations
2. **Learning Rate**: Use cosine annealing or warm restarts
3. **Regularization**: Add dropout and weight decay
4. **Post-processing**: Apply connected component analysis
5. **Ensemble**: Combine multiple model predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ACDC Challenge organizers for providing the dataset
- PyTorch team for the excellent C++ API
- Medical imaging research community
- Attention mechanism and State Space Model (Mamba) research contributors

## References

- [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- CANES-Net: Convolutional Attention-Based Network with Encoder and State Space Model for Medical Image Segmentation

---

For questions or issues, please open an issue on GitHub or contact the maintainers.
