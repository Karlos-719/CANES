# CMake Wrapper Makefile - Enhanced for Visualization
# Usage: make <target> [CONFIG=Debug|Release|RelWithDebInfo|MinSizeRel]

# Default configuration
CONFIG ?= Release

# Build directory structure
BUILD_DIR = build
BUILD_CONFIG_DIR = $(BUILD_DIR)/$(CONFIG)

# Number of parallel jobs
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default target
.PHONY: all
all: build

# Configure CMake for specific build type
.PHONY: configure
configure:
	@echo "Configuring CMake for $(CONFIG) build..."
	@mkdir -p $(BUILD_CONFIG_DIR)
	cmake -B $(BUILD_CONFIG_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) .

# Build all targets
.PHONY: build
build: configure
	@echo "Building $(CONFIG) configuration..."
	cmake --build $(BUILD_CONFIG_DIR) --config $(CONFIG) -j $(JOBS)

# Build specific targets
.PHONY: training
training: configure
	@echo "Building training executable..."
	cmake --build $(BUILD_CONFIG_DIR) --config $(CONFIG) --target UTransMambaNet -j $(JOBS)

.PHONY: inference
inference: configure
	@echo "Building 2D inference visualizer..."
	cmake --build $(BUILD_CONFIG_DIR) --config $(CONFIG) --target InferenceVisualizer -j $(JOBS)

.PHONY: visualizer3d
visualizer3d: configure
	@echo "Building 3D visualizer..."
	cmake --build $(BUILD_CONFIG_DIR) --config $(CONFIG) --target Visualizer3D -j $(JOBS)

.PHONY: visualization
visualization: inference visualizer3d

# Run targets with default parameters
.PHONY: run-training
run-training: training
	@echo "Running training..."
	@cd $(BUILD_CONFIG_DIR) && ./UTransMambaNet train utransmambanet 10

.PHONY: run-inference
run-inference: inference
	@echo "Running 2D inference (provide MODEL_PATH, DATA_PATH, OUTPUT_DIR)..."
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make run-inference MODEL_PATH=<path> DATA_PATH=<path> [OUTPUT_DIR=<path>]"; \
		echo "Example: make run-inference MODEL_PATH=../models/best.pt DATA_PATH=../data/acdc OUTPUT_DIR=./results"; \
		exit 1; \
	fi
	@cd $(BUILD_CONFIG_DIR) && ./InferenceVisualizer $(MODEL_PATH) $(DATA_PATH) $(or $(OUTPUT_DIR),./inference_results)

.PHONY: run-3d
run-3d: visualizer3d
	@echo "Running 3D visualization (provide MODEL_PATH, DATA_PATH, OUTPUT_DIR)..."
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make run-3d MODEL_PATH=<path> DATA_PATH=<path> [OUTPUT_DIR=<path>]"; \
		echo "Example: make run-3d MODEL_PATH=../models/best.pt DATA_PATH=../data/acdc OUTPUT_DIR=./3d_results"; \
		exit 1; \
	fi
	@cd $(BUILD_CONFIG_DIR) && ./Visualizer3D $(MODEL_PATH) $(DATA_PATH) $(or $(OUTPUT_DIR),./3d_results)

# Installation helpers
.PHONY: install-deps-ubuntu
install-deps-ubuntu:
	@echo "Installing dependencies for Ubuntu/Debian..."
	sudo apt update
	sudo apt install -y libopencv-dev libvtk9-dev libvtk9-qt-dev

.PHONY: install-deps-macos
install-deps-macos:
	@echo "Installing dependencies for macOS..."
	brew install opencv vtk

# Check what can be built
.PHONY: check-deps
check-deps:
	@echo "Checking dependencies..."
	@echo "========================"
	@pkg-config --exists opencv4 && echo "✅ OpenCV: Found" || echo "❌ OpenCV: Not found"
	@pkg-config --exists vtk && echo "✅ VTK: Found" || echo "❌ VTK: Not found" || true
	@echo ""
	@echo "If dependencies missing:"
	@echo "  Ubuntu/Debian: make install-deps-ubuntu"
	@echo "  macOS:         make install-deps-macos"

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning $(CONFIG) build..."
	@rm -rf $(BUILD_CONFIG_DIR)

.PHONY: clean-all
clean-all:
	@echo "Cleaning all builds..."
	@rm -rf $(BUILD_DIR)

# Fresh build
.PHONY: rebuild
rebuild: clean build

# Test build (quick verification)
.PHONY: test-build
test-build:
	@echo "Testing build system..."
	@$(MAKE) configure CONFIG=Release
	@echo "✅ CMake configuration successful"
	@echo "Available targets after full build:"
	@echo "  - UTransMambaNet (training)"
	@echo "  - InferenceVisualizer (2D visualization, requires OpenCV)"
	@echo "  - Visualizer3D (3D visualization, requires VTK)"

# Format code (if clang-format available)
.PHONY: format
format:
	@echo "Formatting code..."
	@find . -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" | \
		grep -v build | xargs -r clang-format -i

# Generate compile database for IDEs
.PHONY: compile-db
compile-db:
	@echo "Generating compile database..."
	cmake -B $(BUILD_CONFIG_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON .
	@cp $(BUILD_CONFIG_DIR)/compile_commands.json .

# Create poster-ready results directory
.PHONY: setup-poster
setup-poster:
	@echo "Setting up poster results directory..."
	@mkdir -p poster_results
	@mkdir -p 3d_results
	@echo "Created:"
	@echo "  poster_results/ - for 2D inference outputs"
	@echo "  3d_results/     - for 3D visualization outputs"

# Quick poster generation (if model exists)
.PHONY: generate-poster-images
generate-poster-images: setup-poster inference
	@if [ -z "$(MODEL_PATH)" ] || [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make generate-poster-images MODEL_PATH=<path> DATA_PATH=<path>"; \
		echo "Example: make generate-poster-images MODEL_PATH=./best_model.pt DATA_PATH=./data/acdc"; \
		exit 1; \
	fi
	@echo "Generating poster-quality images..."
	@cd $(BUILD_CONFIG_DIR) && ./InferenceVisualizer $(MODEL_PATH) $(DATA_PATH) ../poster_results
	@echo "✅ Poster images ready in poster_results/"

# Help target
.PHONY: help
help:
	@echo "UTransMambaNet Build System"
	@echo "=========================="
	@echo ""
	@echo "Main targets:"
	@echo "  build              - Build all executables"
	@echo "  training           - Build training executable only"
	@echo "  inference          - Build 2D inference visualizer"
	@echo "  visualizer3d       - Build 3D visualizer"
	@echo "  visualization      - Build both visualizers"
	@echo ""
	@echo "Run targets:"
	@echo "  run-training       - Run training with default params"
	@echo "  run-inference      - Run 2D inference (needs MODEL_PATH, DATA_PATH)"
	@echo "  run-3d            - Run 3D visualization (needs MODEL_PATH, DATA_PATH)"
	@echo ""
	@echo "Poster workflow:"
	@echo "  setup-poster       - Create output directories"
	@echo "  generate-poster-images - Generate visualization (needs MODEL_PATH, DATA_PATH)"
	@echo ""
	@echo "Dependencies:"
	@echo "  check-deps         - Check what dependencies are available"
	@echo "  install-deps-ubuntu - Install deps on Ubuntu/Debian"
	@echo "  install-deps-macos  - Install deps on macOS"
	@echo ""
	@echo "Utilities:"
	@echo "  clean             - Clean current config"
	@echo "  clean-all         - Clean all configs"
	@echo "  format            - Format source code"
	@echo "  compile-db        - Generate compile_commands.json"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make run-inference MODEL_PATH=./model.pt DATA_PATH=./data"
	@echo "  make generate-poster-images MODEL_PATH=./best.pt DATA_PATH=./acdc"

# Default help
.DEFAULT_GOAL := help