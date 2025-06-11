# Replace your CMakeLists.txt with this fixed version
cmake_minimum_required(VERSION 3.10)
project(UTransMambaNet)

# Set policy to make CMake respect _ROOT variables
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
elseif(UNIX)
    # Set PATH environment variable BEFORE finding Torch
    set(ENV{PATH} "/usr/local/cuda-12.9/bin:$ENV{PATH}")
    set(ENV{CUDA_HOME} "/usr/local/cuda-12.9")
    set(ENV{CUDA_ROOT} "/usr/local/cuda-12.9")
    set(ENV{LD_LIBRARY_PATH} "/usr/local/cuda-12.9/lib64:$ENV{LD_LIBRARY_PATH}")

    # Set CMake variables
    set(CMAKE_PREFIX_PATH "/usr/local/cuda-12.9;/root/libtorch")
    set(CMAKE_IGNORE_PATH "/usr/include;/usr/lib/x86_64-linux-gnu")
    set(CMAKE_CUDA_ARCHITECTURES 89)

    # Set CUDA variables
    set(CUDAToolkit_ROOT "/usr/local/cuda-12.9")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.9")
    set(CUDA_HOME "/usr/local/cuda-12.9")
    set(CUDA_ROOT "/usr/local/cuda-12.9")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.9/bin/nvcc")
    set(CUDA_NVCC_EXECUTABLE "/usr/local/cuda-12.9/bin/nvcc")

    # Find CUDA first to create the missing targets
    find_package(CUDAToolkit REQUIRED)

    # Create missing CUDA::nvToolsExt target if not found
    if(NOT TARGET CUDA::nvToolsExt)
        find_library(NVTOOLSEXT_LIBRARY
                NAMES nvToolsExt
                PATHS "/usr/local/cuda-12.9/lib64" "/usr/local/cuda-12.9/lib"
                NO_DEFAULT_PATH
        )

        if(NVTOOLSEXT_LIBRARY)
            add_library(CUDA::nvToolsExt SHARED IMPORTED)
            set_target_properties(CUDA::nvToolsExt PROPERTIES
                    IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/cuda-12.9/include"
            )
            message(STATUS "Created CUDA::nvToolsExt target: ${NVTOOLSEXT_LIBRARY}")
        else()
            # If nvToolsExt is not found, create a dummy target
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
            message(STATUS "Created dummy CUDA::nvToolsExt target (library not found)")
        endif()
    endif()
endif()

# Now find Torch (should work with the missing target created)
find_package(Torch REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

find_package(OpenCV)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  OpenCV_LIBS: ${OpenCV_LIBS}")
else()
    message(WARNING "OpenCV not found - 2D visualization will be disabled")
    message(STATUS "Install OpenCV with: sudo apt install libopencv-dev (Linux) or brew install opencv (macOS)")
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/models)
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${HDF5_INCLUDE_DIRS})

# ============================================================================
# MAIN TRAINING EXECUTABLE
# ============================================================================
add_executable(UTransMambaNet
        main.cpp
        models/unet.cpp
        models/utransmambanet.cpp
        utils/acdc_dataset.cpp
        utils/driver.cpp
        utils/statistics.cpp
        utils/custom_loss.cpp
)

target_link_libraries(UTransMambaNet
        "${TORCH_LIBRARIES}"
        ${HDF5_CXX_LIBRARIES}
)

target_compile_definitions(UTransMambaNet PRIVATE ${HDF5_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ============================================================================
# 2D INFERENCE AND VISUALIZATION EXECUTABLE (MOST IMPORTANT FOR POSTER)
# ============================================================================
if(OpenCV_FOUND)
    add_executable(InferenceVisualizer
        inference_main.cpp
        models/unet.cpp
        models/utransmambanet.cpp
        utils/acdc_dataset.cpp
        utils/statistics.cpp
        utils/custom_loss.cpp
    )

    target_link_libraries(InferenceVisualizer
        "${TORCH_LIBRARIES}"
        ${HDF5_CXX_LIBRARIES}
        ${OpenCV_LIBS}
    )

    target_include_directories(InferenceVisualizer PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/models
        ${CMAKE_SOURCE_DIR}/utils
        ${HDF5_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )

    target_compile_definitions(InferenceVisualizer PRIVATE 
        ${HDF5_DEFINITIONS}
    )

    set_target_properties(InferenceVisualizer PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    message(STATUS "✓ InferenceVisualizer target created (POSTER-READY)")
else()
    message(STATUS "✗ InferenceVisualizer target skipped (OpenCV not found)")
endif()
