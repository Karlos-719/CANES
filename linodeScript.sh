#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install build-essential linux-headers-$(uname -r) -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
sudo apt-get update -y
sudo apt-get -y install cuda

sudo apt update -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.9 -y
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libhdf5-dev \
    unzip \
    gdb \
    cuda-runtime-12-9
wget -o- https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip

# export LD_LIBRARY_PATH="/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}"
# NOTE TO FUTURE
# THE 12.0 RUNTIME does not have v2 and v3 versions of some methods need 12.9. AIt can be installed but it is not linked against, to link against it it needs to be in ld library path

reboot -h now