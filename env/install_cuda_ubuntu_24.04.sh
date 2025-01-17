#!/bin/bash

# Remove any existing NVIDIA repository configuration
sudo rm -f /etc/apt/sources.list.d/cuda*
sudo rm -f /etc/apt/sources.list.d/nvidia*

# Install required packages
sudo apt-get update
sudo apt-get install -y wget gpg-agent

# Download NVIDIA signing key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

# Install the key package
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Add CUDA repository
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | \
    sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

# Pin repository priority
echo -e "Package: *\nPin: origin developer.download.nvidia.com\nPin-Priority: 600" | \
    sudo tee /etc/apt/preferences.d/cuda-repository-pin-600

# Update package list
sudo apt-get update

echo "CUDA repository has been successfully added!"
echo "You can now install CUDA using: sudo apt-get install cuda"
echo "Or install specific versions using: sudo apt-get install cuda-<version>"
echo "Example: sudo apt-get install cuda-12-2"
