# Use the NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Set a label to describe this build
LABEL maintainer="your-email@example.com"

# Install necessary packages
RUN apt-get update && apt-get install -y wget bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
RUN bash ./Mambaforge-Linux-x86_64.sh -b -p /opt/conda
RUN rm ./Mambaforge-Linux-x86_64.sh
RUN /opt/conda/bin/conda init bash
# Set the environment path
ENV PATH=/opt/conda/bin:$PATH

# # Create a new Conda environment named "ubuntu" with Python 3.11
RUN mamba create -y -n ubuntu python=3.11

RUN mamba install -y -n ubuntu cudatoolkit pytorch torchvision torchaudio

# # Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ubuntu", "/bin/bash", "-c"]

# # Make conda environment "ubuntu" activate by default
RUN echo "alias cls=clear" >> ~/.bashrc
RUN echo "conda activate ubuntu" >> ~/.bashrc

# Assuming you'll use bash shell, specify the default command to start it
CMD ["/bin/bash"]
