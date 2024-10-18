FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

RUN apt-get update
RUN apt-get install -y libnvidia-compute-550

# Install necessary packages
RUN apt-get install -y wget ssh openssh-server

#
# User setup
#
ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

# Create a new user with the provided USERNAME, UID, and GID
RUN groupadd -g $GID $USERNAME \
    && useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME

# Set a password for the user
# RUN echo "$USERNAME:password" | chpasswd

# Set the user and prevent the container from running as root
USER $USERNAME

# Set the working directory
WORKDIR /home/$USERNAME

# Download and install Mambaforge for the non-root user
# RUN wget -qO- https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh | bash -s -- -b
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
RUN bash ./Mambaforge-Linux-x86_64.sh -b -p /home/$USERNAME/dconda
RUN rm Mambaforge-Linux-x86_64.sh

# Set the environment path
ENV PATH /home/$USERNAME/dconda/bin:$PATH

# # Create a Python environment called 'ubuntu'
RUN mamba create -n ubuntu python=3.11 -y

# # Activate the environment automatically on login
RUN echo "source /home/${USER}/dconda/bin/activate ubuntu" >> /home/$USERNAME/.bashrc
RUN echo "alias cls=clear" >> /home/$USERNAME/.bashrc


# # Switch back to root to configure SSH
USER root
RUN mkdir /var/run/sshd
# RUN echo 'myuser:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# # Expose the default SSH port
EXPOSE 22

# # Start the SSH daemon
# CMD ["/usr/sbin/sshd", "-D"]
CMD ["/bin/bash"]
