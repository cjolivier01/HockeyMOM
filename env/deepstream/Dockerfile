FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

RUN apt-get install -y sudo nasm gnutls-dev libx264-dev ffmpeg aptitude

ARG USERNAME
ARG UID
ARG GID

ENV SHELL=/bin/bash

# Create a new user with the provided USERNAME, UID, and GID
USER root
RUN echo '%sudo   ALL=(ALL:ALL) ALL' >> /etc/sudoers
RUN echo "$USERNAME ALL=(ALL:ALL) ALL" >> /etc/sudoers
# RUN cat /etc/sudoers
# RUN usermod -aG sudo $USERNAME

# Set a password for the user
# RUN echo "$USERNAME:password" | chpasswd

# Set the user and prevent the container from running as root
USER $USERNAME

# Set the working directory
WORKDIR /home/$USERNAME


EXPOSE 22298

# Start the SSH daemon
# CMD ["/usr/sbin/sshd", "-d", "-p", "22298"]
CMD ["/bin/bash"]
