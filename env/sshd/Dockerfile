#
# Simple sshd container example (for troubleshooting)
#
# Use an official Ubuntu image as the base
FROM ubuntu:latest

# Install OpenSSH server
RUN apt-get update && apt-get install -y openssh-server

# Create the SSH directory and set up a basic configuration
RUN mkdir /var/run/sshd

# Set the root password (for demo purposes, use a secure password in production)
RUN echo 'root:password' | chpasswd

# Allow root login via SSH (for demo purposes)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Expose port 2223
EXPOSE 2223

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D", "-p", "2223"]

