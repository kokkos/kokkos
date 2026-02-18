FROM intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04

# certs may be needed for your system, e.g.
#ADD <your>.crt /etc/pki/ca-trust/source/anchors/<your>.crt
#RUN update-ca-trust

# Base packages for ubuntu
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        bc \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Base developer packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        vim \
        cmake \
        build-essential gfortran \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
