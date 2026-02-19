FROM intel/oneapi-basekit:2024.2.1-0-devel-ubuntu22.04@sha256:c148163a0476fad50ad46a473c03dc5ca9058fdf5cba69287a4836b3d9ae8bff

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
