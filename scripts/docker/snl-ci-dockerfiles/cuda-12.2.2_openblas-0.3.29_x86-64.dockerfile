FROM registry.access.redhat.com/ubi9:9.6@sha256:f22847515e7909e5e6ae980cb8626ecea2411a4277e59852b1661e0aaffb6df2

# certs may be needed for your system, e.g.
#ADD <your>.crt /etc/pki/ca-trust/source/anchors/<your>.crt
#RUN update-ca-trust

# install to download spack
RUN dnf install -y git

# install spack
RUN git clone --branch v1.0.1 --depth 1 https://github.com/spack/spack.git

# build prerequisites
# openblas wants a fortran compiler
# vim,wget just for convenience
RUN dnf install -y gcc-c++ gcc-gfortran xz unzip bzip2 patch \
                   wget vim

RUN . /spack/share/spack/setup-env.sh \
 && spack bootstrap now

ADD cuda-12.2.2_openblas-0.3.29_x86-64.yaml /cuda-12.2.2_openblas-0.3.29_x86-64.yaml

RUN . /spack/share/spack/setup-env.sh \
 && spack env create my_env /cuda-12.2.2_openblas-0.3.29_x86-64.yaml

RUN . /spack/share/spack/setup-env.sh \
 && spack env activate my_env \
 && for j in {1..4}; do \
      spack install & \
    done; wait $(jobs -p); \
    spack clean -a

# from `spack env activate my_env --sh`
ENV CMAKE_PREFIX_PATH=/cuda
ENV CUDA_HOME=/cuda
ENV MANPATH=/cuda/share/man:"$MANPATH"
ENV NVHPC_CUDA_HOME=/cuda
ENV PATH=/cuda/bin:/spack/bin:"$PATH"
ENV PKG_CONFIG_PATH=/cuda/lib/pkgconfig:/cuda/lib64/pkgconfig:"$PKG_CONFIG_PATH"

LABEL org.opencontainers.image.authors="Carl Pearson <cwpears@sandia.gov>"
