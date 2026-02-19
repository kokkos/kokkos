FROM registry.access.redhat.com/ubi9:9.5@sha256:a010f5a7096ea45d77c2df0779c28b1829112816d6fb248bb2d2ed0e04f94498

# certs may be needed for your system, e.g.
#ADD <your>.crt /etc/pki/ca-trust/source/anchors/<your>.crt
#RUN update-ca-trust

# install to download spack
RUN dnf install -y git

# install spack
RUN git clone --branch v0.23.1 --depth 1 https://github.com/spack/spack.git

# build prerequisites
# openblas wants a fortran compiler
# perl-file-which needs make but doesn't declare it as a dep in this spack
RUN dnf install -y gcc-c++ gcc-gfortran xz unzip bzip2 patch make

RUN . /spack/share/spack/setup-env.sh \
 && spack bootstrap now

ADD rocm-6.2.1_openblas-0.3.28_zen3_gfx90a.yaml /rocm-6.2.1_openblas-0.3.28_zen3_gfx90a.yaml

RUN . /spack/share/spack/setup-env.sh \
 && spack env create my_env /rocm-6.2.1_openblas-0.3.28_zen3_gfx90a.yaml

RUN . /spack/share/spack/setup-env.sh \
 && spack env activate my_env \
 && for j in {1..4}; do \
      spack install & \
    done; wait $(jobs -p)

# perl in hipcc complained about this not being set
# RUN dnf install -y locales
# RUN dnf install -y glibc-langpack-en
# RUN localedef -v -c -i en_US -f UTF-8 en_US.UTF-8
ENV LANGUAGE="C"
ENV LC_ALL="C"
ENV LANG="C"

# from `spack activate env my_env --sh
ENV CMAKE_PREFIX_PATH=/rocm:.:"$CMAKE_PREFIX_PATH"
ENV HIPCC_COMPILE_FLAGS_APPEND='--rocm-path=/rocm'
ENV LD_LIBRARY_PATH=/rocm/lib:."$LD_LIBRARY_PATH"
ENV LUA_PATH='/rocm/lib/lua/5.3/?/init.lua;/rocm/lib/lua/5.3/?.lua;/rocm/share/lua/5.3/?/init.lua;/rocm/share/lua/5.3/?.lua;/rocm/lib/lua/5.3/?/init.lua;/rocm/lib/lua/5.3/?.lua;/rocm/share/lua/5.3/?/init.lua;/rocm/share/lua/5.3/?.lua;'"$LUA_PATH"
ENV MANPATH=/rocm/share/man:"$MANPATH":
ENV PATH=/rocm/bin:/rocm:"$PATH"
ENV PERL5LIB=/rocm/lib/perl5:.:"$PERL5LIB"
ENV XLOCALEDIR=/rocm/share/X11/locale:.:"$XLOCALEDIR"

# env | grep HIP
ENV HIPIFY_CLANG_PATH=/rocm
ENV HIP_PLATFORM=amd
ENV HIP_DEVICE_LIB_PATH=/rocm/amdgcn/bitcode
ENV HIP_CLANG_PATH=/rocm/bin
ENV HIP_COMPILER=clang

# env | grep ROCM
ENV ROCM_PATH=/rocm
ENV ROCMINFO_PATH=/rocm

# env | grep AMD
ENV HCC_AMDGPU_TARGET=gfx90a

LABEL org.opencontainers.image.authors="Carl Pearson <cwpears@sandia.gov>"
