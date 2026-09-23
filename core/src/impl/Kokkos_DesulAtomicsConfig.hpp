// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DESUL_ATOMICS_CONFIG_HPP
#define KOKKOS_DESUL_ATOMICS_CONFIG_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_NvidiaGpuArchitectures.hpp>

#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
#include <cuda.h>

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 60
#define DESUL_CUDA_ARCH_IS_PRE_PASCAL
#endif

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 70
#define DESUL_CUDA_ARCH_IS_PRE_VOLTA
#endif

// the load, store, and CAS instructions require PTX ISA 840 which is in
// cuda 12.4. and Clang > 19
// Furthermore, CAS is only available for >=Hopper and we need all three for the
// atomics
#if KOKKOS_IMPL_ARCH_NVIDIA_GPU >= 90 && CUDA_VERSION >= 12040 && \
    (!defined(KOKKOS_COMPILER_CLANG) || KOKKOS_COMPILER_CLANG >= 1900)
#define DESUL_HAVE_16BYTE_LOCK_FREE_ATOMICS_DEVICE
#endif

#endif

#endif
