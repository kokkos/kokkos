// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DESUL_ATOMICS_CONFIG_HPP
#define KOKKOS_DESUL_ATOMICS_CONFIG_HPP

#include <impl/Kokkos_NvidiaGpuArchitectures.hpp>

#ifdef KOKKOS_IMPL_ARCH_NVIDIA_GPU

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 60
#define DESUL_CUDA_ARCH_IS_PRE_PASCAL
#endif

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 70
#define DESUL_CUDA_ARCH_IS_PRE_VOLTA
#endif

#if KOKKOS_IMPL_ARCH_NVIDIA_GPU < 90
#define DESUL_CUDA_ARCH_IS_PRE_HOPPER
#endif

#endif

#endif
