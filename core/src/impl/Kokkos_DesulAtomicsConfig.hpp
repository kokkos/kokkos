// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_DESUL_ATOMICS_CONFIG_HPP
#define KOKKOS_DESUL_ATOMICS_CONFIG_HPP

#if defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL)
#define DESUL_CUDA_ARCH_IS_PRE_PASCAL
#endif

#if defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL) || \
    defined(KOKKOS_ARCH_PASCAL)
#define DESUL_CUDA_ARCH_IS_PRE_VOLTA
#endif

#endif
