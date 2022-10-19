/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_MACROS_HPP_
#define DESUL_ATOMICS_MACROS_HPP_

// Macros

#if (!defined(__CUDA_ARCH__) || !defined(__NVCC__)) &&                    \
    (!defined(__HIP_DEVICE_COMPILE) || !defined(__HIP_PLATFORM_HCC__)) && \
    !defined(__SYCL_DEVICE_ONLY__) && !defined(DESUL_HAVE_OPENMP_ATOMICS)
#define DESUL_IMPL_HAVE_GCC_OR_MSVC_ATOMICS
#endif

// ONLY use GNUC atomics if not compiling for the device
// and we didn't explicitly say to use OpenMP atomics
#if defined(__GNUC__) && defined(DESUL_IMPL_HAVE_GCC_OR_MSVC_ATOMICS)
#define DESUL_HAVE_GCC_ATOMICS
#endif

// Equivalent to above: if we are compiling for the device we
// need to use CUDA/HIP/SYCL atomics instead of MSVC atomics
#if defined(_MSC_VER) && defined(DESUL_IMPL_HAVE_GCC_OR_MSVC_ATOMICS)
#define DESUL_HAVE_MSVC_ATOMICS
#endif

#undef DESUL_IMPL_HAVE_GCC_OR_MSVC_ATOMICS

#ifdef __CUDACC__
#define DESUL_HAVE_CUDA_ATOMICS
#endif

#ifdef __HIPCC__
#define DESUL_HAVE_HIP_ATOMICS
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define DESUL_HAVE_SYCL_ATOMICS
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__SYCL_DEVICE_ONLY__)
#define DESUL_HAVE_GPU_LIKE_PROGRESS
#endif

#if defined(DESUL_HAVE_CUDA_ATOMICS) || defined(DESUL_HAVE_HIP_ATOMICS)
#define DESUL_FORCEINLINE_FUNCTION inline __host__ __device__
#define DESUL_INLINE_FUNCTION inline __host__ __device__
#define DESUL_FUNCTION __host__ __device__
#define DESUL_IMPL_HOST_FUNCTION __host__
#define DESUL_IMPL_DEVICE_FUNCTION __device__
#else
#define DESUL_FORCEINLINE_FUNCTION inline
#define DESUL_INLINE_FUNCTION inline
#define DESUL_FUNCTION
#define DESUL_IMPL_HOST_FUNCTION
#define DESUL_IMPL_DEVICE_FUNCTION
#endif

#if !defined(DESUL_HAVE_GPU_LIKE_PROGRESS)
#define DESUL_HAVE_FORWARD_PROGRESS
#endif

#define DESUL_IMPL_STRIP_PARENS(X) DESUL_IMPL_ESC(DESUL_IMPL_ISH X)
#define DESUL_IMPL_ISH(...) DESUL_IMPL_ISH __VA_ARGS__
#define DESUL_IMPL_ESC(...) DESUL_IMPL_ESC_(__VA_ARGS__)
#define DESUL_IMPL_ESC_(...) DESUL_IMPL_VAN_##__VA_ARGS__
#define DESUL_IMPL_VAN_DESUL_IMPL_ISH

#if defined(__CUDACC__) && defined(__NVCOMPILER)
#include <nv/target>
#define DESUL_IF_ON_DEVICE(CODE) NV_IF_TARGET(NV_IS_DEVICE, CODE)
#define DESUL_IF_ON_HOST(CODE) NV_IF_TARGET(NV_IS_HOST, CODE)
#endif

// FIXME OpenMP Offload differentiate between device and host, but do we need this?
#if defined(DESUL_HAVE_OPENMP_ATOMICS)
#if 0
// Base function.
static constexpr bool desul_impl_omp_on_host() { return true; }

#pragma omp begin declare variant match(device = {kind(host)})
static constexpr bool desul_impl_omp_on_host() { return true; }
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(nohost)})
static constexpr bool desul_impl_omp_on_host() { return false; }
#pragma omp end declare variant

#define DESUL_IF_ON_DEVICE(CODE)             \
  if constexpr (!desul_impl_omp_on_host()) { \
    DESUL_IMPL_STRIP_PARENS(CODE)            \
  }
#define DESUL_IF_ON_HOST(CODE)              \
  if constexpr (desul_impl_omp_on_host()) { \
    DESUL_IMPL_STRIP_PARENS(CODE)           \
  }
#else
#define DESUL_IF_ON_DEVICE(CODE) \
  {}
#define DESUL_IF_ON_HOST(CODE) \
  { DESUL_IMPL_STRIP_PARENS(CODE) }
#endif
#endif

#if !defined(DESUL_IF_ON_HOST) && !defined(DESUL_IF_ON_DEVICE)
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || \
    defined(__SYCL_DEVICE_ONLY__)
#define DESUL_IF_ON_DEVICE(CODE) \
  { DESUL_IMPL_STRIP_PARENS(CODE) }
#define DESUL_IF_ON_HOST(CODE) \
  {}
#else
#define DESUL_IF_ON_DEVICE(CODE) \
  {}
#define DESUL_IF_ON_HOST(CODE) \
  { DESUL_IMPL_STRIP_PARENS(CODE) }
#endif
#endif

#endif  // DESUL_ATOMICS_MACROS_HPP_
