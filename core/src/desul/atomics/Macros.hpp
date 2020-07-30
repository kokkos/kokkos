/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_MACROS_HPP_
#define DESUL_ATOMICS_MACROS_HPP_

// Macros

#if defined(__GNUC__) && (!defined(__CUDA_ARCH__) || !defined(__NVCC__))
#define DESUL_HAVE_GCC_ATOMICS
#endif

#ifdef _MSC_VER
#define DESUL_HAVE_MSVC_ATOMICS
#endif

#ifdef __CUDACC__
#define DESUL_HAVE_CUDA_ATOMICS
#endif

#ifdef __CUDA_ARCH__
#define DESUL_HAVE_GPU_LIKE_PROGRESS
#endif

#ifdef DESUL_HAVE_CUDA_ATOMICS
#define DESUL_FORCEINLINE_FUNCTION inline __host__ __device__
#define DESUL_INLINE_FUNCTION inline __host__ __device__
#define DESUL_FUNCTION __host__ __device__
#else
#define DESUL_FORCEINLINE_FUNCTION inline
#define DESUL_INLINE_FUNCTION inline
#define DESUL_FUNCTION
#endif

#if !defined(__CUDA_ARCH__)
#define DESUL_HAVE_FORWARD_PROGRESS
#endif

#endif  // DESUL_ATOMICS_MACROS_HPP_
