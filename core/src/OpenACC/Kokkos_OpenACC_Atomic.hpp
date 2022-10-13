/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_OPENACC_ATOMIC_HPP
#define KOKKOS_OPENACC_ATOMIC_HPP

#include <impl/Kokkos_Atomic_Memory_Order.hpp>
#include <impl/Kokkos_Memory_Fence.hpp>
#include <openacc.h>
#include <stdio.h>
#include <algorithm>

//#define KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS_ERROR

#if defined(KOKKOS_ENABLE_OPENACC_ATOMICS)
namespace Kokkos {
// OpenACC can do:
// Operations: +, -, *, /, &, |, ^, <<, >>
// Type: int, unsigned int, unsigned long long int
// Variants:
// atomic_exchange/compare_exchange/assign/fetch_add/fetch_sub/fetch_mul/fetch_div/fetch_mod/fetch_max/fetch_min/fetch_and/fetch_or/fetch_xor/fetch_lshift/fetch_rshift/add_fetch/sub_fetch/mul_fetch/div_fetch/mod_fetch/max_fetch/min_fetch/and_fetch/or_fetch/xor_fetch/lshift_fetch/rshift_fetch/
// _atomic_load/store

// atomic_exchange -------------------------------------------------------------

#pragma acc routine seq
template <typename T>
__inline__ T atomic_exchange(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_exchange()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = *ptr;
    *ptr   = val;
  }
  return retval;
}

#pragma acc routine seq
template <>
int atomic_exchange(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicExch((int *)dest, val);
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = *ptr;
    *ptr   = val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
unsigned int atomic_exchange(volatile unsigned int *const dest,
                             const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicExch((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = *ptr;
    *ptr   = val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
unsigned long long int atomic_exchange(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicExch((unsigned long long int *)dest, val);
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = *ptr;
    *ptr   = val;
  }
#endif
  return retval;
}

#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
#pragma acc routine seq
template <>
float atomic_exchange(volatile float *const dest, const float &val) {
  float retval;
  retval = atomicExch((float *)dest, val);
  return retval;
}
#endif

// atomic_compare_exchange -----------------------------------------------------

#pragma acc routine seq
template <typename T>
__inline__ T atomic_compare_exchange(volatile T *const dest, const T &compare,
                                     const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_compare_exchange()] Not supported data type in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = *ptr;
    if (retval == compare) *ptr = val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_compare_exchange(volatile int *const dest,
                                       const int &compare, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicCAS((int *)dest, compare, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_compare_exchange()] Not supported atomic operation "
        "in OpenACC; "
        "exit!\n");
  }
  int *ptr = const_cast<int *>(dest);
  {
    retval = *ptr;
    if (retval == compare) *ptr = val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_compare_exchange(
    volatile unsigned int *const dest, const unsigned int &compare,
    const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicCAS((unsigned int *)dest, compare, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_compare_exchange()] Not supported atomic operation "
        "in OpenACC; "
        "exit!\n");
  }
  unsigned int *ptr = const_cast<unsigned int *>(dest);
  {
    retval = *ptr;
    if (retval == compare) *ptr = val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_compare_exchange(
    volatile unsigned long long int *const dest,
    const unsigned long long int &compare, const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicCAS((unsigned long long int *)dest, compare, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_compare_exchange()] Not supported atomic operation "
        "in OpenACC; "
        "exit!\n");
  }
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
  {
    retval = *ptr;
    if (retval == compare) *ptr = val;
  }
#endif
  return retval;
}

// atomic_assign ---------------------------------------------------------------

#pragma acc routine seq
template <typename T>
__inline__ void atomic_assign(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_assign()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T *ptr = const_cast<T *>(dest);
  ptr[0] = val;
}

#pragma acc routine seq
template <>
__inline__ void atomic_assign(volatile int *const dest, const int &val) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch((int *)dest, val);
#else
  int *ptr                    = const_cast<int *>(dest);
#pragma acc atomic write
  ptr[0]                      = val;
#endif
}

#pragma acc routine seq
template <>
__inline__ void atomic_assign(volatile unsigned int *const dest,
                              const unsigned int &val) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch((unsigned int *)dest, val);
#else
  unsigned int *ptr           = const_cast<unsigned int *>(dest);
#pragma acc atomic write
  ptr[0]                      = val;
#endif
}

#pragma acc routine seq
template <>
__inline__ void atomic_assign(volatile unsigned long long int *const dest,
                              const unsigned long long int &val) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch((unsigned long long int *)dest, val);
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic write
  ptr[0]                      = val;
#endif
}

// // atomic_fetch_add
// ------------------------------------------------------------

#pragma acc routine seq
template <typename T>
inline T atomic_fetch_add(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_add()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] += val;
  }
  return retval;
}

#pragma acc routine seq
template <>
inline int atomic_fetch_add(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((int *)dest, val);
#else
  int *ptr                    = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] += val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
inline unsigned int atomic_fetch_add(volatile unsigned int *const dest,
                                     const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] += val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
inline unsigned long long int atomic_fetch_add(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((unsigned long long int *)dest, val);
#else
  unsigned long long *ptr = const_cast<unsigned long long *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] += val;
  }
#endif
  return retval;
}

/*
#pragma acc routine seq
template <>
inline  int64_t atomic_fetch_add(
    volatile int64_t *const dest,
    const int64_t &val) {
    int64_t retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
        retval = atomicAdd((int64_t *)dest, val);
#else
        int64_t *ptr = const_cast<int64_t *>(dest);
#pragma acc atomic capture
    {
      retval = ptr[0];
      ptr[0] += val;
    }
#endif
    return retval;
}
*/

// atmic_fetch_sub -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_sub(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_sub()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] -= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_sub(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicSub((int *)dest, val);
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] -= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_sub(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicSub((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] -= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_sub(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] -= val;
  }
  return retval;
}

// atmic_fetch_mul -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_mul(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_mul()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] *= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_mul(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] *= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_mul(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] *= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_mul(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] *= val;
  }
  return retval;
}

// atmic_fetch_div -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_div(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_div()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] /= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_div(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] /= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_div(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] /= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_div(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] /= val;
  }
  return retval;
}

// atmic_fetch_lshift
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_lshift(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_lshift()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] = ptr[0] << val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_lshift(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] << val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_lshift(volatile unsigned int *const dest,
                                            const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] << val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_lshift(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] << val;
  }
  return retval;
}

// atmic_fetch_rshift
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_rshift(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_rshift()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] = ptr[0] >> val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_rshift(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] >> val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_rshift(volatile unsigned int *const dest,
                                            const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] >> val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_rshift(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] = ptr[0] >> val;
  }
  return retval;
}

// atomic_fetch_mod
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_mod(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_mod()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  retval = ptr[0];
  ptr[0] = ptr[0] % val;
  return retval;
}

// atomic_fetch_max
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_max(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_max()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  retval = ptr[0];
  ptr[0] = std::max(ptr[0], val);
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_max(volatile int *const dest, const int &val) {
  int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_max()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  int *ptr = const_cast<int *>(dest);
  retval   = ptr[0];
  ptr[0]   = std::max(ptr[0], val);
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_max(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((unsigned int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_max()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned int *ptr = const_cast<unsigned int *>(dest);
  retval            = ptr[0];
  ptr[0]            = std::max(ptr[0], val);
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_max(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((unsigned long long int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_max()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
  retval                      = ptr[0];
  ptr[0]                      = std::max(ptr[0], val);
#endif
  return retval;
}

// atomic_max
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_max(volatile T *const dest, const T &val) {
  return atomic_fetch_max(dest, val);
}

// atomic_fetch_min
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_min(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_min()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  retval = ptr[0];
  ptr[0] = std::min(ptr[0], val);
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_min(volatile int *const dest, const int &val) {
  int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_min()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  int *ptr = const_cast<int *>(dest);
  retval   = ptr[0];
  ptr[0]   = std::min(ptr[0], val);
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_min(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((unsigned int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_min()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned int *ptr = const_cast<unsigned int *>(dest);
  retval            = ptr[0];
  ptr[0]            = std::min(ptr[0], val);
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_min(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((unsigned long long int *)dest, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_min()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
  retval                      = ptr[0];
  ptr[0]                      = std::min(ptr[0], val);
#endif
  return retval;
}

// atomic_min
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_min(volatile T *const dest, const T &val) {
  return atomic_fetch_min(dest, val);
}

// atomic_fetch_or -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_or(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_or()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] |= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_or(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicOr((int *)dest, val);
#else
  int *ptr                    = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] |= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_or(volatile unsigned int *const dest,
                                        const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicOr((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] |= val;
  }
#endif
  return retval;
}

//[FIXME_OPENACC] CUDA intrinsic: Unhandled builtin: 609 (__pgi_atomicOrul),
// OpenACC
// atomic: wrong result, Serial template: Unsupported local variable
#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_or(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS_ERROR
  retval = atomicOr((unsigned long long int *)dest, val);
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] |= val;
  }
#endif
  return retval;
}

// atomic_fetch_and ------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_and(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_and()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] &= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_and(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAnd((int *)dest, val);
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] &= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_and(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAnd((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] &= val;
  }
#endif
  return retval;
}

//[FIXME_OPENACC] CUDA intrinsic: Unhandled builtin: 609 (__pgi_atomicAndul),
// OpenACC
// atomic: wrong result, Serial template: Unsupported local variable
#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_and(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS_ERROR
  //[ERROR with NVHPC V22.2] Unhandled builtin: 605 (__pgi_atomicAndul)
  retval = atomicAnd((unsigned long long int *)dest, val);
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] &= val;
  }
#endif
  return retval;
}

// atomic_fetch_xor ------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_fetch_xor(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_fetch_xor()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    retval = ptr[0];
    ptr[0] ^= val;
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_fetch_xor(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((int *)dest, val);
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] ^= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_fetch_xor(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((unsigned int *)dest, val);
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] ^= val;
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_fetch_xor(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((unsigned long long int *)dest, val);
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    retval = ptr[0];
    ptr[0] ^= val;
  }
#endif
  return retval;
}

// // atomic_add_fetch
// ------------------------------------------------------------

#pragma acc routine seq
template <typename T>
inline T atomic_add_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_add_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] += val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
inline int atomic_add_fetch(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((int *)dest, val);
  retval = retval + val;
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] += val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
inline unsigned int atomic_add_fetch(volatile unsigned int *const dest,
                                     const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((unsigned int *)dest, val);
  retval = retval + val;
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] += val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
inline unsigned long long int atomic_add_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd((unsigned long long int *)dest, val);
  retval = retval + val;
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] += val;
    retval = ptr[0];
  }
#endif
  return retval;
}

// atmic_sub_fetch -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_sub_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_sub_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] -= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_sub_fetch(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicSub((int *)dest, val);
  retval = retval - val;
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] -= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_sub_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicSub((unsigned int *)dest, val);
  retval = retval - val;
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] -= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

// atmic_mul_fetch -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_mul_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_mul_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] *= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_mul_fetch(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] *= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_mul_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] *= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_mul_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] *= val;
    retval = ptr[0];
  }
  return retval;
}

// atmic_div_fetch -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_div_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_div_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] /= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_div_fetch(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] /= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_div_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] /= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_div_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] /= val;
    retval = ptr[0];
  }
  return retval;
}

// atmic_lshift_fetch
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_lshift_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_lshift_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] = ptr[0] << val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_lshift_fetch(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] << val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_lshift_fetch(volatile unsigned int *const dest,
                                            const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] << val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_lshift_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] << val;
    retval = ptr[0];
  }
  return retval;
}

// atmic_rshift_fetch
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_rshift_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_rshift_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] = ptr[0] >> val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_rshift_fetch(volatile int *const dest, const int &val) {
  int retval;
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] >> val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_rshift_fetch(volatile unsigned int *const dest,
                                            const unsigned int &val) {
  unsigned int retval;
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] >> val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_rshift_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] = ptr[0] >> val;
    retval = ptr[0];
  }
  return retval;
}

// atomic_mod_fetch
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_mod_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_mod_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  ptr[0] = ptr[0] % val;
  retval = ptr[0];
  return retval;
}

// atomic_max_fetch
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_max_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_max_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  ptr[0] = std::max(ptr[0], val);
  retval = ptr[0];
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_max_fetch(volatile int *const dest, const int &val) {
  int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((int *)dest, val);
  retval = std::max(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_max_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  int *ptr = const_cast<int *>(dest);
  ptr[0]   = std::max(ptr[0], val);
  retval   = ptr[0];
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_max_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((unsigned int *)dest, val);
  retval = std::max(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_max_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned int *ptr = const_cast<unsigned int *>(dest);
  ptr[0]            = std::max(ptr[0], val);
  retval            = ptr[0];
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_max_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMax((unsigned long long int *)dest, val);
  retval = std::max(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_max_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
  ptr[0]                      = std::max(ptr[0], val);
  retval                      = ptr[0];
#endif
  return retval;
}

// atomic_min_fetch
// -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_min_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_min_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  ptr[0] = std::min(ptr[0], val);
  retval = ptr[0];
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_min_fetch(volatile int *const dest, const int &val) {
  int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((int *)dest, val);
  retval = std::min(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_min_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  int *ptr = const_cast<int *>(dest);
  ptr[0]   = std::min(ptr[0], val);
  retval   = ptr[0];
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_min_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((unsigned int *)dest, val);
  retval = std::min(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_min_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned int *ptr = const_cast<unsigned int *>(dest);
  ptr[0]            = std::min(ptr[0], val);
  retval            = ptr[0];
#endif
  return retval;
}
#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_min_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval = 0;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicMin((unsigned long long int *)dest, val);
  retval = std::min(retval, val);
#else
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_min_fetch()] Not supported atomic operation in "
        "OpenACC; "
        "exit!\n");
  }
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
  ptr[0]                      = std::min(ptr[0], val);
  retval                      = ptr[0];
#endif
  return retval;
}

// atomic_or_fetch -------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_or_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_or_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] |= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_or_fetch(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicOr((int *)dest, val);
  retval = retval | val;
#else
  int *ptr                    = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] |= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_or_fetch(volatile unsigned int *const dest,
                                        const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicOr((unsigned int *)dest, val);
  retval = retval | val;
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] |= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

//[FIXME_OPENACC] CUDA intrinsic: Unhandled builtin: 609 (__pgi_atomicOrul),
// OpenACC
// atomic: wrong result, Serial template: Unsupported local variable
#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_or_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS_ERROR
  retval = atomicOr((unsigned long long int *)dest, val);
  retval = retval | val;
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] |= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

// atomic_and_fetch ------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_and_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_and_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] &= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_and_fetch(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAnd((int *)dest, val);
  retval = retval & val;
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] &= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_and_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAnd((unsigned int *)dest, val);
  retval = retval & val;
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] &= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

//[FIXME_OPENACC] CUDA intrinsic: Unhandled builtin: 609 (__pgi_atomicAnul),
// OpenACC
// atomic: wrong result, Serial template: Unsupported local variable
#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_and_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS_ERROR
  retval = atomicAnd((unsigned long long int *)dest, val);
  retval = retval & val;
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] &= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

// atomic_xor_fetch ------------------------------------------------------------

#pragma acc routine seq
template <class T>
__inline__ T atomic_xor_fetch(volatile T *const dest, const T &val) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_xor_fetch()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  T retval;
  T *ptr = const_cast<T *>(dest);
  {
    ptr[0] ^= val;
    retval = ptr[0];
  }
  return retval;
}

#pragma acc routine seq
template <>
__inline__ int atomic_xor_fetch(volatile int *const dest, const int &val) {
  int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((int *)dest, val);
  retval = retval ^ val;
#else
  int *ptr = const_cast<int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] ^= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned int atomic_xor_fetch(volatile unsigned int *const dest,
                                         const unsigned int &val) {
  unsigned int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((unsigned int *)dest, val);
  retval = retval ^ val;
#else
  unsigned int *ptr = const_cast<unsigned int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] ^= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

#pragma acc routine seq
template <>
__inline__ unsigned long long int atomic_xor_fetch(
    volatile unsigned long long int *const dest,
    const unsigned long long int &val) {
  unsigned long long int retval;
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicXor((unsigned long long int *)dest, val);
  retval = retval ^ val;
#else
  unsigned long long int *ptr = const_cast<unsigned long long int *>(dest);
#pragma acc atomic capture
  {
    ptr[0] ^= val;
    retval = ptr[0];
  }
#endif
  return retval;
}

namespace Impl {

#pragma acc routine seq
template <typename T, typename MemoryOrder>
__inline__ void _atomic_store(T *ptr, T val, MemoryOrder) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort(
        "[ERROR in atomic_store()] Not supported data type in OpenACC; "
        "exit!\n");
  }
  ptr[0] = val;
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ void _atomic_store(int *ptr, int val, MemoryOrder) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch(ptr, val);
#else
#pragma acc atomic write
  ptr[0] = val;
#endif
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ void _atomic_store(unsigned int *ptr, unsigned int val,
                              MemoryOrder) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch(ptr, val);
#else
#pragma acc atomic write
  ptr[0] = val;
#endif
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ void _atomic_store(unsigned long long int *ptr,
                              unsigned long long int val, MemoryOrder) {
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  atomicExch(ptr, val);
#else
#pragma acc atomic write
  ptr[0] = val;
#endif
}

#pragma acc routine seq
template <typename T, typename MemoryOrder>
__inline__ T _atomic_load(T *ptr, MemoryOrder) {
  T retval{};
  retval = ptr[0];
  return retval;
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ int _atomic_load(int *ptr, MemoryOrder) {
  int retval{};
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd(ptr, 0);
#else
#pragma acc atomic read
  retval = ptr[0];
#endif
  return retval;
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ unsigned int _atomic_load(unsigned int *ptr, MemoryOrder) {
  unsigned int retval{};
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd(ptr, 0);
#else
#pragma acc atomic read
  retval = ptr[0];
#endif
  return retval;
}

#pragma acc routine seq
template <typename MemoryOrder>
__inline__ unsigned long long int _atomic_load(unsigned long long int *ptr,
                                               MemoryOrder) {
  unsigned long long int retval{};
#ifdef KOKKOS_ENABLE_CUDA_ATOMIC_INTRINSICS
  retval = atomicAdd(ptr, 0);
#else
#pragma acc atomic read
  retval = ptr[0];
#endif
  return retval;
}

}  // namespace Impl
}  // namespace Kokkos
#endif

#endif
