/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#include "desul/atomics/Common.hpp"

#ifdef DESUL_HAVE_GCC_ATOMICS
#if !defined(DESUL_HAVE_16BYTE_COMPARE_AND_SWAP) && !defined(__CUDACC__)
// This doesn't work in WSL??
//#define DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#endif
namespace desul {

#if defined(__clang__) && (__clang_major__>=7)
// Disable warning for large atomics on clang 7 and up (checked with godbolt)
// error: large atomic operation may incur significant performance penalty [-Werror,-Watomic-alignment]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Watomic-alignment"
#endif
template<class MemoryOrder, class MemoryScope>
void atomic_thread_fence(MemoryOrder, MemoryScope) {
  __atomic_thread_fence(GCCMemoryOrder<MemoryOrder>::value);
}

template <typename T, class MemoryOrder, class MemoryScope>
T atomic_exchange(
    T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  __atomic_exchange(
     dest, &value, &return_val, GCCMemoryOrder<MemoryOrder>::value);
  return return_val;
}

// Failure mode for atomic_compare_exchange_n cannot be RELEASE nor ACQREL so
// Those two get handled separatly.
template <typename T, class MemoryOrder, class MemoryScope>
T atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, GCCMemoryOrder<MemoryOrder>::value, GCCMemoryOrder<MemoryOrder>::value);
  return compare;
}

template <typename T, class MemoryScope>
T atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  return compare;
}

template <typename T, class MemoryScope>
T atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  return compare;
}
#if defined(__clang__) && (__clang_major__>=7)
#pragma GCC diagnostic pop
#endif
}  // namespace desul
#endif
#endif
