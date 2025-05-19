/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_

#include <omp.h>

#include <desul/atomics/Adapt_GCC.hpp>
#include <desul/atomics/Common.hpp>
#include <desul/atomics/Thread_Fence_OpenMP.hpp>

namespace desul {
namespace Impl {

template <class T, class MemoryOrder, class MemoryScope>
T host_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  if (!std::is_same<MemoryOrder, MemoryOrderRelaxed>::value) {
    atomic_thread_fence(MemoryOrderAcquire(), MemoryScope());
  }
  T& x = *dest;
#pragma omp atomic capture
  {
    return_val = x;
    x = value;
  }
  if (!std::is_same<MemoryOrder, MemoryOrderRelaxed>::value) {
    atomic_thread_fence(MemoryOrderRelease(), MemoryScope());
  }
  return return_val;
}

// OpenMP doesn't have compare exchange, so we use built-in functions and rely on
// testing that this works Note that means we test this in OpenMPTarget offload regions!
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<atomic_always_lock_free(sizeof(T)), T> host_atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  using cas_t = atomic_compare_exchange_t<T>;
  cas_t retval = __sync_val_compare_and_swap(reinterpret_cast<volatile cas_t*>(dest),
                                             reinterpret_cast<cas_t&>(compare),
                                             reinterpret_cast<cas_t&>(value));
  return reinterpret_cast<T&>(retval);
}

template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!atomic_always_lock_free(sizeof(T)), T>  // FIXME_OPENMP
host_atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope) {
#if 0
  (void)__atomic_compare_exchange(dest,
                                  &compare,
                                  &value,
                                  false,
                                  GCCMemoryOrder<MemoryOrder>::value,
                                  GCCMemoryOrder<MemoryOrder>::value);
#else
  (void)dest;
  (void)value;
#endif
  return compare;
}

#if 0  // FIXME_OPENMP

// Disable warning for large atomics on clang 7 and up (checked with godbolt)
// clang-format off
// error: large atomic operation may incur significant performance penalty [-Werror,-Watomic-alignment]
// clang-format on
#if defined(__clang__) && (__clang_major__ >= 7)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Watomic-alignment"
#endif

// Make 16 byte cas work on host at least
#pragma omp begin declare variant match(device = {kind(host)})
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!atomic_always_lock_free(sizeof(T)) && (sizeof(T) == 16), T>
host_atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  (void)__atomic_compare_exchange(dest,
                                  &compare,
                                  &value,
                                  false,
                                  GCCMemoryOrder<MemoryOrder>::value,
                                  GCCMemoryOrder<MemoryOrder>::value);
  return compare;
}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(nohost)})
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!atomic_always_lock_free(sizeof(T)) && (sizeof(T) == 16), T>
device_atomic_compare_exchange(
    T* /*dest*/, T /*compare*/, T value, MemoryOrder, MemoryScope) {
  // FIXME_OPENMP make sure this never gets called
  return value;
}
#pragma omp end declare variant

#if defined(__clang__) && (__clang_major__ >= 7)
#pragma GCC diagnostic pop
#endif

#endif

}  // namespace Impl
}  // namespace desul

#endif
