/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_OPENMP_HPP_
#include "desul/atomics/Common.hpp"
#include<cstdio>

#ifdef DESUL_HAVE_OPENMP_ATOMICS
namespace desul {

#if _OPENMP > 201800
// atomic_thread_fence for Core Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeCore) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeCore) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore) {
  #pragma omp flush release
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore) {
  #pragma omp flush acquire
}
// atomic_thread_fence for Device Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeDevice) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeDevice) {
  #pragma omp flush acq_rel
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) {
  #pragma omp flush release
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) {
  #pragma omp flush acquire
}
#else
// atomic_thread_fence for Core Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeCore) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeCore) {
  #pragma omp flush
}
// atomic_thread_fence for Device Scope
inline void atomic_thread_fence(MemoryOrderSeqCst, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcqRel, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderRelease, MemoryScopeDevice) {
  #pragma omp flush
}
inline void atomic_thread_fence(MemoryOrderAcquire, MemoryScopeDevice) {
  #pragma omp flush
}
#endif

template <typename T, class MemoryOrder, class MemoryScope>
T atomic_exchange(
    T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  if(!std::is_same<MemoryOrder,MemoryOrderRelaxed>::value)
    atomic_thread_fence(MemoryOrderAcquire(),MemoryScope());
  T& x = *dest;
  #pragma omp atomic capture
  { return_val = x; x = value; }
  if(!std::is_same<MemoryOrder,MemoryOrderRelaxed>::value)
    atomic_thread_fence(MemoryOrderRelease(),MemoryScope());
  return return_val;
}

// OpenMP doesn't have compare exchange, so we use build-ins and rely on testing that this works
// Note that means we test this in OpenMPTarget offload regions!
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<Impl::atomic_always_lock_free(sizeof(T)),T> atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  using cas_t = typename Impl::atomic_compare_exchange_type<sizeof(T)>::type;
  cas_t retval = __sync_val_compare_and_swap(
     reinterpret_cast<volatile cas_t*>(dest), 
     reinterpret_cast<cas_t&>(compare), 
     reinterpret_cast<cas_t&>(value));
  return reinterpret_cast<T&>(retval);
}

}  // namespace desul
#endif
#endif
