/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_
#define DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_

#include <desul/atomics/Common.hpp>
#include <algorithm>
#include <impl/Kokkos_Error.hpp>

#ifdef KOKKOS_COMPILER_NVHPC
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
#define DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,R) std::enable_if_t<std::is_same_v<T,int> \
	|| std::is_same_v<T,unsigned int> || std::is_same_v<T,unsigned long long> \
	|| std::is_same_v<T,float> || std::is_same_v<T,double>, R>
#else
#define DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,R) std::enable_if_t<std::is_same_v<T,int> \
	|| std::is_same_v<T,unsigned int> || std::is_same_v<T,unsigned long long> \
	|| std::is_same_v<T,float>, R>
#endif
#else
#define DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,R) std::enable_if_t<std::is_arithmetic<T>::value, R>
#endif

namespace desul {
namespace Impl {

// clang-format off
//<editor-fold desc="device_atomic_fetch_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}">
#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_add(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = lptr[0]; lptr[0] += val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_inc(
T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = lptr[0]; lptr[0] += T(1); }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_sub(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr -= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_dec(
T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr -= T(1); }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_mul(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr *= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_div(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr /= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_lshift(
T* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr = *lptr << val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_rshift(
T* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr = *lptr >> val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_fetch_mod(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_fetch_mod(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  tmp = *lptr; *lptr = *lptr % val;
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_fetch_max(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_fetch_max(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  tmp = *lptr; *lptr = std::max(*lptr, val);
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_fetch_max(
int* const ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMax(const_cast<int *>(ptr), val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_fetch_max(
unsigned int* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMax(const_cast<unsigned int *>(ptr), val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_fetch_max(
unsigned long long* const ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMax(const_cast<unsigned long long *>(ptr), val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
T device_atomic_fetch_min(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_fetch_min(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  tmp = *lptr; *lptr = std::min(*lptr, val);
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_fetch_min(
int* const ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMin(const_cast<int *>(ptr), val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_fetch_min(
unsigned int* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMin(const_cast<unsigned int *>(ptr), val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_fetch_min(
unsigned long long* const ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMin(const_cast<unsigned long long *>(ptr), val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_and(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr &= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_or(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr |= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_xor(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr ^= val; }
  return tmp;
}
//</editor-fold>

//<editor-fold desc="device_atomic_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}_fetch">
#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_add_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr += val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_inc_fetch(
    T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr += T(1); tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_sub_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr -= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_dec_fetch(
    T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr -= T(1); tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_mul_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr *= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_div_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr /= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_lshift_fetch(
    T* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr = *lptr << val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_rshift_fetch(
    T* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_host)) {
    *lptr = *lptr >> val; tmp = *lptr;
  } else {
#pragma acc atomic capture
    { *lptr = *lptr >> val; tmp = *lptr; }
  }
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_mod_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_mod_fetch(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  *lptr = *lptr % val; tmp = *lptr;
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_max_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_max_fetch(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  *lptr = std::max(*lptr, val); tmp = *lptr;
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_max_fetch(
int* const ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMax(const_cast<int *>(ptr), val);
  tmp = std::max(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_max_fetch(
unsigned int* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMax(const_cast<unsigned int *>(ptr), val);
  tmp = std::max(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_max_fetch(
unsigned long long* const ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMax(const_cast<unsigned long long *>(ptr), val);
  tmp = std::max(tmp, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
T device_atomic_min_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  if(acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_min_fetch(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  *lptr = std::min(*lptr, val); tmp = *lptr;
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_min_fetch(
int* const ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMin(const_cast<int *>(ptr), val);
  tmp = std::min(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_min_fetch(
unsigned int* const ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMin(const_cast<unsigned int *>(ptr), val);
  tmp = std::min(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_min_fetch(
unsigned long long* const ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMin(const_cast<unsigned long long *>(ptr), val);
  tmp = std::min(tmp, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_and_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr &= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_or_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr |= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_xor_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr ^= val; tmp = *lptr; }
  return tmp;
}
//</editor-fold>

//<editor-fold desc="device_atomic_{store,load}">

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,void) device_atomic_store(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
#pragma acc atomic write
    *ptr = val;
}

#pragma acc routine seq
template <class T, class MemoryOrder>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,void) device_atomic_store(
    T* const ptr, const T val, MemoryOrder, MemoryScopeDevice) {
  device_atomic_store(ptr, val, MemoryOrderRelaxed(), MemoryScopeDevice()); 
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_load(
    const T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T retval{};
#pragma acc atomic read
  retval = *ptr;
  return retval;
}

#pragma acc routine seq
template <class T, class MemoryOrder>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_load(
    const T* const ptr, MemoryOrder, MemoryScopeDevice) {
  T retval{};
  retval = device_atomic_load(ptr, MemoryOrderRelaxed(), MemoryScopeDevice());
  return retval;
}

//</editor-fold>
// clang-format on

}  // namespace Impl
}  // namespace desul

#undef DESUL_IMPL_ATOMICS_OPENACC_PREFIX

#endif
