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
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = ptr[0]; ptr[0] += val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_inc(
T* ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = ptr[0]; ptr[0] += T(1); }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_sub(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr -= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_dec(
T* ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr -= T(1); }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_mul(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr *= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_div(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr /= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_lshift(
T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr = *ptr << val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_rshift(
T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr = *ptr >> val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_fetch_mod(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_fetch_mod(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  tmp = *ptr; *ptr = *ptr % val;
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_fetch_max(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_fetch_max(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  tmp = *ptr; *ptr = std::max(*ptr, val);
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_fetch_max(
int* ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMax(ptr, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_fetch_max(
unsigned int* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMax(ptr, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_fetch_max(
unsigned long long* ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMax(ptr, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
T device_atomic_fetch_min(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_fetch_min(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  tmp = *ptr; *ptr = std::min(*ptr, val);
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_fetch_min(
int* ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMin(ptr, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_fetch_min(
unsigned int* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMin(ptr, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_fetch_min(
unsigned long long* ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMin(ptr, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_and(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr &= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_or(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr |= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_fetch_xor(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { tmp = *ptr; *ptr ^= val; }
  return tmp;
}
//</editor-fold>

//<editor-fold desc="device_atomic_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}_fetch">
#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_add_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr += val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_inc_fetch(
    T* ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr += T(1); tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_sub_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr -= val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_dec_fetch(
    T* ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr -= T(1); tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_mul_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr *= val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_div_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr /= val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_lshift_fetch(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr = *ptr << val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_rshift_fetch(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_host)) {
    *ptr = *ptr >> val; tmp = *ptr;
  } else {
#pragma acc atomic capture
    { *ptr = *ptr >> val; tmp = *ptr; }
  }
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_mod_fetch(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_mod_fetch(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  *ptr = *ptr % val; tmp = *ptr;
  return tmp;
}

#pragma acc routine seq
template <class T>
T device_atomic_max_fetch(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_max_fetch(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  *ptr = std::max(*ptr, val); tmp = *ptr;
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_max_fetch(
int* ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMax(ptr, val);
  tmp = std::max(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_max_fetch(
unsigned int* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMax(ptr, val);
  tmp = std::max(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_max_fetch(
unsigned long long* ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMax(ptr, val);
  tmp = std::max(tmp, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
T device_atomic_min_fetch(
T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  if(acc_on_device(acc_device_not_host)) {
    printf("Kokkos Error in device_atomic_min_fetch(): Not supported atomic "
                  "operation in the OpenACC backend\n");
  }
  *ptr = std::min(*ptr, val); tmp = *ptr;
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
inline int device_atomic_min_fetch(
int* ptr, const int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  int tmp;
  tmp = atomicMin(ptr, val);
  tmp = std::min(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned int device_atomic_min_fetch(
unsigned int* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned int tmp;
  tmp = atomicMin(ptr, val);
  tmp = std::min(tmp, val);
  return tmp;
}

#pragma acc routine seq
inline unsigned long long device_atomic_min_fetch(
unsigned long long* ptr, const unsigned long long val, MemoryOrderRelaxed, MemoryScopeDevice) {
  unsigned long long tmp;
  tmp = atomicMin(ptr, val);
  tmp = std::min(tmp, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_and_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr &= val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_or_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr |= val; tmp = *ptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
DESUL_IMPL_ATOMICS_OPENACC_PREFIX(T,T) device_atomic_xor_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
#pragma acc atomic capture
  { *ptr ^= val; tmp = *ptr; }
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
