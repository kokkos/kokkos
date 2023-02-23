/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_OPENACC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_OPENACC_HPP_

#include <openacc.h>

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Thread_Fence_OpenACC.hpp>

namespace desul {
namespace Impl {

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  if (acc_on_device(acc_device_not_host)) {
#pragma acc atomic capture
    {
      return_val = *dest;
      *dest = value;
    }
  } else {
      return_val = *dest;
      *dest = value;
  }
  return return_val;
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!std::is_arithmetic<T>::value, T> device_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  return_val = *dest;
  *dest = value;
  return return_val;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
int device_atomic_exchange(int* dest, int value, MemoryOrder, MemoryScope) {
  return atomicExch(dest, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
unsigned int device_atomic_exchange(unsigned int* dest, unsigned int value, MemoryOrder, MemoryScope) {
  return atomicExch(dest, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
long long int device_atomic_exchange(long long int* dest, long long int value, MemoryOrder, MemoryScope) {
  return atomicExch(dest, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
unsigned long long int device_atomic_exchange(unsigned long long int* dest, unsigned long long int value, MemoryOrder, MemoryScope) {
  return atomicExch(dest, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
float device_atomic_exchange(float * dest, float value, MemoryOrder, MemoryScope) {
  return atomicExch(dest, value);
}
#endif

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
T device_atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope) {
  T current_val = *dest;
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_compare_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  } else {
    if (current_val == compare) *dest = value;
  }
  return current_val;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
int device_atomic_compare_exchange(int* dest, int compare, int value, MemoryOrder, MemoryScope) {
  return  atomicCAS(dest, compare, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
unsigned int device_atomic_compare_exchange(unsigned int* dest, unsigned int compare, unsigned int value, MemoryOrder, MemoryScope) {
  return  atomicCAS(dest, compare, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
long long int device_atomic_compare_exchange(long long int* dest, long long int compare, long long int value, MemoryOrder, MemoryScope) {
  return  atomicCAS(dest, compare, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
unsigned long long int device_atomic_compare_exchange(unsigned long long int* dest, unsigned long long int compare, unsigned long long int value, MemoryOrder, MemoryScope) {
  return  atomicCAS(dest, compare, value);
}

#pragma acc routine seq
template <class MemoryOrder, class MemoryScope>
float device_atomic_compare_exchange(float* dest, float compare, float value, MemoryOrder, MemoryScope) {
  return  atomicCAS(dest, compare, value);
}
#endif

}  // namespace Impl
}  // namespace desul

#endif
