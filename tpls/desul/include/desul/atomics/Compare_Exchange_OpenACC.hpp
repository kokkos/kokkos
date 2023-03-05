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
#include <impl/Kokkos_Error.hpp>

namespace desul {
namespace Impl {

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!std::is_arithmetic<T>::value, T> device_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope scope) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  //FIXME_OPENACC OpenACC lock APIs are not implemented.
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  T return_val = *dest;
  *dest = value;
  //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  //unlock_address_openacc((void*)dest, scope);
  return return_val;
}

#ifdef KOKKOS_COMPILER_NVHPC

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_arithmetic<T>::value && (sizeof(T) != 4) && (sizeof(T) != 8), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope scope) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  T return_val = *dest;
  *dest = value;
  //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  //unlock_address_openacc((void*)dest, scope);
  return return_val;
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicExch(reinterpret_cast<unsigned int*>(dest),
                                       reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 8), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long int is 64-bit");
  unsigned long long int return_val = atomicExch(reinterpret_cast<unsigned long long int*>(dest),
                                       reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 4), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(float) == 4,
                "this function assumes an float is 32-bit");
  float return_val = atomicExch(reinterpret_cast<float*>(dest),
                                       reinterpret_cast<float&>(value));
  return reinterpret_cast<T&>(return_val);
}

#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 8), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(double) == 8,
                "this function assumes an double is 64-bit");
  double return_val = atomicExch(reinterpret_cast<double*>(dest),
                                       reinterpret_cast<double&>(value));
  return reinterpret_cast<T&>(return_val);
}

#else

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 8), T> 
	device_atomic_exchange(T* const dest, T value, MemoryOrder, MemoryScope scope) {
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  T return_val = *dest;
  *dest = value;
  //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  //unlock_address_openacc((void*)dest, scope);
  return return_val;
}

#endif

#else

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_exchange(T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
#pragma acc atomic capture
  {
    return_val = *dest;
    *dest = value;
  }
  return return_val;
}

#endif


#ifdef KOKKOS_COMPILER_NVHPC

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<!std::is_arithmetic<T>::value || ((sizeof(T) != 4) && (sizeof(T) != 8)), T> 
	device_atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  T current_val = *dest;
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_compare_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  if (current_val == compare) {
    *dest = value;
    //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  }
  //unlock_address_openacc((void*)dest, scope);
  return current_val;
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 4), T> device_atomic_compare_exchange(T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4,
                "this function assumes an unsigned int is 32-bit");
  unsigned int return_val = atomicCAS(reinterpret_cast<unsigned int*>(dest),
                                      reinterpret_cast<unsigned int&>(compare),
                                      reinterpret_cast<unsigned int&>(value));
  return reinterpret_cast<T&>(return_val);
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_integral<T>::value && (sizeof(T) == 8), T> device_atomic_compare_exchange(T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8,
                "this function assumes an unsigned long long int is 64-bit");
  unsigned long long int return_val = atomicCAS(reinterpret_cast<unsigned long long int*>(dest),
                                      reinterpret_cast<unsigned long long int&>(compare),
                                      reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 4), T> device_atomic_compare_exchange(T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(float) == 4,
                "this function assumes an float is 32-bit");
  float return_val = atomicCAS(reinterpret_cast<float*>(dest),
                                      reinterpret_cast<float&>(compare),
                                      reinterpret_cast<float&>(value));
  return reinterpret_cast<T&>(return_val);
}

#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 8), T> device_atomic_compare_exchange(T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(double) == 8,
                "this function assumes an double is 64-bit");
  double return_val = atomicCAS(reinterpret_cast<double*>(dest),
                                      reinterpret_cast<double&>(compare),
                                      reinterpret_cast<double&>(value));
  return reinterpret_cast<T&>(return_val);
}

#else

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
std::enable_if_t<std::is_floating_point<T>::value && (sizeof(T) == 8), T> device_atomic_compare_exchange(T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  T current_val = *dest;
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_compare_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  if (current_val == compare) {
    *dest = value;
    //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  }
  //unlock_address_openacc((void*)dest, scope);
  return current_val;
}

#endif

#else

#pragma acc routine seq
template <class T, class MemoryOrder, class MemoryScope>
T device_atomic_compare_exchange(T* dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  T current_val = *dest;
  if (acc_on_device(acc_device_not_host)) {
    Kokkos::abort("Kokkos Error in device_atomic_compare_exchange(): Not supported atomic "
                  "operation in the OpenACC backend");
  }
  // Acquire a lock for the address
  //while (!lock_address_openacc((void*)dest, scope)) {
  //}
  //device_atomic_thread_fence(MemoryOrderAcquire(), scope);
  if (current_val == compare) {
    *dest = value;
    //device_atomic_thread_fence(MemoryOrderRelease(), scope);
  }
  //unlock_address_openacc((void*)dest, scope);
  return current_val;
}

#endif

}  // namespace Impl
}  // namespace desul

#endif
