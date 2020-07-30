/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_CUDA_HPP_
#define DESUL_ATOMICS_CUDA_HPP_

#ifdef DESUL_HAVE_CUDA_ATOMICS
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=700)) || !defined(__NVCC__)
#include <desul/atomics/cuda/CUDA_asm.hpp>
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__<700)
namespace desul {
namespace Impl {
template<class T>
struct is_cuda_atomic_integer_type {
  static constexpr bool value = std::is_same<T,int>::value ||
                                std::is_same<T,unsigned int>::value ||
                                std::is_same<T,unsigned long long int>::value;
};

template<class T>
struct is_cuda_atomic_add_type {
  static constexpr bool value = is_cuda_atomic_integer_type<T>::value ||
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
                                std::is_same<T,double>::value || 
#endif
                                std::is_same<T,float>::value;
};

template<class T>
struct is_cuda_atomic_sub_type {
  static constexpr bool value = std::is_same<T,int>::value ||
                                std::is_same<T,unsigned int>::value;
};
} // Impl

// Atomic Add
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_add_type<T>::value,T>::type
atomic_fetch_add(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicAdd(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_add_type<T>::value,T>::type
atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicAdd(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_add_type<T>::value,T>::type
atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_add(dest,val,MemoryOrder(),MemoryScopeDevice());
}


// Atomic Sub 
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_sub_type<T>::value,T>::type
atomic_fetch_sub(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicSub(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_sub_type<T>::value,T>::type
atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicSub(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_sub_type<T>::value,T>::type
atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_sub(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic Inc
__device__ inline
unsigned int atomic_fetch_inc(unsigned int* dest, unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicInc(dest,val);
}

template<class MemoryOrder>
__device__ inline
unsigned int atomic_fetch_inc(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  unsigned int return_val = atomicInc(dest,val);
  __threadfence();
  return return_val;
}

template<class MemoryOrder>
__device__ inline
unsigned int atomic_fetch_inc(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_inc(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic Inc
__device__ inline
unsigned int atomic_fetch_dec(unsigned int* dest, unsigned int val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicDec(dest,val);
}

template<class MemoryOrder>
__device__ inline
unsigned int atomic_fetch_dec(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  unsigned int return_val = atomicDec(dest,val);
  __threadfence();
  return return_val;
}

template<class MemoryOrder>
__device__ inline
unsigned int atomic_fetch_dec(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_dec(dest,val,MemoryOrder(),MemoryScopeDevice());
}


// Atomic Max
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_max(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicMax(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicMax(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_max(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic Min
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_min(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicMin(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicMin(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_min(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic And
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_and(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicAnd(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicAnd(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_and(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic XOR
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_xor(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicXor(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicXor(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_xor(dest,val,MemoryOrder(),MemoryScopeDevice());
}

// Atomic OR
template<class T>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_or(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicOr(dest,val);
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicOr(dest,val);
  __threadfence();
  return return_val;
}

template<class T, class MemoryOrder>
__device__ inline
typename std::enable_if<Impl::is_cuda_atomic_integer_type<T>::value,T>::type
atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_or(dest,val,MemoryOrder(),MemoryScopeDevice());
}
} // desul
#endif

#endif  // DESUL_HAVE_CUDA_ATOMICS
#endif
