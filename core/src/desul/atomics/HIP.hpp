/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_CUDA_HPP_
#define DESUL_ATOMICS_CUDA_HPP_

#ifdef __HIP_DEVICE_COMPILE__
namespace desul {
namespace Impl {
template <typename T>
struct is_hip_atomic_integer_type {
  static constexpr bool value = std::is_same<T, int>::value ||
                                std::is_same<T, unsigned int>::value ||
                                std::is_same<T, unsigned long long int>::value;
};

template <typename T>
struct is_hip_atomic_add_type {
  static constexpr bool value = is_hip_atomic_integer_type<T>::value ||
                                std::is_same<T, double>::value ||
                                std::is_same<T, float>::value;
};

template <typename T>
struct is_hip_atomic_sub_type {
  static constexpr bool value =
      std::is_same<T, int>::value || std::is_same<T, unsigned int>::value;
};
}  // namespace Impl

// Atomic Add
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_add_type<T>::value, T>::type
    atomic_fetch_add(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicAdd(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_add_type<T>::value, T>::type
    atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicAdd(dest, val);
  __threadfence();

  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_add_type<T>::value, T>::type
    atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_add(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic Sub
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_sub_type<T>::value, T>::type
    atomic_fetch_sub(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicSub(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_sub_type<T>::value, T>::type
    atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicSub(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_sub_type<T>::value, T>::type
    atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_sub(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic Inc
__device__ inline unsigned int atomic_fetch_inc(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrderRelaxed,
                                                MemoryScopeDevice) {
  return atomicInc(dest, val);
}

template <typename MemoryOrder>
__device__ inline unsigned int atomic_fetch_inc(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrder,
                                                MemoryScopeDevice) {
  __threadfence();
  unsigned int return_val = atomicInc(dest, val);
  __threadfence();
  return return_val;
}

template <typename MemoryOrder>
__device__ inline unsigned int atomic_fetch_inc(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrder,
                                                MemoryScopeCore) {
  return atomic_fetch_inc(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic Dec
__device__ inline unsigned int atomic_fetch_dec(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrderRelaxed,
                                                MemoryScopeDevice) {
  return atomicDec(dest, val);
}

template <typename MemoryOrder>
__device__ inline unsigned int atomic_fetch_dec(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrder,
                                                MemoryScopeDevice) {
  __threadfence();
  unsigned int return_val = atomicDec(dest, val);
  __threadfence();
  return return_val;
}

template <typename MemoryOrder>
__device__ inline unsigned int atomic_fetch_dec(unsigned int* dest,
                                                unsigned int val,
                                                MemoryOrder,
                                                MemoryScopeCore) {
  return atomic_fetch_dec(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic Max
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_max(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicMax(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicMax(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_max(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic Min
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_min(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicMin(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicMin(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_min(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic And
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_and(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicAnd(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicAnd(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_and(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic XOR
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_xor(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicXor(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_typename_atomic_integer_type<T>::value, T>::type
    atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicXor(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_xor(dest, val, MemoryOrder(), MemoryScopeDevice());
}

// Atomic OR
template <typename T>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_or(T* dest, T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  return atomicOr(dest, val);
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  __threadfence();
  T return_val = atomicOr(dest, val);
  __threadfence();
  return return_val;
}

template <typename T, typename MemoryOrder>
__device__ inline
    typename std::enable_if<Impl::is_hip_atomic_integer_type<T>::value, T>::type
    atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeCore) {
  return atomic_fetch_or(dest, val, MemoryOrder(), MemoryScopeDevice());
}
}  // namespace desul

#endif
#endif
