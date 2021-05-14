/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_SYCL_HPP_
#define DESUL_ATOMICS_SYCL_HPP_

#ifdef DESUL_HAVE_SYCL_ATOMICS

namespace desul {
namespace Impl {
template<class T>
struct is_sycl_atomic_type {
  static constexpr bool value = std::is_same<T,int>::value ||
                                std::is_same<T,unsigned int>::value ||
				std::is_same<T,long>::value ||
				std::is_same<T,unsigned long>::value ||
				std::is_same<T,long long>::value ||
                                std::is_same<T,unsigned long long int>::value ||
				std::is_same<T, float>::value ||
				std::is_same<T, double>::value;
};
} // Impl

// Atomic Add
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_add(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
	 sycl::ONEAPI::atomic_ref<T, sycl::ONEAPI::memory_order::relaxed, sycl::ONEAPI::memory_scope::device,  sycl::access::address_space::global_device_space> dest_ref(*dest);
  dest_ref.fetch_add(val);
  return *dest;
}

// Atomic Sub 
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_sub(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic Inc
template<class MemoryOrder>
inline
unsigned int atomic_fetch_inc(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic Dec
template<class MemoryOrder>
inline
unsigned int atomic_fetch_dec(unsigned int* dest, unsigned int val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic Max
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_max(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic Min
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_min(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic And
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_and(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic XOR
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_xor(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

// Atomic OR
template<class T, class MemoryOrder>
inline
typename std::enable_if<Impl::is_sycl_atomic_type<T>::value,T>::type
atomic_fetch_or(T* dest, T val, MemoryOrder, MemoryScopeDevice) {
  return *dest;
}

} // desul
#endif  // DESUL_HAVE_SYCL_ATOMICS
#endif  // DESUL_ATOMICS_SYCL_HPP_
