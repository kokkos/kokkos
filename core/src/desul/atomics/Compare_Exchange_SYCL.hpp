/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_SYCL_HPP_
#include "desul/atomics/Common.hpp"
#include <CL/sycl.hpp>


#ifdef DESUL_HAVE_SYCL_ATOMICS

namespace desul {

template<class MemoryOrder>
struct DesulToSYCLMemoryOrder;
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderSeqCst> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::seq_cst;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderAcquire> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::acquire;;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderRelease> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::release;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderAcqRel> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::acq_rel;
};
template<>
struct DesulToSYCLMemoryOrder<MemoryOrderRelaxed> {
  static constexpr sycl::ONEAPI::memory_order value = sycl::ONEAPI::memory_order::relaxed;
};

template<class MemoryScope>
struct DesulToSYCLMemoryScope;
template<>
struct DesulToSYCLMemoryScope<MemoryScopeCore> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::work_group;
};
template<>
struct DesulToSYCLMemoryScope<MemoryScopeDevice> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::device;
};
template<>
struct DesulToSYCLMemoryScope<MemoryScopeSystem> {
  static constexpr sycl::ONEAPI::memory_scope value = sycl::ONEAPI::memory_scope::system;
};

template<class MemoryOrder, class MemoryScope>
inline void atomic_thread_fence(MemoryOrder, MemoryScope) {
  sycl::ONEAPI::atomic_fence(DesulToSYCLMemoryOrder<MemoryOrder>::value,
                             DesulToSYCLMemoryScope<MemoryScope>::value);
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4, "this function assumes an unsigned int is 32-bit");
  sycl::ONEAPI::atomic_ref<
    unsigned int, 
    DesulToSYCLMemoryOrder<MemoryOrder>::value, 
    DesulToSYCLMemoryScope<MemoryScope>::value, 
    sycl::access::address_space::global_device_space> 
  dest_ref(*reinterpret_cast<unsigned int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned int*>(&compare), 
                                   *reinterpret_cast<unsigned int*>(&value));
  return compare;
}
template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8, "this function assumes an unsigned long long  is 64-bit");
  sycl::ONEAPI::atomic_ref<
    unsigned long long int, 
    DesulToSYCLMemoryOrder<MemoryOrder>::value,
    DesulToSYCLMemoryScope<MemoryScope>::value, 
    sycl::access::address_space::global_device_space> 
  dest_ref(*reinterpret_cast<unsigned long long int*>(dest));
  dest_ref.compare_exchange_strong(*reinterpret_cast<unsigned long long int*>(&compare),
                                   *reinterpret_cast<unsigned long long int*>(&value));
  return compare;
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_exchange(
    T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned int) == 4, "this function assumes an unsigned int is 32-bit");
  sycl::ONEAPI::atomic_ref<
    unsigned int, 
    DesulToSYCLMemoryOrder<MemoryOrder>::value, 
    DesulToSYCLMemoryScope<MemoryScope>::value,  
    sycl::access::address_space::global_device_space> 
  dest_ref(*reinterpret_cast<unsigned int*>(dest));
  unsigned int return_val = dest_ref.exchange(*reinterpret_cast<unsigned int*>(&value));
  return reinterpret_cast<T&>(return_val);
}
template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_exchange(
    T* const dest, T value, MemoryOrder, MemoryScope) {
  static_assert(sizeof(unsigned long long int) == 8, "this function assumes an unsigned long long  is 64-bit");
  sycl::ONEAPI::atomic_ref<
    unsigned long long int,
    DesulToSYCLMemoryOrder<MemoryOrder>::value,
    DesulToSYCLMemoryScope<MemoryScope>::value,
    sycl::access::address_space::global_device_space>
  dest_ref(*reinterpret_cast<unsigned long long int*>(dest));
  unsigned long long int return_val =
      dest_ref.exchange(reinterpret_cast<unsigned long long int&>(value));
  return reinterpret_cast<T&>(return_val);
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<(sizeof(T) != 8) && (sizeof(T) != 4), T>::type atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrder, MemoryScope scope) {
  // FIXME_SYCL not implemented
  assert(false);
  return compare;  
}

template <typename T, class MemoryOrder, class MemoryScope>
typename std::enable_if<(sizeof(T) != 8) && (sizeof(T) != 4), T>::type atomic_exchange(
    T* const dest, T value, MemoryOrder, MemoryScope scope) {
  assert(false);
  return value;
}

}

#endif
#endif
