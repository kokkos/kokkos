/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_
#define DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

// clang-format off
//<editor-fold desc="device_atomic_fetch_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}">
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_add(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = lptr[0]; lptr[0] += val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_sub(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr -= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_mul(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr *= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_div(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr /= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_lshift(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr = *lptr << val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_rshift(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr = *lptr >> val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_mod(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_fetch_mod(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { tmp = *lptr; *lptr = *lptr % val; }
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_max(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  tmp = atomicMax(const_cast<T *>(ptr), val);
  return tmp;
}
#else
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_max(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_fetch_max(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { tmp = *lptr; *lptr = std::max(*lptr, val); }
  return tmp;
}
#endif

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_min(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  tmp = atomicMin(const_cast<T *>(ptr), val);
  return tmp;
}
#else
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_min(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_fetch_min(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { tmp = *lptr; *lptr = std::min(*lptr, val); }
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_and(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr &= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_or(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { tmp = *lptr; *lptr |= val; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_fetch_xor(
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
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_add_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr += val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_sub_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr -= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_mul_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr *= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_div_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr /= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_lshift_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr = *lptr << val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_rshift_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr = *lptr >> val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_mod_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_mod_fetch(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { *lptr = *lptr % val; tmp = *lptr; }
  return tmp;
}

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_max_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  tmp = atomicMax(const_cast<T *>(ptr), val);
  tmp = std::max(tmp, val);
  return tmp;
}
#else
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_max_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_max_fetch(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { *lptr = std::max(*lptr, val); tmp = *lptr; }
  return tmp;
}
#endif

#ifdef KOKKOS_COMPILER_NVHPC
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_min_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  tmp = atomicMin(const_cast<T *>(ptr), val);
  tmp = std::min(tmp, val);
  return tmp;
}
#else
#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_min_fetch(
T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  static_assert(!Kokkos::Impl::always_true<T>::value,
                "Kokkos Error in device_atomic_min_fetch(): Not supported atomic "
                "operation in the OpenACC backend");
  T tmp;
  T *lptr = const_cast<T *>(ptr);
  { *lptr = std::min(*lptr, val); tmp = *lptr; }
  return tmp;
}
#endif

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_and_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr &= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_or_fetch(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  T tmp;
  T *lptr = const_cast<T *>(ptr);
#pragma acc atomic capture
  { *lptr |= val; tmp = *lptr; }
  return tmp;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_xor_fetch(
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
std::enable_if_t<std::is_arithmetic<T>::value, void> device_atomic_store(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
#pragma acc atomic write
  *ptr = val;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<!std::is_arithmetic<T>::value, void> device_atomic_store(
    T* const ptr, const T val, MemoryOrderRelaxed, MemoryScopeDevice) {
  *ptr = val;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<std::is_arithmetic<T>::value, T> device_atomic_load(
    T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T retval{};
#pragma acc atomic read
  retval = *ptr;
  return retval;
}

#pragma acc routine seq
template <class T>
std::enable_if_t<!std::is_arithmetic<T>::value, T> device_atomic_load(
    T* const ptr, MemoryOrderRelaxed, MemoryScopeDevice) {
  T retval{};
  retval = *ptr;
  return retval;
}
//</editor-fold>
// clang-format on

}  // namespace Impl
}  // namespace desul

#endif
