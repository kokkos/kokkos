/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_
#define DESUL_ATOMICS_FETCH_OP_OPENACC_HPP_

#include <algorithm>
#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

#ifdef __NVCOMPILER
template <class T, class R>
using acc_enable_if_supported_arithmetic_type =
    std::enable_if_t<(std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                      std::is_same_v<T, unsigned long long> || std::is_same_v<T, float>
#ifndef DESUL_CUDA_ARCH_IS_PRE_PASCAL
                      || std::is_same_v<T, double>
#endif
                      ),
                     R>;
template <class T, class R>
using acc_enable_if_supported_integral_type =
    std::enable_if_t<(std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                      std::is_same_v<T, unsigned long long>),
                     R>;
#else
template <class T, class R>
using acc_enable_if_supported_arithmetic_type =
    std::enable_if_t<std::is_arithmetic_v<T>, R>;
template <class T, class R>
using acc_enable_if_supported_integral_type =
    std::enable_if_t<std::is_integral_v<T>, R>;
#endif

//<editor-fold
// desc="device_atomic_fetch_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}">
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_add(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr += val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_inc(
    T* ptr, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr += T(1);
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_sub(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr -= val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_dec(
    T* ptr, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr -= T(1);
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_mul(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr *= val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_div(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr /= val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_fetch_lshift(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr = *ptr << val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_fetch_rshift(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr = *ptr >> val;
  }
  return old;
}

#ifdef __NVCOMPILER
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_max(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T old;
  old = atomicMax(ptr, val);
  return old;
}
#endif

#ifdef __NVCOMPILER
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_fetch_min(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  int old;
  old = atomicMin(ptr, val);
  return old;
}
#endif

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_fetch_and(T* ptr,
                                                                    const T val,
                                                                    MemoryOrderRelaxed,
                                                                    MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr &= val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_fetch_or(T* ptr,
                                                                   const T val,
                                                                   MemoryOrderRelaxed,
                                                                   MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr |= val;
  }
  return old;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_fetch_xor(T* ptr,
                                                                    const T val,
                                                                    MemoryOrderRelaxed,
                                                                    MemoryScope) {
  T old;
#pragma acc atomic capture
  {
    old = *ptr;
    *ptr ^= val;
  }
  return old;
}
//</editor-fold>

//<editor-fold
// desc="device_atomic_{add,sub,mul,div,lshift,rshift,mod,max,min,and,or,xor}_fetch">
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_add_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr += val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_inc_fetch(
    T* ptr, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr += T(1);
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_sub_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr -= val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_dec_fetch(
    T* ptr, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr -= T(1);
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_mul_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr *= val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_div_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr /= val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_lshift_fetch(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr = *ptr << val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_rshift_fetch(
    T* ptr, const unsigned int val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr = *ptr >> val;
    tmp = *ptr;
  }
  return tmp;
}

#ifdef __NVCOMPILER
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_max_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
  tmp = atomicMax(ptr, val);
  tmp = std::max(tmp, val);
  return tmp;
}
#endif

#ifdef __NVCOMPILER
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_min_fetch(
    T* ptr, const T val, MemoryOrderRelaxed, MemoryScope) {
  T tmp;
  tmp = atomicMin(ptr, val);
  tmp = std::min(tmp, val);
  return tmp;
}
#endif

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_and_fetch(T* ptr,
                                                                    const T val,
                                                                    MemoryOrderRelaxed,
                                                                    MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr &= val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_or_fetch(T* ptr,
                                                                   const T val,
                                                                   MemoryOrderRelaxed,
                                                                   MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr |= val;
    tmp = *ptr;
  }
  return tmp;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_integral_type<T, T> device_atomic_xor_fetch(T* ptr,
                                                                    const T val,
                                                                    MemoryOrderRelaxed,
                                                                    MemoryScope) {
  T tmp;
#pragma acc atomic capture
  {
    *ptr ^= val;
    tmp = *ptr;
  }
  return tmp;
}
//</editor-fold>

//<editor-fold desc="device_atomic_{store,load}">
#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, void> device_atomic_store(T* const ptr,
                                                                     const T val,
                                                                     MemoryOrderRelaxed,
                                                                     MemoryScope) {
#pragma acc atomic write
  *ptr = val;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, void> device_atomic_store(T* const ptr,
                                                                     const T val,
                                                                     MemoryOrderRelease,
                                                                     MemoryScope) {
  if (acc_on_device(acc_device_not_host)) {
    printf(
        "DESUL error in device_atomic_exchange(): Not supported atomic "
        "operation in the OpenACC backend\n");
  }
#pragma acc atomic write
  *ptr = val;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_load(const T* const ptr,
                                                                 MemoryOrderRelaxed,
                                                                 MemoryScope) {
  T retval;
#pragma acc atomic read
  retval = *ptr;
  return retval;
}

#pragma acc routine seq
template <class T, class MemoryScope>
acc_enable_if_supported_arithmetic_type<T, T> device_atomic_load(const T* const ptr,
                                                                 MemoryOrderAcquire,
                                                                 MemoryScope) {
  T retval;
  if (acc_on_device(acc_device_not_host)) {
    printf(
        "DESUL error in device_atomic_exchange(): Not supported atomic "
        "operation in the OpenACC backend\n");
  }
#pragma acc atomic read
  retval = *ptr;
  return retval;
}
//</editor-fold>

}  // namespace Impl
}  // namespace desul

#endif
