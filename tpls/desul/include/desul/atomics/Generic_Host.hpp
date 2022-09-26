/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_GENERIC_HOST_HPP_
#define DESUL_ATOMICS_GENERIC_HOST_HPP_

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Compare_Exchange.hpp>
#include <desul/atomics/Lock_Based_Fetch_Op.hpp>
#include <desul/atomics/Operator_Function_Objects.hpp>
#include <type_traits>

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace desul {
namespace Impl {

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires host_atomic_always_lock_free(sizeof(T))
          std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>
inline T host_atomic_fetch_oper(const Oper& op,
                                T* const dest,
                                dont_deduce_this_parameter_t<const T> val,
                                MemoryOrder order,
                                MemoryScope scope) {
  using cas_t = atomic_compare_exchange_t<T>;
  cas_t oldval = reinterpret_cast<cas_t&>(*dest);
  cas_t assume = oldval;

  do {
    if (check_early_exit(op, reinterpret_cast<T&>(oldval), val))
      return reinterpret_cast<T&>(oldval);
    assume = oldval;
    T newval = op.apply(reinterpret_cast<T&>(assume), val);
    oldval = host_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),
                                          assume,
                                          reinterpret_cast<cas_t&>(newval),
                                          order,
                                          scope);
  } while (assume != oldval);

  return reinterpret_cast<T&>(oldval);
}

template <class Oper,
          class T,
          class MemoryOrder,
          class MemoryScope,
          // equivalent to:
          //   requires host_atomic_always_lock_free(sizeof(T))
          std::enable_if_t<atomic_always_lock_free(sizeof(T)), int> = 0>
inline T host_atomic_oper_fetch(const Oper& op,
                                T* const dest,
                                dont_deduce_this_parameter_t<const T> val,
                                MemoryOrder order,
                                MemoryScope scope) {
  using cas_t = atomic_compare_exchange_t<T>;
  cas_t oldval = reinterpret_cast<cas_t&>(*dest);
  T newval = val;
  cas_t assume = oldval;
  do {
    if (check_early_exit(op, reinterpret_cast<T&>(oldval), val))
      return reinterpret_cast<T&>(oldval);
    assume = oldval;
    newval = op.apply(reinterpret_cast<T&>(assume), val);
    oldval = host_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),
                                          assume,
                                          reinterpret_cast<cas_t&>(newval),
                                          order,
                                          scope);
  } while (assume != oldval);

  return newval;
}

template <class Oper, class T, class MemoryOrder>
inline T host_atomic_fetch_oper(const Oper& op,
                                T* const dest,
                                dont_deduce_this_parameter_t<const T> val,
                                MemoryOrder /*order*/,
                                MemoryScopeCaller /*scope*/) {
  T oldval = *dest;
  *dest = op.apply(oldval, val);
  return oldval;
}

template <class Oper, class T, class MemoryOrder>
inline T host_atomic_oper_fetch(const Oper& op,
                                T* const dest,
                                dont_deduce_this_parameter_t<const T> val,
                                MemoryOrder /*order*/,
                                MemoryScopeCaller /*scope*/) {
  T oldval = *dest;
  T newval = op.apply(oldval, val);
  *dest = newval;
  return newval;
}

// Fetch_Oper atomics: return value before operation
template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_add(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(AddOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_sub(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(SubOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_max(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(MaxOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_min(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(MinOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_mul(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(MulOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_div(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(DivOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_mod(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(ModOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_and(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(AndOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_or(T* const dest,
                              const T val,
                              MemoryOrder order,
                              MemoryScope scope) {
  return host_atomic_fetch_oper(OrOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_xor(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_fetch_oper(XorOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_nand(T* const dest,
                                const T val,
                                MemoryOrder order,
                                MemoryScope scope) {
  return host_atomic_fetch_oper(NandOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_lshift(T* const dest,
                                  const unsigned int val,
                                  MemoryOrder order,
                                  MemoryScope scope) {
  return host_atomic_fetch_oper(
      LShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_rshift(T* const dest,
                                  const unsigned int val,
                                  MemoryOrder order,
                                  MemoryScope scope) {
  return host_atomic_fetch_oper(
      RShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

// Oper Fetch atomics: return value after operation
template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_add_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(AddOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_sub_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(SubOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_max_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(MaxOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_min_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(MinOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_mul_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(MulOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_div_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(DivOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_mod_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(ModOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_and_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(AndOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_or_fetch(T* const dest,
                              const T val,
                              MemoryOrder order,
                              MemoryScope scope) {
  return host_atomic_oper_fetch(OrOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_xor_fetch(T* const dest,
                               const T val,
                               MemoryOrder order,
                               MemoryScope scope) {
  return host_atomic_oper_fetch(XorOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_nand_fetch(T* const dest,
                                const T val,
                                MemoryOrder order,
                                MemoryScope scope) {
  return host_atomic_oper_fetch(NandOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_lshift_fetch(T* const dest,
                                  const unsigned int val,
                                  MemoryOrder order,
                                  MemoryScope scope) {
  return host_atomic_oper_fetch(
      LShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_rshift_fetch(T* const dest,
                                  const unsigned int val,
                                  MemoryOrder order,
                                  MemoryScope scope) {
  return host_atomic_oper_fetch(
      RShiftOper<T, const unsigned int>(), dest, val, order, scope);
}

// Other atomics

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_load(const T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_fetch_oper(
      LoadOper<T, const T>(), const_cast<T*>(dest), T(), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_store(T* const dest,
                              const T val,
                              MemoryOrder order,
                              MemoryScope scope) {
  (void)host_atomic_fetch_oper(StoreOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_add(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_add(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_sub(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_sub(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_mul(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_mul(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_div(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_div(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_min(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_min(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_max(T* const dest,
                            const T val,
                            MemoryOrder order,
                            MemoryScope scope) {
  (void)host_atomic_fetch_max(dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_inc_fetch(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_add_fetch(dest, T(1), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_dec_fetch(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_sub_fetch(dest, T(1), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_inc(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_fetch_add(dest, T(1), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_inc_mod(T* const dest,
                                   T val,
                                   MemoryOrder order,
                                   MemoryScope scope) {
  static_assert(std::is_unsigned<T>::value,
                "Signed types not supported by host_atomic_fetch_inc_mod.");
  return host_atomic_fetch_oper(IncModOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_dec(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_fetch_sub(dest, T(1), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline T host_atomic_fetch_dec_mod(T* const dest,
                                   T val,
                                   MemoryOrder order,
                                   MemoryScope scope) {
  static_assert(std::is_unsigned<T>::value,
                "Signed types not supported by host_atomic_fetch_dec_mod.");
  return host_atomic_fetch_oper(DecModOper<T, const T>(), dest, val, order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_inc(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_add(dest, T(1), order, scope);
}

template <class T, class MemoryOrder, class MemoryScope>
inline void host_atomic_dec(T* const dest, MemoryOrder order, MemoryScope scope) {
  return host_atomic_sub(dest, T(1), order, scope);
}

// FIXME
template <class T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
inline bool host_atomic_compare_exchange_strong(T* const dest,
                                                T& expected,
                                                T desired,
                                                SuccessMemoryOrder success,
                                                FailureMemoryOrder /*failure*/,
                                                MemoryScope scope) {
  T const old = host_atomic_compare_exchange(dest, expected, desired, success, scope);
  if (old != expected) {
    expected = old;
    return false;
  } else {
    return true;
  }
}

template <class T,
          class SuccessMemoryOrder,
          class FailureMemoryOrder,
          class MemoryScope>
inline bool host_atomic_compare_exchange_weak(T* const dest,
                                              T& expected,
                                              T desired,
                                              SuccessMemoryOrder success,
                                              FailureMemoryOrder failure,
                                              MemoryScope scope) {
  return host_atomic_compare_exchange_strong(
      dest, expected, desired, success, failure, scope);
}

}  // namespace Impl
}  // namespace desul

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif
