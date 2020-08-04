/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_MSVC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_MSVC_HPP_
#include "desul/atomics/Common.hpp"
#include <type_traits>
#ifdef DESUL_HAVE_MSVC_ATOMICS

#ifndef DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#define DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#endif
#include <windows.h>

namespace desul {
template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 1, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  CHAR return_val =
      _InterlockedCompareExchange8((CHAR*)dest, *((CHAR*)&val), *((CHAR*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 2, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  SHORT return_val =
      _InterlockedCompareExchange16((SHORT*)dest, *((SHORT*)&val), *((SHORT*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  LONG return_val =
      _InterlockedCompareExchange((LONG*)dest, *((LONG*)&val), *((LONG*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  LONG64 return_val = _InterlockedCompareExchange64(
      (LONG64*)dest, *((LONG64*)&val), *((LONG64*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 16, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderRelaxed, MemoryScope) {
  Dummy16ByteValue* val16 = reinterpret_cast<Dummy16ByteValue*>(&val);
  (void)_InterlockedCompareExchange128(reinterpret_cast<LONG64*>(dest),
                                       val16->value2,
                                       val16->value1,
                                       (reinterpret_cast<LONG64*>(&compare)));
  return compare;
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 1, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  CHAR return_val =
      _InterlockedCompareExchange8((CHAR*)dest, *((CHAR*)&val), *((CHAR*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 2, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  SHORT return_val =
      _InterlockedCompareExchange16((SHORT*)dest, *((SHORT*)&val), *((SHORT*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 4, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  LONG return_val =
      _InterlockedCompareExchange((LONG*)dest, *((LONG*)&val), *((LONG*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 8, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  LONG64 return_val = _InterlockedCompareExchange64(
      (LONG64*)dest, *((LONG64*)&val), *((LONG64*)&compare));
  return *(reinterpret_cast<T*>(&return_val));
}

template <typename T, class MemoryScope>
typename std::enable_if<sizeof(T) == 16, T>::type atomic_compare_exchange(
    T* const dest, T compare, T val, MemoryOrderSeqCst, MemoryScope) {
  Dummy16ByteValue* val16 = reinterpret_cast<Dummy16ByteValue*>(&val);
  (void)_InterlockedCompareExchange128(reinterpret_cast<LONG64*>(dest),
                                       val16->value2,
                                       val16->value1,
                                       (reinterpret_cast<LONG64*>(&compare)));
  return compare;
}

}  // namespace desul
#endif
#endif
