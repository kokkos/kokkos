/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_SERIAL_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_SERIAL_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

template <class T, class MemoryScope>
T host_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderRelaxed, MemoryScope) {
  T old = *dest;
  if (old == compare) {
    *dest = value;
  } else {
    old = compare;
  }
  return compare;
}
template <class T, class MemoryScope>
T host_atomic_compare_exchange(
    T* const dest, T compare, T value, MemoryOrderSeqCst, MemoryScope) {
  T old = *dest;
  if (old == compare) {
    *dest = value;
  } else {
    old = compare;
  }
  return compare;
}

}  // namespace Impl
}  // namespace desul

#endif
