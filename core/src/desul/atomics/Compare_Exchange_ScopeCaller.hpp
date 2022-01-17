/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_SCOPECALLER_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_SCOPECALLER_HPP_
#include "desul/atomics/Common.hpp"

namespace desul {

template <class MemoryOrder>
void atomic_thread_fence(MemoryOrder, MemoryScopeCaller) {}

template <typename T, class MemoryOrder>
T atomic_exchange(T* dest,
                  Impl::dont_deduce_this_parameter_t<const T> value,
                  MemoryOrder,
                  MemoryScopeCaller) {
  T return_val = *dest;
  *dest = value;
  return return_val;
}

template <typename T, class MemoryOrder>
T atomic_compare_exchange(T* dest,
                          Impl::dont_deduce_this_parameter_t<const T> compare,
                          Impl::dont_deduce_this_parameter_t<const T> value,
                          MemoryOrder,
                          MemoryScopeCaller) {
  T current_val = *dest;
  if (current_val == compare) *dest = value;
  return current_val;
}

}  // namespace desul
#endif
