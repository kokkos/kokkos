/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FETCH_OP_GCC_HPP_
#define DESUL_ATOMICS_FETCH_OP_GCC_HPP_

#include <desul/atomics/Adapt_GCC.hpp>
#include <type_traits>

namespace desul {
namespace Impl {

// clang-format off
#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MEMORY_ORDER, MEMORY_SCOPE)                                 \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_##FETCH_OP  (T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_##FETCH_OP  (dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }                                                                                                                              \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_##OP_U##fetch(T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_##OP_U##fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }

#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(FETCH_OP) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderRelaxed, MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderRelaxed, MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderRelaxed, MemoryScopeCore  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderSeqCst , MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderSeqCst , MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, MemoryOrderSeqCst , MemoryScopeCore  )
// clang-format on

DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_add)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_sub)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_and)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_xor)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_or)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_nand)

#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE

}  // namespace Impl
}  // namespace desul

#endif
