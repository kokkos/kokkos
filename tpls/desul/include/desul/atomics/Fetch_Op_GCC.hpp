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
#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MEMORY_ORDER, MEMORY_SCOPE)                                 \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_##FETCH_OP  (T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_##FETCH_OP  (dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }                                                                                                                              \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_##OP_FETCH(T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_##OP_FETCH(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }

#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(FETCH_OP, OP_FETCH) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderRelaxed, MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderRelaxed, MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderRelaxed, MemoryScopeCore  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderSeqCst , MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderSeqCst , MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(FETCH_OP, OP_FETCH, MemoryOrderSeqCst , MemoryScopeCore  )
// clang-format on

DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_add,  add_fetch)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_sub,  sub_fetch)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_and,  and_fetch)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_xor,  xor_fetch)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_or,   or_fetch)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(fetch_nand, nand_fetch)

#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE

}  // namespace Impl
}  // namespace desul

#endif
