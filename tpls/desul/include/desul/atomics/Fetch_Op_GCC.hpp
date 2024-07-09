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
#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MEMORY_ORDER, MEMORY_SCOPE)                                 \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic_fetch##U_OP  (T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic_fetch##U_OP  (dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }                                                                                                                              \
  template <class T>                                                                                                          \
  std::enable_if_t<std::is_integral<T>::value, T> host_atomic##U_OP##_fetch(T* const dest, T value, MEMORY_ORDER, MEMORY_SCOPE) { \
    return __atomic##U_OP##_fetch(dest, value, GCCMemoryOrder<MEMORY_ORDER>::value);                                              \
  }

#define DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(U_OP) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderRelaxed, MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderRelaxed, MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderRelaxed, MemoryScopeCore  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderSeqCst , MemoryScopeNode  ) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderSeqCst , MemoryScopeDevice) \
   DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE(U_OP, MemoryOrderSeqCst , MemoryScopeCore  )
// clang-format on

DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_add)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_sub)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_and)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_xor)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_or)
DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL(_nand)

#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL
#undef DESUL_IMPL_GCC_HOST_ATOMIC_FETCH_OP_INTEGRAL_ORDER_SCOPE

}  // namespace Impl
}  // namespace desul

#endif
