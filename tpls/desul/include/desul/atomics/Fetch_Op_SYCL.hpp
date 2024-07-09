/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_FETCH_OP_SYCL_HPP_
#define DESUL_ATOMICS_FETCH_OP_SYCL_HPP_

#include <desul/atomics/Adapt_SYCL.hpp>
#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

// clang-format off
#define DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, TYPE)                                     \
  template <class MemoryOrder>                                                            \
  TYPE device_atomic_fetch##U_OPER(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);                \
    return dest_ref.fetch##U_OPER(val);                                                    \
  }                                                                                       \
  template <class MemoryOrder>                                                            \
  TYPE device_atomic_fetch##U_OPER(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeCore  ) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);                  \
    return dest_ref.fetch##U_OPER(val);                                                    \
  }
// clang-format on

#define DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(U_OPER) \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, int)           \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, unsigned int)  \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, long)          \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, unsigned long) \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, long long)     \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, unsigned long long)

#define DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT(U_OPER) \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, float)               \
  DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER(U_OPER, double)

DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_add)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_sub)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_and)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_or)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_xor)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_min)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL(_max)

DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT(_add)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT(_sub)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT(_min)
DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT(_max)

#undef DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_FLOATING_POINT
#undef DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER_INTEGRAL
#undef DESUL_IMPL_SYCL_ATOMIC_FETCH_OPER

}  // namespace Impl
}  // namespace desul

#endif
