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
#include <desul/atomics/Operator_Function_Objects.hpp>

namespace desul {
namespace Impl {

// clang-format off
#define DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, TYPE)                                             \
  template <class MemoryOrder>                                                              \
  TYPE device_atomic_fetch_##OPER(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeDevice) {   \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);                  \
    return dest_ref.fetch_##OPER(val);                                                      \
  }                                                                                         \
  template <class MemoryOrder>                                                              \
  TYPE device_atomic_fetch_##OPER(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeCore  ) {   \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);                    \
    return dest_ref.fetch_##OPER(val);                                                      \
  }                                                                                         \
  template <class MemoryOrder>                                                              \
  TYPE device_atomic_##OPER##_fetch(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);                  \
    return OPER##_fetch_operator<TYPE, TYPE>::apply(dest_ref.fetch_##OPER(val), val);       \
  }                                                                                         \
  template <class MemoryOrder>                                                              \
  TYPE device_atomic_##OPER##_fetch(TYPE* dest, TYPE val, MemoryOrder, MemoryScopeCore)   { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);                    \
    return OPER##_fetch_operator<TYPE, TYPE>::apply(dest_ref.fetch_##OPER(val), val);       \
  }
// clang-format on

#define DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(OPER) \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, int)           \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, unsigned int)  \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, long)          \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, unsigned long) \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, long long)     \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, unsigned long long)

#define DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT(OPER) \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, float)               \
  DESUL_IMPL_SYCL_ATOMIC_OPER(OPER, double)

DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(add)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(sub)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(and)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(or)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(xor)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(min)
DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL(max)

DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT(add)
DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT(sub)
DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT(min)
DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT(max)

// clang-format off
#define DESUL_IMPL_SYCL_ATOMIC_INC(TYPE)                                     \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_fetch_inc(TYPE* dest, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);   \
    return dest_ref++;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_fetch_inc(TYPE* dest, MemoryOrder, MemoryScopeCore  ) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);     \
    return dest_ref++;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_inc_fetch(TYPE* dest, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);   \
    return ++dest_ref;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_inc_fetch(TYPE* dest, MemoryOrder, MemoryScopeCore)   { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);     \
    return ++dest_ref;                                                       \
  }

#define DESUL_IMPL_SYCL_ATOMIC_DEC(TYPE)                                     \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_fetch_dec(TYPE* dest, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);   \
    return dest_ref--;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_fetch_dec(TYPE* dest, MemoryOrder, MemoryScopeCore  ) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);     \
    return dest_ref--;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_dec_fetch(TYPE* dest, MemoryOrder, MemoryScopeDevice) { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeDevice> dest_ref(*dest);   \
    return --dest_ref;                                                       \
  }                                                                          \
  template <class MemoryOrder>                                               \
  TYPE device_atomic_dec_fetch(TYPE* dest, MemoryOrder, MemoryScopeCore)   { \
    sycl_atomic_ref<TYPE, MemoryOrder, MemoryScopeCore> dest_ref(*dest);     \
    return --dest_ref;                                                       \
  }

#define DESUL_IMPL_SYCL_ATOMIC_INC_DEC(TYPE) \
  DESUL_IMPL_SYCL_ATOMIC_INC(TYPE)           \
  DESUL_IMPL_SYCL_ATOMIC_DEC(TYPE)
// clang-format on

DESUL_IMPL_SYCL_ATOMIC_INC_DEC(int)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(unsigned int)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(long)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(unsigned long)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(long long)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(unsigned long long)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(float)
DESUL_IMPL_SYCL_ATOMIC_INC_DEC(double)

#undef DESUL_IMPL_SYCL_ATOMIC_INC_DEC
#undef DESUL_IMPL_SYCL_ATOMIC_INC
#undef DESUL_IMPL_SYCL_ATOMIC_DEC
#undef DESUL_IMPL_SYCL_ATOMIC_OPER_FLOATING_POINT
#undef DESUL_IMPL_SYCL_ATOMIC_OPER_INTEGRAL
#undef DESUL_IMPL_SYCL_ATOMIC_OPER

}  // namespace Impl
}  // namespace desul

#endif
