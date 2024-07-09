/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMIC_REF_HPP_
#define DESUL_ATOMIC_REF_HPP_

#include <desul/atomics/Common.hpp>
#include <desul/atomics/Generic.hpp>
#include <desul/atomics/Macros.hpp>

namespace desul {

template <typename T, typename MemoryOrder, typename MemoryScope>
class AtomicRef {
  T* ptr_;

 public:
  using value_type = T;
  using memory_order = MemoryOrder;
  using memory_scope = MemoryScope;

  DESUL_FUNCTION explicit AtomicRef(T& obj) : ptr_(&obj) {}

  DESUL_FUNCTION T operator=(T desired) const noexcept {
    store(desired);
    return desired;
  }

  DESUL_FUNCTION operator T() const noexcept { return load(); }

  DESUL_FUNCTION T load() const noexcept {
    return desul::atomic_load(ptr_, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION void store(T desired) const noexcept {
    return desul::atomic_store(ptr_, desired, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION T exchange(T desired) const noexcept {
    return desul::atomic_exchange(ptr_, desired, MemoryOrder(), MemoryScope());
  }

  // TODO compare_exchange_{weak,strong} and is_lock_free

#define DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(FETCH_OP, OP_U)                        \
  DESUL_FUNCTION T##FETCH_OP(T arg) const noexcept {                             \
    return desul::atomic_##FETCH_OP(ptr_, arg, MemoryOrder(), MemoryScope());    \
  }                                                                              \
  DESUL_FUNCTION T OP_U##fetch(T arg) const noexcept {                           \
    return desul::atomic_##OP_U##fetch(ptr_, arg, MemoryOrder(), MemoryScope()); \
  }

#define DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(COMPD_ASGMT, OP_U) \
  DESUL_FUNCTION T operator COMPD_ASGMT(T arg) const noexcept {            \
    return OP_U##fetch(arg);                                               \
  }

  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_add, add_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(+=, add_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_sub, sub_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(-=, sub_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_min, min_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_max, max_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_mul, mul_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(*=, mul_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_div, div_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(/=, div_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_mod, mod_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(%=, mod_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_and, and_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(&=, and_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_or, or_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(|=, or_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_xor, xor_)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(^=, xor_)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(fetch_nand, nand_)

#undef DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP
#undef DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP

#define DESUL_IMPL_DEFINE_ATOMIC_INCREMENT_DECREMENT(OPER, NAME)             \
  DESUL_FUNCTION T fetch_##NAME() const noexcept {                           \
    return desul::atomic_fetch_##NAME(ptr_, MemoryOrder(), MemoryScope());   \
  }                                                                          \
  DESUL_FUNCTION T NAME##_fetch() const noexcept {                           \
    return desul::atomic_##NAME##_fetch(ptr_, MemoryOrder(), MemoryScope()); \
  }                                                                          \
  DESUL_FUNCTION T operator OPER() const noexcept { return NAME##_fetch(); } \
  DESUL_FUNCTION T operator OPER(int) const noexcept { return fetch_##NAME(); }

  DESUL_IMPL_DEFINE_ATOMIC_INCREMENT_DECREMENT(++, inc)
  DESUL_IMPL_DEFINE_ATOMIC_INCREMENT_DECREMENT(--, dec)

#undef DESUL_IMPL_DEFINE_ATOMIC_INCREMENT_DECREMENT
};

}  // namespace desul

#endif
