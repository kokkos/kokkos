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

#define DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(OP)                                   \
  DESUL_FUNCTION T fetch_##OP(T arg) const noexcept {                           \
    return desul::atomic_fetch_##OP(ptr_, arg, MemoryOrder(), MemoryScope());   \
  }                                                                             \
  DESUL_FUNCTION T OP##_fetch(T arg) const noexcept {                           \
    return desul::atomic_##OP##_fetch(ptr_, arg, MemoryOrder(), MemoryScope()); \
  }

#define DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(COMPD_ASGMT, OP) \
  DESUL_FUNCTION T operator COMPD_ASGMT(T arg) const noexcept {          \
    return OP##_fetch(arg);                                              \
  }

  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(add)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(+=, add)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(sub)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(-=, sub)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(min)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(max)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(mul)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(*=, mul)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(div)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(/=, div)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(mod)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(%=, mod)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(and)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(&=, and)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(or)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(|=, or)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(xor)
  DESUL_IMPL_DEFINE_ATOMIC_COMPOUND_ASSIGNMENT_OP(^=, xor)
  DESUL_IMPL_DEFINE_ATOMIC_FETCH_OP(nand)

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
