// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_BASE_HPP
#define KOKKOS_SIMD_BASE_HPP

#include <Kokkos_SIMD_Common.hpp>
#include "impl/Kokkos_SIMD_Impl_Macros.hpp"

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_Base.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Derived>
class basic_simd_mask_base {
 public:
  KOKKOS_SIMD_IMPL_SUBSCRIPT_OPERATOR([], subscript_operator, Impl::simd_size_t)

  KOKKOS_SIMD_IMPL_UNARY_OPERATOR(!, operator_lnot)
  KOKKOS_SIMD_IMPL_UNARY_OPERATOR(~, operator_bnot)

  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(&&, operator_land, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(||, operator_lor, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(&, operator_band, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(|, operator_bor, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(^, operator_xor, Derived const&,
                                   Derived const&)

  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(&=, operator_bandeq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(|=, operator_boreq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(^=, operator_xoreq,
                                                Derived const&)

  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(==, operator_eq, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(!=, operator_ne, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(>=, operator_ge, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(<=, operator_le, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(>, operator_gt, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(<, operator_lt, Derived const&)
};

template <typename Derived>
class basic_simd_base {
 public:
  KOKKOS_SIMD_IMPL_SUBSCRIPT_OPERATOR([], subscript_operator, Impl::simd_size_t)

  KOKKOS_SIMD_IMPL_UNARY_OPERATOR(-, operator_neg)
  KOKKOS_SIMD_IMPL_UNARY_OPERATOR(~, operator_bnot)

  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(+, operator_plus, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(-, operator_minus, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(*, operator_mul, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(/, operator_div, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(&, operator_band, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(|, operator_bor, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(^, operator_xor, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(<<, operator_sll, Derived const&,
                                   Impl::simd_size_t)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(<<, operator_sll, Derived const&,
                                   Derived const&)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(>>, operator_sra, Derived const&,
                                   Impl::simd_size_t)
  KOKKOS_SIMD_IMPL_BINARY_OPERATOR(>>, operator_sra, Derived const&,
                                   Derived const&)

  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(+=, operator_pluseq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(-=, operator_minuseq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(*=, operator_muleq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(/=, operator_diveq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(&=, operator_bandeq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(|=, operator_boreq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(^=, operator_xoreq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(<<=, operator_slleq,
                                                Derived const&)
  KOKKOS_SIMD_IMPL_COMPOUND_ASSIGNMENT_OPERATOR(>>=, operator_sraeq,
                                                Derived const&)

  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(==, operator_eq, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(!=, operator_ne, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(>=, operator_ge, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(<=, operator_le, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(>, operator_gt, Derived const&)
  KOKKOS_SIMD_IMPL_COMPARISON_OPERATOR(<, operator_lt, Derived const&)
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
