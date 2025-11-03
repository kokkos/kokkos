// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_BASE_HPP
#define KOKKOS_SIMD_BASE_HPP

#include <Kokkos_SIMD_Common.hpp>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_Base.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

#define REQUIRE_INVOCABLE(FUNC_NAME) \
  requires requires(Derived d) {     \
    { d.FUNC_NAME() };               \
  }

#define REQUIRE_INVOCABLE_OP(FUNC_NAME) \
  requires requires(Derived d) {        \
    { d.operator_##FUNC_NAME() };       \
  }

#define REQUIRE_INVOCABLE_OP_WITH_ARG(FUNC_NAME, ARG) \
  requires requires(Derived d) {                      \
    { d.operator_##FUNC_NAME(ARG) };                  \
  }

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename Derived>
class basic_simd_mask_base {
 public:
  KOKKOS_FORCEINLINE_FUNCTION auto operator[](std::size_t i) const
    requires requires(Derived d) {
      { d.subscript_operator(i) };
    }
  {
    KOKKOS_IF_ON_HOST(
        (return static_cast<const Derived*>(this)->subscript_operator(i);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))
  }

  KOKKOS_FORCEINLINE_FUNCTION Derived operator!() const noexcept
      REQUIRE_INVOCABLE(operator_not) {
    KOKKOS_IF_ON_HOST((return static_cast<const Derived*>(this)->operator_not();))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator~() const noexcept
      REQUIRE_INVOCABLE(operator_not) {
    KOKKOS_IF_ON_HOST((return static_cast<const Derived*>(this)->operator_not();))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }

  KOKKOS_FORCEINLINE_FUNCTION Derived operator&&(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(land, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_land(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator||(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(lor, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_lor(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator&(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(band, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_band(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator|(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(bor, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_bor(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator^(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(bxor, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_bxor(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }

  KOKKOS_FORCEINLINE_FUNCTION Derived operator==(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(eq, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_eq(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator!=(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(ne, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_ne(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived();))
  }
};

template <typename Derived>
class basic_simd_base {
 public:
  KOKKOS_FORCEINLINE_FUNCTION auto operator[](std::size_t i) const
    requires requires(Derived d) {
      { d.subscript_operator(i) };
    }
  {
    KOKKOS_IF_ON_HOST(
        (return static_cast<const Derived*>(this)->subscript_operator(i);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))
  }

  KOKKOS_FORCEINLINE_FUNCTION auto operator!() const noexcept
      REQUIRE_INVOCABLE(operator_not) {
    KOKKOS_IF_ON_HOST((return static_cast<const Derived*>(this)->operator_not();))
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator~() const noexcept
      REQUIRE_INVOCABLE(operator_not) {
    KOKKOS_IF_ON_HOST((return static_cast<const Derived*>(this)->operator_not();))
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator-() const noexcept
      REQUIRE_INVOCABLE_OP(neg) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_neg();))
    KOKKOS_IF_ON_DEVICE((return typename Derived::value_type{};))
  }

  KOKKOS_FORCEINLINE_FUNCTION Derived operator+(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(plus, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_plus(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator-(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(minus, rhs) {
    KOKKOS_IF_ON_HOST(
        (return static_cast<Derived*>(this)->operator_minus(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator*(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(mul, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_mul(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator/(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(div, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_div(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator<<(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(sll, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_sll(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator>>(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(sra, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_sra(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator>>(int rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(sra, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_sra(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION Derived operator<<(int rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(sll, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_sll(rhs);))
    KOKKOS_IF_ON_DEVICE((return Derived{};))
  }

  KOKKOS_FORCEINLINE_FUNCTION auto operator==(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(eq, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_eq(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator!=(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(ne, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_ne(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator>=(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(ge, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_ge(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator<=(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(le, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_le(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator<(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(lt, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_lt(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
  KOKKOS_FORCEINLINE_FUNCTION auto operator>(Derived const& rhs) noexcept
      REQUIRE_INVOCABLE_OP_WITH_ARG(gt, rhs) {
    KOKKOS_IF_ON_HOST((return static_cast<Derived*>(this)->operator_gt(rhs);))
    KOKKOS_IF_ON_DEVICE((return typename Derived::mask_type{};))
  }
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
