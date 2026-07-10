// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_BASE_HPP
#define KOKKOS_SIMD_BASE_HPP

#include <Kokkos_SIMD_Common.hpp>
#include <impl/Kokkos_SIMD_Impl_Macros.hpp>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_Base.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

namespace Kokkos::Experimental::Impl {

template <typename Derived>
class basic_simd_mask_base {
 private:
  // using impl_ops = typename Derived::impl_ops;
  // using value_type = typename Derived::value_type;
  // using vector_type = typename Derived::impl_vector_type;

  KOKKOS_FORCEINLINE_FUNCTION
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator[](simd_size_t lane) const
    requires requires { Derived::impl_ops::extract(derived(), lane); }
  {
    return Derived::impl_ops::extract(derived(), lane);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator!() const noexcept
    requires requires { Derived::impl_ops::lnot(derived()); }
  {
    return Derived(Derived::impl_ops::lnot(derived()));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator~() const noexcept
    requires requires { Derived::impl_ops::bnot(derived()); }
  {
    return Derived(Derived::impl_ops::bnot(derived()));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator&&(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::land(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::land(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator||(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::lor(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::lor(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator&(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::band(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::band(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator|(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bor(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bor(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator^(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bxor(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bxor(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator&=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bandeq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bandeq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator|=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::boreq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::boreq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator^=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bxoreq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bxoreq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator==(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator!=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::neq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::neq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::ge(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::ge(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::le(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::le(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::gt(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::gt(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::lt(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::lt(lhs, rhs));
  }
};

template <typename Derived>
class basic_simd_base {
 private:
  // using impl_ops = typename Derived::impl_ops;
  // using value_type = typename Derived::value_type;
  // using vector_type = typename Derived::impl_vector_type;
  // using mask_type = typename Derived::mask_type;

  KOKKOS_FORCEINLINE_FUNCTION
  const Derived& derived() const { return static_cast<const Derived&>(*this); }

 public:
  // subscript
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator[](simd_size_t lane) const
    requires requires { Derived::impl_ops::extract(derived(), lane); }
  {
    return Derived::impl_ops::extract(derived(), lane);
  }

  // unary
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator-() const noexcept
    requires requires { Derived::impl_ops::neg(derived()); }
  {
    return Derived(Derived::impl_ops::neg(derived()));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto operator~() const noexcept
    requires requires { Derived::impl_ops::bnot(derived()); }
  {
    return Derived(Derived::impl_ops::bnot(derived()));
  }

  // binary
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator+(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::plus(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::plus(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator-(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::minus(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::minus(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator*(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::multiply(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::multiply(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator/(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::divide(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::divide(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator&(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::band(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::band(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator|(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bor(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bor(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator^(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bxor(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bxor(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<<(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::sll(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sll(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>>(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::sra(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sra(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<<(Derived const& lhs, simd_size_t rhs) noexcept
    requires requires { Derived::impl_ops::sll(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sll(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>>(Derived const& lhs, simd_size_t rhs) noexcept
    requires requires { Derived::impl_ops::sra(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sra(lhs, rhs));
  }

  // compound
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator+=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::plus_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::plus_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator-=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::minus_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::minus_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator*=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::multiply_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::multiply_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator/=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::divide_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::divide_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator&=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::band_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::band_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator|=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bor_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bor_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator^=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::bxor_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::bxor_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<<=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::sll_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sll_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>>=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::sra_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sra_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<<=(Derived const& lhs, simd_size_t rhs) noexcept
    requires requires { Derived::impl_ops::sll_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sll_eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>>=(Derived const& lhs, simd_size_t rhs) noexcept
    requires requires { Derived::impl_ops::sra_eq(lhs, rhs); }
  {
    return Derived(Derived::impl_ops::sra_eq(lhs, rhs));
  }

  // comparison
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator==(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::eq(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::eq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator!=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::neq(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::neq(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::ge(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::ge(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<=(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::le(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::le(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator>(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::gt(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::gt(lhs, rhs));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr friend auto operator<(Derived const& lhs, Derived const& rhs) noexcept
    requires requires { Derived::impl_ops::lt(lhs, rhs); }
  {
    return typename Derived::mask_type(Derived::impl_ops::lt(lhs, rhs));
  }
};

}  // namespace Kokkos::Experimental::Impl

#endif