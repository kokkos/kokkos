//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_SIMD_NEON_HPP
#define KOKKOS_SIMD_NEON_HPP

#include <functional>
#include <type_traits>

#include <Kokkos_SIMD_Common.hpp>

#include <arm_neon.h>

namespace Kokkos {

namespace Experimental {

namespace simd_abi {

template <int N>
class neon_fixed_size {};

}  // namespace simd_abi

template <>
class simd_mask<double, simd_abi::neon_fixed_size<2>> {
  uint64x2_t m_value;

 public:
  class reference {
    uint64x2_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(uint64x2_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      m_mask = vsetq_lane_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0, m_mask, m_lane);
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return vgetq_lane_u64(m_mask, m_lane) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::neon_fixed_size<2>;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(vdupq_n_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0))
  {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, simd_abi::neon_fixed_size<2>> const& i32_mask);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 2;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      uint64x2_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator uint64x2_t()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<uint64x2_t&>(m_value), int(i)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(vorrq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(vandq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<uint64x2_t>(simd_mask(true));
    return simd_mask(veorq_u64(m_value, true_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    uint64x2_t const elementwise_equality = vceqq_u64(m_value, other.m_value);
    uint32x2_t const narrow_elementwise_equality = vqmovn_u64(elementwise_equality);
    uint64x1_t const overall_equality_neon = vreinterpret_u64_u32(narrow_elementwise_equality);
    uint64_t const overall_equality = vget_lane_u64(overall_equality_neon, 0);
    return overall_equality == 0xFFFFFFFFFFFFFFFFULL;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::int32_t, simd_abi::neon_fixed_size<2>> {
  uint32x2_t m_value;

 public:
  class reference {
    uint32x2_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(uint32x2_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      m_mask = vset_lane_u32(value ? 0xFFFFFFFFU : 0, m_mask, m_lane);
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return vget_lane_u32(m_mask, m_lane) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::neon_fixed_size<2>;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(vdup_n_u32(value ? 0xFFFFFFFFU : 0))
  {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, simd_abi::neon_fixed_size<2>> const& i32_mask);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 2;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      uint32x2_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator uint32x2_t()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<uint32x2_t&>(m_value), int(i)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(vorr_u32(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(vand_u32(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<uint32x2_t>(simd_mask(true));
    return simd_mask(veor_u32(m_value, true_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    uint32x2_t const elementwise_equality = vceq_u32(m_value, other.m_value);
    uint64x1_t const overall_equality_neon = vreinterpret_u64_u32(elementwise_equality);
    uint64_t const overall_equality = vget_lane_u64(overall_equality_neon, 0);
    return overall_equality == 0xFFFFFFFFFFFFFFFFULL;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::int64_t, simd_abi::neon_fixed_size<2>> {
  uint64x2_t m_value;

 public:
  class reference {
    uint64x2_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(uint64x2_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      m_mask = vsetq_lane_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0, m_mask, m_lane);
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return vgetq_lane_u64(m_mask, m_lane) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::neon_fixed_size<2>;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(vdupq_n_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0))
  {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, simd_abi::neon_fixed_size<2>> const& i32_mask);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 2;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      uint64x2_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator uint64x2_t()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<uint64x2_t&>(m_value), int(i)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(vorrq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(vandq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<uint64x2_t>(simd_mask(true));
    return simd_mask(veorq_u64(m_value, true_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    uint64x2_t const elementwise_equality = vceqq_u64(m_value, other.m_value);
    uint32x2_t const narrow_elementwise_equality = vqmovn_u64(elementwise_equality);
    uint64x1_t const overall_equality_neon = vreinterpret_u64_u32(narrow_elementwise_equality);
    uint64_t const overall_equality = vget_lane_u64(overall_equality_neon, 0);
    return overall_equality == 0xFFFFFFFFFFFFFFFFULL;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd_mask<std::uint64_t, simd_abi::neon_fixed_size<2>> {
  uint64x2_t m_value;

 public:
  class reference {
    uint64x2_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(uint64x2_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      m_mask = vsetq_lane_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0, m_mask, m_lane);
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      return vgetq_lane_u64(m_mask, m_lane) != 0;
    }
  };
  using value_type = bool;
  using abi_type   = simd_abi::neon_fixed_size<2>;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(vdupq_n_u64(value ? 0xFFFFFFFFFFFFFFFFULL : 0))
  {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<std::int32_t, simd_abi::neon_fixed_size<2>> const& i32_mask);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 2;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      uint64x2_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator uint64x2_t()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(
        reference(const_cast<uint64x2_t&>(m_value), int(i)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(vorrq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(vandq_u64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const true_value = static_cast<uint64x2_t>(simd_mask(true));
    return simd_mask(veorq_u64(m_value, true_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    uint64x2_t const elementwise_equality = vceqq_u64(m_value, other.m_value);
    uint32x2_t const narrow_elementwise_equality = vqmovn_u64(elementwise_equality);
    uint64x1_t const overall_equality_neon = vreinterpret_u64_u32(narrow_elementwise_equality);
    uint64_t const overall_equality = vget_lane_u64(overall_equality_neon, 0);
    return overall_equality == 0xFFFFFFFFFFFFFFFFULL;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd<double, simd_abi::neon_fixed_size<2>> {
  float64x2_t m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::neon_fixed_size<2>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    float64x2_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(float64x2_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(double value) const {
      m_value = vsetq_lane_f64(value, m_mask, m_lane);
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator double() const {
      return vgetq_lane_f64(m_mask, m_lane);
    }
  };
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 2;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(vdupq_n_f64(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                // basically, can you do { value_type r =
                // gen(std::integral_constant<std::size_t, i>()); }
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION simd(G&& gen)
  {
    vsetq_lane_f64(gen(std::integral_constant<std::size_t, 0>()), m_value, 0);
    vsetq_lane_f64(gen(std::integral_constant<std::size_t, 1>()), m_value, 1);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      float64x2_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return vgetq_lane(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = vld1q_f64(ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    vst1q_f64(ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator float64x2_t()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<(simd const& other) const {
    return mask_type(vcltq_f64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>(simd const& other) const {
    return mask_type(vcgtq_f64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<=(simd const& other) const {
    return mask_type(vcleq_f64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>=(simd const& other) const {
    return mask_type(vcgeq_f64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator==(simd const& other) const {
    return mask_type(vceqq_f64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator!=(simd const& other) const {
    return !(operator==(other));
  }
};

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::neon_fixed_size<2>>
    operator*(simd<double, simd_abi::neon_fixed_size<2>> const& lhs,
              simd<double, simd_abi::neon_fixed_size<2>> const& rhs) {
  return simd<double, simd_abi::neon_fixed_size<2>>(
      vmulq_f64(static_cast<float64x2_t>(lhs), static_cast<float64x2_t>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::neon_fixed_size<2>>
    operator/(simd<double, simd_abi::neon_fixed_size<2>> const& lhs,
              simd<double, simd_abi::neon_fixed_size<2>> const& rhs) {
  return simd<double, simd_abi::neon_fixed_size<2>>(
      vdivq_f64(static_cast<float64x2_t>(lhs), static_cast<float64x2_t>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::neon_fixed_size<2>>
    operator+(simd<double, simd_abi::neon_fixed_size<2>> const& lhs,
              simd<double, simd_abi::neon_fixed_size<2>> const& rhs) {
  return simd<double, simd_abi::neon_fixed_size<2>>(
      vaddq_f64(static_cast<float64x2_t>(lhs), static_cast<float64x2_t>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::neon_fixed_size<2>>
    operator-(simd<double, simd_abi::neon_fixed_size<2>> const& lhs,
              simd<double, simd_abi::neon_fixed_size<2>> const& rhs) {
  return simd<double, simd_abi::neon_fixed_size<2>>(
      vsubq_f64(static_cast<float64x2_t>(lhs), static_cast<float64x2_t>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::neon_fixed_size<2>>
    operator-(simd<double, simd_abi::neon_fixed_size<2>> const& a) {
  return simd<double, simd_abi::neon_fixed_size<2>>(
      vnegq_f64(static_cast<float64x2_t>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> copysign(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_xor_pd(_mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)),
                    _mm256_and_pd(sign_mask, static_cast<__m256d>(b))));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> abs(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> sqrt(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_sqrt_pd(static_cast<__m256d>(a)));
}

#ifdef __INTEL_COMPILER

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> cbrt(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_cbrt_pd(static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> exp(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_exp_pd(static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> log(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_log_pd(static_cast<__m256d>(a)));
}

#endif

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> fma(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_fmadd_pd(static_cast<__m256d>(a), static_cast<__m256d>(b),
                      static_cast<__m256d>(c)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> max(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_max_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> min(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_min_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> condition(
    simd_mask<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_blendv_pd(static_cast<__m256d>(c), static_cast<__m256d>(b),
                       static_cast<__m256d>(a)));
}

template <>
class simd<std::int32_t, simd_abi::avx2_fixed_size<4>> {
  __m128i m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm_set1_epi32(value_type(value))) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(std::int32_t a, std::int32_t b,
                                             std::int32_t c, std::int32_t d)
      : m_value(_mm_setr_epi32(a, b, c, d)) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION simd(G&& gen)
      : simd(gen(std::integral_constant<std::size_t, 0>()),
             gen(std::integral_constant<std::size_t, 1>()),
             gen(std::integral_constant<std::size_t, 2>()),
             gen(std::integral_constant<std::size_t, 3>())) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m128i const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::uint64_t, abi_type> const& other);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm_maskload_epi32(ptr, static_cast<__m128i>(mask_type(true)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask_type(true)), m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m128i()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator==(simd const& other) const {
    return mask_type(_mm_cmpeq_epi32(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>(simd const& other) const {
    return mask_type(_mm_cmpgt_epi32(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<(simd const& other) const {
    return mask_type(_mm_cmplt_epi32(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<=(simd const& other) const {
    return ((*this) < other) || ((*this) == other);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>=(simd const& other) const {
    return ((*this) > other) || ((*this) == other);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator!=(simd const& other) const {
    return !((*this) == other);
  }
};

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    operator-(simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(
      _mm_sub_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    operator+(simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(
      _mm_add_epi32(static_cast<__m128i>(lhs), static_cast<__m128i>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    condition(simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& a,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& b,
              simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(_mm_castps_si128(
      _mm_blendv_ps(_mm_castsi128_ps(static_cast<__m128i>(c)),
                    _mm_castsi128_ps(static_cast<__m128i>(b)),
                    _mm_castsi128_ps(static_cast<__m128i>(a)))));
}

template <>
class simd<std::int64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi64x(value_type(value))) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(std::int64_t a, std::int64_t b,
                                             std::int64_t c, std::int64_t d)
      : m_value(_mm256_setr_epi64x(a, b, c, d)) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION simd(G&& gen)
      : simd(gen(std::integral_constant<std::size_t, 0>()),
             gen(std::integral_constant<std::size_t, 1>()),
             gen(std::integral_constant<std::size_t, 2>()),
             gen(std::integral_constant<std::size_t, 3>())) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256i const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(
      simd<std::uint64_t, abi_type> const& other);
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(
      simd<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi64(ptr, static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_maskstore_epi64(ptr, static_cast<__m256i>(mask_type(true)), m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }
  // AVX2 only has eq and gt comparisons for int64
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator==(simd const& other) const {
    return mask_type(_mm256_cmpeq_epi64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>(simd const& other) const {
    return mask_type(_mm256_cmpgt_epi64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<(simd const& other) const {
    return other > (*this);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<=(simd const& other) const {
    return ((*this) < other) || ((*this) == other);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>=(simd const& other) const {
    return ((*this) > other) || ((*this) == other);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator!=(simd const& other) const {
    return !((*this) == other);
  }
};

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    operator-(simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
      _mm256_sub_epi64(static_cast<__m256i>(lhs), static_cast<__m256i>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    operator-(simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(0) - a;
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int64_t, simd_abi::avx2_fixed_size<4>> condition(
    simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a,
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& b,
    simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(_mm256_castpd_si256(
      _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                       _mm256_castsi256_pd(static_cast<__m256i>(b)),
                       _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

template <>
class simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> {
  __m256i m_value;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = simd_mask<value_type, abi_type>;
  using reference  = value_type&;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(_mm256_set1_epi64x(bit_cast<std::int64_t>(value_type(value)))) {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr simd(__m256i const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int32_t, abi_type> const& other)
      : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other))) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd(
      simd<std::int64_t, abi_type> const& other)
      : m_value(static_cast<__m256i>(other)) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd
  operator>>(unsigned int rhs) const {
    return _mm256_srli_epi64(m_value, rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator>>(
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& rhs) const {
    return _mm256_srlv_epi64(m_value,
                             _mm256_cvtepi32_epi64(static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd
  operator<<(unsigned int rhs) const {
    return _mm256_slli_epi64(m_value, rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator<<(
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& rhs) const {
    return _mm256_sllv_epi64(m_value,
                             _mm256_cvtepi32_epi64(static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd
  operator&(simd const& other) const {
    return _mm256_and_si256(m_value, other.m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd
  operator|(simd const& other) const {
    return _mm256_or_si256(m_value, other.m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256i()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator==(simd const& other) const {
    return mask_type(_mm256_cmpeq_epi64(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator!=(simd const& other) const {
    return !((*this) == other);
  }
};

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::simd(
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& other)
    : m_value(static_cast<__m256i>(other)) {}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> condition(
    simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& a,
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& b,
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(_mm256_castpd_si256(
      _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                       _mm256_castsi256_pd(static_cast<__m256i>(b)),
                       _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::simd(
    simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& other) {
  for (std::size_t i = 0; i < 4; ++i) {
    (*this)[i] = std::int32_t(other[i]);
  }
}

template <>
class const_where_expression<simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                             simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = simd<double, abi_type>;
  using mask_type  = simd_mask<double, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr mask_type const&
  mask() const {
    return m_mask;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr value_type const&
  value() const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, element_aligned_tag) const {
    _mm256_maskstore_pd(mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask)),
                        static_cast<__m256d>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      double* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }
};

template <>
class where_expression<simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                       simd<double, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<double, simd_abi::avx2_fixed_size<4>>,
          simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<double, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_maskload_pd(
        mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      double const* mem,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_pd(
        _mm256_set1_pd(0.0), mem, static_cast<__m128i>(index),
        static_cast<__m256d>(m_mask), 8));
  }
  template <class U,
            std::enable_if_t<std::is_convertible_v<
                                 U, simd<double, simd_abi::avx2_fixed_size<4>>>,
                             bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<double, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = simd<double, simd_abi::avx2_fixed_size<4>>(_mm256_blendv_pd(
        static_cast<__m256d>(m_value), static_cast<__m256d>(x_as_value_type),
        static_cast<__m256d>(m_mask)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
    simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = simd<std::int32_t, abi_type>;
  using mask_type  = simd_mask<std::int32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr mask_type const&
  mask() const {
    return m_mask;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr value_type const&
  value() const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, element_aligned_tag) const {
    _mm_maskstore_epi32(mem, static_cast<__m128i>(m_mask),
                        static_cast<__m128i>(m_value));
  }
};

template <>
class where_expression<simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
                       simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
          simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      simd<std::int32_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm_maskload_epi32(mem, static_cast<__m128i>(m_mask)));
  }
};

}  // namespace Experimental
}  // namespace Kokkos

#endif
