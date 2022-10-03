/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_SIMD_AVX2_HPP
#define KOKKOS_SIMD_AVX2_HPP

#include <functional>
#include <type_traits>

#include <Kokkos_SIMD_Common.hpp>

#include <immintrin.h>

namespace Kokkos {
namespace Experimental {

namespace simd_abi {

template <int N>
class avx2_fixed_size {};

}  // namespace simd_abi

template <>
class simd_mask<double, simd_abi::avx2_fixed_size<4>> {
  __m256d m_value;

 public:
  class reference {
    __m256d& m_mask;
    int m_lane;
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION __m256d bit_mask() const {
      return _mm256_castsi256_pd(
          _mm256_setr_epi64x(
            -std::int64_t(m_lane == 0),
            -std::int64_t(m_lane == 1),
            -std::int64_t(m_lane == 2),
            -std::int64_t(m_lane == 3)));
    }

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(__m256d& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      if (value) {
        m_mask = _mm256_or_pd(bit_mask(), m_mask);
      } else {
        m_mask = _mm256_andnot_pd(bit_mask(), m_mask);
      }
      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      auto const bm = bit_mask();
      return 0 != _mm256_testc_pd(_mm256_and_pd(bm, m_mask), bm);
    }
  };
  using value_type                                  = bool;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(value_type value)
      : m_value(_mm256_castsi256_pd(
          _mm256_set1_epi64x(
            -std::int64_t(value)))
  {
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 4;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      __m256d const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256d()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return static_cast<value_type>(reference(m_value, int(i)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator||(simd_mask const& other) const {
    return simd_mask(_mm256_or_pd(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask
  operator&&(simd_mask const& other) const {
    return simd_mask(_mm256_and_pd(m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask operator!() const {
    auto const __m256d true_value(static_cast<__m256d>(simd_mask(true)));
    return simd_mask(_mm256_andnot(m_value, true_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      simd_mask const& other) const {
    return 0 != _mm256_testc_pd(m_value, other.m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      simd_mask const& other) const {
    return !operator==(other);
  }
};

template <>
class simd<double, simd_abi::avx2_fixed_size<4>> {
  __m256d m_value;

 public:
  using value_type = double;
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
      : m_value(_mm256_set1_pd(value_type(value))) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(double a, double b, double c,
                                             double d)
      : m_value(_mm256_setr_pd(a, b, c, d)) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      __m256d const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reinterpret_cast<value_type*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reinterpret_cast<value_type const*>(&m_value)[i];
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_loadu_pd(ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_storeu_pd(ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator __m256d()
      const {
    return m_value;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_LT_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_GT_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator<=(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_LE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator>=(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_GE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator==(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_EQ_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator!=(simd const& other) const {
    return mask_type(_mm256_cmp_pd(m_value, other.m_value, _CMP_NEQ_OS));
  }
};

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx2_fixed_size<4>>
    operator*(simd<double, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<double, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_mul_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx2_fixed_size<4>>
    operator/(simd<double, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<double, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_div_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx2_fixed_size<4>>
    operator+(simd<double, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<double, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_add_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx2_fixed_size<4>>
    operator-(simd<double, simd_abi::avx2_fixed_size<4>> const& lhs,
              simd<double, simd_abi::avx2_fixed_size<4>> const& rhs) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_sub_pd(static_cast<__m256d>(lhs), static_cast<__m256d>(rhs)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::avx2_fixed_size<4>>
    operator-(simd<double, simd_abi::avx2_fixed_size<4>> const& a) {
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_sub_pd(_mm256_set1_pd(0.0), static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
simd<double, simd_abi::avx2_fixed_size<4>> copysign(
    simd<double, simd_abi::avx2_fixed_size<4>> const& a,
    simd<double, simd_abi::avx2_fixed_size<4>> const& b) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_xor_pd(
          _mm256_andnot_pd(
              sign_mask, static_cast<__m256d>(a)),
          _mm256_and_pd(
              sign_mask, static_cast<__m256d>(b)));
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
      _mm256_blendv_pd(static_cast<__m256d>(a), static_cast<__m256d>(c),
                           static_cast<__m256d>(b)));
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
//KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
//void scatter_to(
//    double* mem,
//    simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) const {
//  _mm256_mask_i32scatter_pd(mem, static_cast<__mmask8>(m_mask),
//                            static_cast<__m256i>(index),
//                            static_cast<__m256d>(m_value), 8);
//}
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
    m_value = value_type(_mm256_maskload_pd(mem,
          _mm256_castpd_si256(static_cast<__m256d>(m_mask))));
  }
//KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
//void gather_from(
//    double const* mem,
//    simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
//  m_value = value_type(_mm256_mask_i32gather_pd(
//      _mm256_set1_pd(0.0), static_cast<__mmask8>(m_mask),
//      static_cast<__m256i>(index), mem, 8));
//}
  template <class U, std::enable_if_t<
                         std::is_convertible_v<
                             U, simd<double, simd_abi::avx2_fixed_size<4>>>,
                         bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<double, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = simd<double, simd_abi::avx2_fixed_size<4>>(_mm256_blendv_pd(
        static_cast<__m256d>(m_value),
        static_cast<__m256d>(x_as_value_type),
        static_cast<__m256d>(m_mask)));
  }
};

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION double reduce(
    const_where_expression<simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                           simd<double, simd_abi::avx2_fixed_size<4>>> const&
        x,
    double, std::plus<>) {
  simd<double, simd_abi::avx2_fixed_size<4>> masked(0.0);
  where(x.mask(), masked) = x.value();
  return _mm256_hadd_pd(static_cast<__m256d>(masked));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
