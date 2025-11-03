// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_AVX2_HPP
#define KOKKOS_SIMD_AVX2_HPP

#include <functional>
#include <type_traits>

#include <Kokkos_SIMD_Common.hpp>
#include <Kokkos_BitManipulation.hpp>  // bit_cast

#include <immintrin.h>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_AVX2.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

namespace Kokkos {

namespace Experimental {

namespace simd_abi {

template <int N>
class avx2_fixed_size {};

}  // namespace simd_abi

template <>
class basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256d;
#else
  using implementation_type = char[sizeof(__m256d)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(value))))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask(
            [&](std::size_t i) { return static_cast<double>(other[i]); }) {}
  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
      basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int64_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_castsi256_pd(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>())))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm256_movemask_pd(m_value) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm256_andnot_pd(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_pd(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_pd(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_pd(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_pd(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_xor_pd(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_movemask_pd(m_value) ==
                           _mm256_movemask_pd(rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m128;
#else
  using implementation_type = char[sizeof(__m128)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_castsi128_ps(_mm_set1_epi32(-std::int32_t(value))))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask(
            [&](std::size_t i) { return static_cast<float>(other[i]); }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in) {
  }
#endif
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_castsi128_ps(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>())))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm_movemask_ps(m_value) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm_andnot_ps(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_and_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_or_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_and_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_or_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_xor_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_movemask_ps(m_value) ==
                           _mm_movemask_ps(rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256;
#else
  using implementation_type = char[sizeof(__m256)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 8; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_castsi256_ps(_mm256_set1_epi32(-std::int32_t(value))))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask(
            [&](std::size_t i) { return static_cast<float>(other[i]); }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_castsi256_ps(_mm256_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 4>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 5>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 6>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 7>())))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm256_movemask_ps(m_value) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm256_andnot_ps(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_xor_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_movemask_ps(m_value) ==
                           _mm256_movemask_ps(rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m128i;
#else
  using implementation_type = char[sizeof(__m128i)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_set1_epi32(-std::int32_t(value)))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : basic_simd_mask(
            [&](std::size_t i) { return static_cast<std::int32_t>(other[i]); })
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>()))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm_movemask_ps(_mm_castsi128_ps(m_value)) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm_andnot_si128(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) const noexcept {
    return basic_simd_mask(_mm_and_si128(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) const noexcept {
    return basic_simd_mask(_mm_or_si128(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_and_si128(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_or_si128(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm_xor_si128(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) const noexcept {
    return basic_simd_mask(_mm_movemask_ps(_mm_castsi128_ps(m_value)) ==
                           _mm_movemask_ps(_mm_castsi128_ps(rhs.m_value)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) const noexcept {
    return !operator_ne(rhs);
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type = char[sizeof(__m256i)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 8; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi32(-std::int32_t(value)))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](std::size_t i) {
          return static_cast<std::int32_t>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_setr_epi32(
            -std::int32_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 3>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 4>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 5>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 6>())),
            -std::int32_t(gen(std::integral_constant<std::size_t, 7>()))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm256_movemask_ps(_mm256_castsi256_ps(m_value)) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm256_andnot_si256(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_xor_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(
        _mm256_movemask_ps(_mm256_castsi256_ps(m_value)) ==
        _mm256_movemask_ps(_mm256_castsi256_ps(rhs.m_value)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type = char[sizeof(__m256i)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(-std::int64_t(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](std::size_t i) {
          return static_cast<std::int64_t>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<double, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>()))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm256_andnot_si256(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_xor_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(
        _mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
        _mm256_movemask_pd(_mm256_castsi256_pd(rhs.m_value)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type = char[sizeof(__m256i)];
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;
  implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(-std::int64_t(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      __m256i const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](std::size_t i) {
          return static_cast<std::uint64_t>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<double, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<std::size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<std::size_t, 3>()))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    return (_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) & (1 << i)) != 0;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_not() const noexcept {
    return basic_simd_mask(
        _mm256_andnot_si256(m_value, basic_simd_mask(true).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_lor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_land(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_band(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_and_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_or_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_bxor(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(_mm256_xor_si256(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_eq(basic_simd_mask const& rhs) noexcept {
    return basic_simd_mask(
        _mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
        _mm256_movemask_pd(_mm256_castsi256_pd(rhs.m_value)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd_mask
  operator_ne(basic_simd_mask const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtps_pd(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm_cvtepi32_ps(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
    basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castsi256_ps(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm_castps_si128(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
    basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castps_si256(static_cast<__m256>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<double, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castpd_si256(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<double, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_castpd_si256(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    basic_simd_mask<std::int64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

template <>
class basic_simd<double, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<double, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256d;
#else
  using implementation_type char[sizeof(__m256d)];
#endif
  implementation_type m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_pd(value_type(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_setr_pd(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm256_load_pd(ptr);
        } else { m_value = _mm256_loadu_pd(ptr); }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST(
        (m_value = _mm256_maskload_pd(
             ptr, _mm256_castpd_si256(static_cast<__m256d>(mask)));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_loadu_pd(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm256_load_pd(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_storeu_pd(ptr, m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm256_store_pd(ptr, m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_base<
      basic_simd<double, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    value_type tmp[size()];
    _mm256_storeu_pd(tmp, m_value);
    return tmp[i];
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_neg() const noexcept {
    return basic_simd(
        _mm256_sub_pd(_mm256_set1_pd(0.0), static_cast<__m256d>(m_value)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_add_pd(static_cast<__m256d>(m_value),
                                    static_cast<__m256d>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sub_pd(static_cast<__m256d>(m_value),
                                    static_cast<__m256d>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_mul_pd(static_cast<__m256d>(m_value),
                                    static_cast<__m256d>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_div(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_div_pd(static_cast<__m256d>(m_value),
                                    static_cast<__m256d>(rhs)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_EQ_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_NEQ_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_GE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_LE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_GT_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_LT_OS));
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
copysign(Experimental::basic_simd<
             double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
         Experimental::basic_simd<
             double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_xor_pd(_mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)),
                    _mm256_and_pd(sign_mask, static_cast<__m256d>(b))));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  __m256d const sign_mask = _mm256_set1_pd(-0.0);
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_round_pd(static_cast<__m256d>(a),
                      (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_round_pd(static_cast<__m256d>(a),
                      (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_round_pd(static_cast<__m256d>(a),
                      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_round_pd(static_cast<__m256d>(a),
                      (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
sqrt(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_sqrt_pd(static_cast<__m256d>(a)));
}

#ifdef __INTEL_COMPILER

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
cbrt(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_cbrt_pd(static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
exp(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_exp_pd(static_cast<__m256d>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
log(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_log_pd(static_cast<__m256d>(a)));
}

#endif

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
fma(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& c) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_fmadd_pd(static_cast<__m256d>(a), static_cast<__m256d>(b),
                      static_cast<__m256d>(c)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
max(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_max_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
min(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_min_pd(static_cast<__m256d>(a), static_cast<__m256d>(b)));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const double* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_pd(ptr, static_cast<__m256d>(simd));
  } else {
    _mm256_storeu_pd(ptr, static_cast<__m256d>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_pd(ptr, _mm256_castpd_si256(static_cast<__m256d>(mask)),
                      static_cast<__m256d>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_pd(ptr, _mm256_castpd_si256(static_cast<__m256d>(mask)),
                      static_cast<__m256d>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(
      _mm256_blendv_pd(static_cast<__m256d>(c), static_cast<__m256d>(b),
                       static_cast<__m256d>(a)));
}

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m128;
#else
  using implementation_type char[sizeof(__m128)];
#endif
  implementation_type m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_set1_ps(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<double, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_setr_ps(gen(std::integral_constant<std::size_t, 0>()),
                            gen(std::integral_constant<std::size_t, 1>()),
                            gen(std::integral_constant<std::size_t, 2>()),
                            gen(std::integral_constant<std::size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm_load_ps(ptr);
        } else { m_value = _mm_loadu_ps(ptr); }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((m_value = _mm_maskload_ps(
                           ptr, _mm_castps_si128(static_cast<__m128>(mask)));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm_loadu_ps(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm_load_ps(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm_storeu_ps(ptr, m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm_store_ps(ptr, m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    auto index = _mm_cvtsi32_si128(i);
    auto tmp   = _mm_permutevar_ps(m_value, index);
    return _mm_cvtss_f32(tmp);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_neg() const noexcept {
    return basic_simd(_mm_sub_ps(_mm_set1_ps(0.0), m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_add_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_sub_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_mul_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_div(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_div_ps(m_value, rhs.m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpeq_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpneq_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpge_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmple_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpgt_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmplt_ps(m_value, rhs.m_value));
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
copysign(Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
         Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  __m128 const sign_mask = _mm_set1_ps(-0.0);
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_xor_ps(_mm_andnot_ps(sign_mask, static_cast<__m128>(a)),
                 _mm_and_ps(sign_mask, static_cast<__m128>(b))));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> abs(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  __m128 const sign_mask = _mm_set1_ps(-0.0);
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_andnot_ps(sign_mask, static_cast<__m128>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_round_ps(static_cast<__m128>(a),
                   (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_round_ps(static_cast<__m128>(a),
                   (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_round_ps(static_cast<__m128>(a),
                   (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_round_ps(static_cast<__m128>(a),
                   (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
sqrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_sqrt_ps(static_cast<__m128>(a)));
}

#ifdef __INTEL_COMPILER

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
cbrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_cbrt_ps(static_cast<__m128>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> exp(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_exp_ps(static_cast<__m128>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> log(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_log_ps(static_cast<__m128>(a)));
}

#endif

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> fma(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& c) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_fmadd_ps(static_cast<__m128>(a), static_cast<__m128>(b),
                   static_cast<__m128>(c)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> max(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_max_ps(static_cast<__m128>(a), static_cast<__m128>(b)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> min(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_min_ps(static_cast<__m128>(a), static_cast<__m128>(b)));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const float* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm_store_ps(ptr, static_cast<__m128>(simd));
  } else {
    _mm_storeu_ps(ptr, static_cast<__m128>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm_maskstore_ps(ptr, _mm_castps_si128(static_cast<__m128>(mask)),
                   static_cast<__m128>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm_maskstore_ps(ptr, _mm_castps_si128(static_cast<__m128>(mask)),
                   static_cast<__m128>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& c) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(_mm_blendv_ps(
      static_cast<__m128>(c), static_cast<__m128>(b), static_cast<__m128>(a)));
}

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<8>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256;
#else
  using implementation_type char[sizeof(__m256)];
#endif
  implementation_type m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 8; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_ps(value_type(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd(G&& gen)
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_setr_ps(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>()),
                               gen(std::integral_constant<std::size_t, 4>()),
                               gen(std::integral_constant<std::size_t, 5>()),
                               gen(std::integral_constant<std::size_t, 6>()),
                               gen(std::integral_constant<std::size_t, 7>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm256_load_ps(ptr);
        } else { m_value = _mm256_loadu_ps(ptr); }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST(
        (m_value = _mm256_maskload_ps(
             ptr, _mm256_castps_si256(static_cast<__m256>(mask)));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_loadu_ps(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm256_load_ps(ptr);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_storeu_ps(ptr, m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm256_store_ps(ptr, m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<8>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    auto index = _mm256_set1_epi32(i);
    auto tmp   = _mm256_permutevar8x32_ps(m_value, index);
    return _mm256_cvtss_f32(tmp);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_neg() const noexcept {
    return basic_simd(_mm256_sub_ps(_mm256_set1_ps(0.0), m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_add_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sub_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_mul_ps(m_value, rhs.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_div(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_div_ps(m_value, rhs.m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_EQ_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_NEQ_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_GE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_LE_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_GT_OS));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_LT_OS));
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
copysign(Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
         Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  __m256 const sign_mask = _mm256_set1_ps(-0.0);
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_xor_ps(_mm256_andnot_ps(sign_mask, static_cast<__m256>(a)),
                    _mm256_and_ps(sign_mask, static_cast<__m256>(b))));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> abs(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  __m256 const sign_mask = _mm256_set1_ps(-0.0);
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_andnot_ps(sign_mask, static_cast<__m256>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
floor(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_round_ps(static_cast<__m256>(a),
                      (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
ceil(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_round_ps(static_cast<__m256>(a),
                      (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
round(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_round_ps(static_cast<__m256>(a),
                      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
trunc(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_round_ps(static_cast<__m256>(a),
                      (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
sqrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_sqrt_ps(static_cast<__m256>(a)));
}

#ifdef __INTEL_COMPILER

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
cbrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_cbrt_ps(static_cast<__m256>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> exp(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_exp_ps(static_cast<__m256>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> log(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_log_ps(static_cast<__m256>(a)));
}

#endif

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> fma(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& c) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_fmadd_ps(static_cast<__m256>(a), static_cast<__m256>(b),
                      static_cast<__m256>(c)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> max(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_max_ps(static_cast<__m256>(a), static_cast<__m256>(b)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> min(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_min_ps(static_cast<__m256>(a), static_cast<__m256>(b)));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(const float* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_ps(ptr, static_cast<__m256>(simd));
  } else {
    _mm256_storeu_ps(ptr, static_cast<__m256>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
    FlagType) {
  _mm256_maskstore_ps(ptr, _mm256_castps_si256(static_cast<__m256>(mask)),
                      static_cast<__m256>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
    FlagType) {
  _mm256_maskstore_ps(ptr, _mm256_castps_si256(static_cast<__m256>(mask)),
                      static_cast<__m256>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<8>> condition(
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& a,
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& b,
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& c) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(_mm256_blendv_ps(
      static_cast<__m256>(c), static_cast<__m256>(b), static_cast<__m256>(a)));
}

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m128i;
#else
  using implementation_type = char[sizeof(__m128i)];
#endif
  implementation_type m_value;
  using base_t = Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_set1_epi32(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<double, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm_setr_epi32(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm_load_si128(reinterpret_cast<__m128i const*>(ptr));
        } else {
          m_value = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr));
        }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST(
        (m_value = _mm_maskload_epi32(ptr, static_cast<__m128i>(mask));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm_maskload_epi32(ptr, static_cast<__m128i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm_maskload_epi32(ptr, static_cast<__m128i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask_type(true)), m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask_type(true)), m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    switch (i) {
      case 0: return _mm_extract_epi32(m_value, 0x0);
      case 1: return _mm_extract_epi32(m_value, 0x1);
      case 2: return _mm_extract_epi32(m_value, 0x2);
      case 3: return _mm_extract_epi32(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
// missing return statement warning with cuda 13
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1300) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_not() const noexcept {
    return basic_simd(
        _mm_andnot_si128(m_value, basic_simd(~value_type(0)).m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_band(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_and_si128(static_cast<__m128i>(m_value),
                                    static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bor(basic_simd const& rhs) noexcept {
    return basic_simd(
        _mm_or_si128(static_cast<__m128i>(m_value), static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bxor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_xor_si128(static_cast<__m128i>(m_value),
                                    static_cast<__m128i>(rhs)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_add_epi32(static_cast<__m128i>(m_value),
                                    static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_sub_epi32(static_cast<__m128i>(m_value),
                                    static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_mullo_epi32(static_cast<__m128i>(m_value),
                                      static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_sllv_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(basic_simd const& rhs) noexcept {
    return basic_simd(_mm_srav_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(int rhs) noexcept {
    return basic_simd(_mm_srai_epi32(static_cast<__m128i>(m_value), rhs));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(int rhs) noexcept {
    return basic_simd(_mm_slli_epi32(static_cast<__m128i>(m_value), rhs));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpeq_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return !operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return operator_gt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return operator_lt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmplt_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm_cmpgt_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  __m128i const rhs = static_cast<__m128i>(a);
  return Experimental::basic_simd<std::int32_t,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm_abs_epi32(rhs));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_cvtepi32_pd(static_cast<__m128i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_cvtepi32_pd(static_cast<__m128i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_cvtepi32_pd(static_cast<__m128i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_cvtepi32_pd(static_cast<__m128i>(a)));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::int32_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr, [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm_store_si128(reinterpret_cast<__m128i*>(ptr),
                    static_cast<__m128i>(simd));
  } else {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr),
                     static_cast<__m128i>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask),
                      static_cast<__m128i>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm_maskstore_epi32(ptr, static_cast<__m128i>(mask),
                      static_cast<__m128i>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(
      _mm_castps_si128(
          _mm_blendv_ps(_mm_castsi128_ps(static_cast<__m128i>(c)),
                        _mm_castsi128_ps(static_cast<__m128i>(b)),
                        _mm_castsi128_ps(static_cast<__m128i>(a)))));
}

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type char[sizeof(__m256i)];
#endif
  implementation_type m_value;
  using base_t = Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 8; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      __m256i const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi32(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(
            _mm256_setr_epi32(gen(std::integral_constant<std::size_t, 0>()),
                              gen(std::integral_constant<std::size_t, 1>()),
                              gen(std::integral_constant<std::size_t, 2>()),
                              gen(std::integral_constant<std::size_t, 3>()),
                              gen(std::integral_constant<std::size_t, 4>()),
                              gen(std::integral_constant<std::size_t, 5>()),
                              gen(std::integral_constant<std::size_t, 6>()),
                              gen(std::integral_constant<std::size_t, 7>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm256_load_si256(reinterpret_cast<__m256i const*>(ptr));
        } else {
          m_value = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr));
        }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST(
        (m_value = _mm256_maskload_epi32(ptr, static_cast<__m256i>(mask));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi32(ptr, static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm256_maskload_epi32(ptr, static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_maskstore_epi32(ptr, static_cast<__m256i>(mask_type(true)), m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm256_maskstore_epi32(ptr, static_cast<__m256i>(mask_type(true)), m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const {
    return m_value;
  }

 private:
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
// _mm256_cvtsi256_si32 was not added in GCC until 11
#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU < 1100)
    value_type tmp[size()];
    _mm256_maskstore_epi32(tmp, static_cast<__m256i>(mask_type(true)), m_value);
    return tmp[i];
#else
    auto index = _mm256_set1_epi32(i);
    auto tmp   = _mm256_permutevar8x32_epi32(m_value, index);
    return _mm256_cvtsi256_si32(tmp);
#endif
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_not() const noexcept {
    return basic_simd(
        _mm256_andnot_si256(m_value, basic_simd(~value_type(0)).m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_band(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_and_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_or_si256(static_cast<__m256i>(m_value),
                                      static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bxor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_xor_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_add_epi32(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sub_epi32(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_mullo_epi32(static_cast<__m256i>(m_value),
                                         static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sllv_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_srav_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(int rhs) noexcept {
    return basic_simd(_mm256_slli_epi32(static_cast<__m256i>(m_value), rhs));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(int rhs) noexcept {
    return basic_simd(_mm256_srai_epi32(static_cast<__m256i>(m_value), rhs));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return !operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return operator_gt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return operator_lt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmpgt_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return !operator_ge(rhs);
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>
abs(Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  __m256i const rhs = static_cast<__m256i>(a);
  return Experimental::basic_simd<std::int32_t,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_abs_epi32(rhs));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
floor(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_cvtepi32_ps(static_cast<__m256i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
ceil(Experimental::basic_simd<
     std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_cvtepi32_ps(static_cast<__m256i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
round(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_cvtepi32_ps(static_cast<__m256i>(a)));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
trunc(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  return Experimental::basic_simd<float,
                                  Experimental::simd_abi::avx2_fixed_size<8>>(
      _mm256_cvtepi32_ps(static_cast<__m256i>(a)));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(const std::int32_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr, [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                       static_cast<__m256i>(simd));
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                        static_cast<__m256i>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
    FlagType) {
  _mm256_maskstore_epi32(ptr, static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
    FlagType) {
  _mm256_maskstore_epi32(ptr, static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> condition(
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& a,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& b,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& c) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(
      _mm256_castps_si256(
          _mm256_blendv_ps(_mm256_castsi256_ps(static_cast<__m256i>(c)),
                           _mm256_castsi256_ps(static_cast<__m256i>(b)),
                           _mm256_castsi256_ps(static_cast<__m256i>(a)))));
}

template <>
class basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type char[sizeof(__m256i)];
#endif
  implementation_type m_value;
  using base_t = Impl::basic_simd_base<
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  static_assert(sizeof(long long) == 8);

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(
      basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      __m256i const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::uint64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(
            _mm256_setr_epi64x(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        } else {
          m_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        }))
  }
  template <typename FlagType>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                        static_cast<__m256i>(mask));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr),
                           static_cast<__m256i>(mask_type(true)), m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr),
                           static_cast<__m256i>(mask_type(true)), m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const {
    return m_value;
  }

 private:
  friend class basic_simd_base<
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    switch (i) {
      case 0: return _mm256_extract_epi64(m_value, 0x0);
      case 1: return _mm256_extract_epi64(m_value, 0x1);
      case 2: return _mm256_extract_epi64(m_value, 0x2);
      case 3: return _mm256_extract_epi64(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1300) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_not() const noexcept {
    return basic_simd(
        _mm256_andnot_si256(m_value, basic_simd(~value_type(0)).m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_neg() const noexcept {
    return basic_simd(
        _mm256_sub_epi64(_mm256_set1_epi64x(0), static_cast<__m256i>(m_value)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_add_epi64(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sub_epi64(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit signed integers is not available in AVX2
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd([&](std::size_t i) { return m_value[i] * rhs[i]; });
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_band(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_and_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_or_si256(static_cast<__m256i>(m_value),
                                      static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bxor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_xor_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sllv_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(basic_simd const& rhs) noexcept {
    return basic_simd([&](std::size_t i) { return m_value[i] >> rhs[i]; });
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(int rhs) noexcept {
    return basic_simd(_mm256_slli_epi64(static_cast<__m256i>(m_value), rhs));
  }
  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(int rhs) noexcept {
    return basic_simd([&](std::size_t i) { return m_value[i] >> rhs; });
  }

  // AVX2 only has eq and gt comparisons for int64
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return !operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ge(basic_simd const& rhs) noexcept {
    return operator_gt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_le(basic_simd const& rhs) noexcept {
    return operator_lt(rhs) || operator_eq(rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_gt(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmpgt_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_lt(basic_simd const& rhs) noexcept {
    return !operator_lt(rhs);
  }
};

}  // namespace Experimental

// Manually computing absolute values, because _mm256_abs_epi64
// is not in AVX2; it's available in AVX512.
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<std::int64_t,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      [&](std::size_t i) { return (a[i] < 0) ? -a[i] : a[i]; });
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::int64_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr, [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                       static_cast<__m256i>(simd));
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                        static_cast<__m256i>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr,
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                         static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr,
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                         static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
      _mm256_castpd_si256(
          _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                           _mm256_castsi256_pd(static_cast<__m256i>(b)),
                           _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

template <>
class basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
#ifndef KOKKOS_DEVICE_SIMD
  using implementation_type = __m256i;
#else
  using implementation_type char[sizeof(__m256i)];
#endif
  implementation_type m_value;
  using base_t = Impl::basic_simd_base<
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t size() { return 4; }

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept                  = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd const&) noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(basic_simd&&) noexcept      = default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd const&) =
      default;
  KOKKOS_FORCEINLINE_FUNCTION basic_simd& operator=(basic_simd&&) noexcept =
      default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(
            Kokkos::bit_cast<std::int64_t>(value_type(value))))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd(
      __m256i const& value_in) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(value_in)
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(basic_simd(
            [&](std::size_t i) { return static_cast<value_type>(other[i]); }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<std::size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
#ifndef KOKKOS_DEVICE_SIMD
      : m_value(
            _mm256_setr_epi64x(gen(std::integral_constant<std::size_t, 0>()),
                               gen(std::integral_constant<std::size_t, 1>()),
                               gen(std::integral_constant<std::size_t, 2>()),
                               gen(std::integral_constant<std::size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        if constexpr (std::is_same_v<
                          FlagType,
                          simd_flags<simd_alignment_vector_aligned>>) {
          m_value = _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
        } else {
          m_value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        }))
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, FlagType) noexcept {
    KOKKOS_IF_ON_HOST((
        m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                        static_cast<__m256i>(mask));))
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_load() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = _mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                    static_cast<__m256i>(mask_type(true)));
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr),
                           static_cast<__m256i>(mask_type(true)), m_value);
  }
  KOKKOS_DEPRECATED_WITH_COMMENT("Use simd_unchecked_store() instead")
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(ptr),
                           static_cast<__m256i>(mask_type(true)), m_value);
  }
#endif

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const {
    return m_value;
  }

 private:
  friend class basic_simd_base<
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  subscript_operator(std::size_t i) const {
    switch (i) {
      case 0: return _mm256_extract_epi64(m_value, 0x0);
      case 1: return _mm256_extract_epi64(m_value, 0x1);
      case 2: return _mm256_extract_epi64(m_value, 0x2);
      case 3: return _mm256_extract_epi64(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1300) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_not() const noexcept {
    return basic_simd(
        _mm256_andnot_si256(m_value, basic_simd(~value_type(0)).m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_plus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_add_epi64(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_minus(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_sub_epi64(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit unsigned integers is not available in AVX2
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_mul(basic_simd const& rhs) noexcept {
    return basic_simd([&](std::size_t i) { return m_value[i] * rhs[i]; });
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_band(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_and_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_or_si256(static_cast<__m256i>(m_value),
                                      static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_bxor(basic_simd const& rhs) noexcept {
    return basic_simd(_mm256_xor_si256(static_cast<__m256i>(m_value),
                                       static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(basic_simd const& rhs) noexcept {
    return _mm256_sllv_epi64(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(basic_simd const& rhs) noexcept {
    return _mm256_srlv_epi64(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sll(int rhs) noexcept {
    return _mm256_slli_epi64(static_cast<__m256i>(m_value), rhs);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd
  operator_sra(int rhs) noexcept {
    return _mm256_srli_epi64(static_cast<__m256i>(m_value), rhs);
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_eq(basic_simd const& rhs) noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  operator_ne(basic_simd const& rhs) noexcept {
    return !operator_eq(rhs);
  }
};

}  // namespace Experimental

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return a;
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  return Experimental::basic_simd<double,
                                  Experimental::simd_abi::avx2_fixed_size<4>>(
      _mm256_setr_pd(a[0], a[1], a[2], a[3]));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::uint64_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_unchecked_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_unchecked_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename... Flags>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_partial_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_partial_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr, [[maybe_unused]] FlagType flag = simd_flag_default) {
  if constexpr (std::is_same_v<FlagType,
                               simd_flags<simd_alignment_vector_aligned>>) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                       static_cast<__m256i>(simd));
  } else {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                        static_cast<__m256i>(simd));
  }
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                         static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

template <typename FlagType>
KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    FlagType) {
  _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                         static_cast<__m256i>(mask),
                         static_cast<__m256i>(simd));
}

KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(
      _mm256_castpd_si256(
          _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                           _mm256_castsi256_pd(static_cast<__m256i>(b)),
                           _mm256_castsi256_pd(static_cast<__m256i>(a)))));
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtps_pd(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<double, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtpd_ps(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm_cvtepi32_ps(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<8>>::basic_simd(
    basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_ps(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm_cvtps_epi32(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<double, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtpd_epi32(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd(
    basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtps_epi32(static_cast<__m256>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::uint64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    basic_simd<std::int64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
template <>
class KOKKOS_DEPRECATED const_where_expression<
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>,
    basic_simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = basic_simd<double, abi_type>;
  using mask_type  = basic_simd_mask<double, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, element_aligned_tag) const {
    _mm256_maskstore_pd(mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask)),
                        static_cast<__m256d>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, vector_aligned_tag) const {
    _mm256_maskstore_pd(mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask)),
                        static_cast<__m256d>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(double* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
                      index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED
    where_expression<basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>,
                     basic_simd<double, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>,
          basic_simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      basic_simd<double, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_maskload_pd(
        mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, vector_aligned_tag) {
    m_value = value_type(_mm256_maskload_pd(
        mem, _mm256_castpd_si256(static_cast<__m256d>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      double const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_pd(
        static_cast<__m256d>(m_value), mem, static_cast<__m128i>(index),
        static_cast<__m256d>(m_mask), 8));
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, basic_simd<double, simd_abi::avx2_fixed_size<4>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<double, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = basic_simd<double, simd_abi::avx2_fixed_size<4>>(_mm256_blendv_pd(
        static_cast<__m256d>(m_value), static_cast<__m256d>(x_as_value_type),
        static_cast<__m256d>(m_mask)));
  }
};

template <>
class KOKKOS_DEPRECATED
    const_where_expression<basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>,
                           basic_simd<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = basic_simd<float, abi_type>;
  using mask_type  = basic_simd_mask<float, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, element_aligned_tag) const {
    _mm_maskstore_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask)),
                     static_cast<__m128>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, vector_aligned_tag) const {
    _mm_maskstore_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask)),
                     static_cast<__m128>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(float* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
                      index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED
    where_expression<basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>,
                     basic_simd<float, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>,
          basic_simd<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask_arg,
      basic_simd<float, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, element_aligned_tag) {
    m_value = value_type(
        _mm_maskload_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, vector_aligned_tag) {
    m_value = value_type(
        _mm_maskload_ps(mem, _mm_castps_si128(static_cast<__m128>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      float const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm_mask_i32gather_ps(static_cast<__m128>(m_value),
                                               mem, static_cast<__m128i>(index),
                                               static_cast<__m128>(m_mask), 4));
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, basic_simd<float, simd_abi::avx2_fixed_size<4>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<float, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = basic_simd<float, simd_abi::avx2_fixed_size<4>>(_mm_blendv_ps(
        static_cast<__m128>(m_value), static_cast<__m128>(x_as_value_type),
        static_cast<__m128>(m_mask)));
  }
};

template <>
class KOKKOS_DEPRECATED
    const_where_expression<basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>,
                           basic_simd<float, simd_abi::avx2_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using value_type = basic_simd<float, abi_type>;
  using mask_type  = basic_simd_mask<float, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, element_aligned_tag) const {
    _mm256_maskstore_ps(mem, _mm256_castps_si256(static_cast<__m256>(m_mask)),
                        static_cast<__m256>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, vector_aligned_tag) const {
    _mm256_maskstore_ps(mem, _mm256_castps_si256(static_cast<__m256>(m_mask)),
                        static_cast<__m256>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(float* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const&
                      index) const {
    for (std::size_t lane = 0; lane < value_type::size(); ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED
    where_expression<basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>,
                     basic_simd<float, simd_abi::avx2_fixed_size<8>>>
    : public const_where_expression<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>,
          basic_simd<float, simd_abi::avx2_fixed_size<8>>> {
 public:
  where_expression(
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask_arg,
      basic_simd<float, simd_abi::avx2_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, element_aligned_tag) {
    m_value = value_type(_mm256_maskload_ps(
        mem, _mm256_castps_si256(static_cast<__m256>(m_mask))));
  }
  void copy_from(float const* mem, vector_aligned_tag) {
    m_value = value_type(_mm256_maskload_ps(
        mem, _mm256_castps_si256(static_cast<__m256>(m_mask))));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      float const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_ps(
        static_cast<__m256>(m_value), mem, static_cast<__m256i>(index),
        static_cast<__m256>(m_mask), 4));
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, basic_simd<float, simd_abi::avx2_fixed_size<8>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<float, simd_abi::avx2_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = basic_simd<float, simd_abi::avx2_fixed_size<8>>(_mm256_blendv_ps(
        static_cast<__m256>(m_value), static_cast<__m256>(x_as_value_type),
        static_cast<__m256>(m_mask)));
  }
};

template <>
class KOKKOS_DEPRECATED const_where_expression<
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = basic_simd<std::int32_t, abi_type>;
  using mask_type  = basic_simd_mask<std::int32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, element_aligned_tag) const {
    _mm_maskstore_epi32(mem, static_cast<__m128i>(m_mask),
                        static_cast<__m128i>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, vector_aligned_tag) const {
    _mm_maskstore_epi32(mem, static_cast<__m128i>(m_mask),
                        static_cast<__m128i>(m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(std::int32_t* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
                      index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED where_expression<
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>,
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
          mask_arg,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value = value_type(_mm_maskload_epi32(mem, static_cast<__m128i>(m_mask)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, vector_aligned_tag) {
    m_value = value_type(_mm_maskload_epi32(mem, static_cast<__m128i>(m_mask)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int32_t const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm_mask_i32gather_epi32(
        static_cast<__m128i>(m_value), mem, static_cast<__m128i>(index),
        static_cast<__m128i>(m_mask), 4));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<U>(x));
    m_value = basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(
        _mm_castps_si128(_mm_blendv_ps(
            _mm_castsi128_ps(static_cast<__m128i>(m_value)),
            _mm_castsi128_ps(static_cast<__m128i>(x_as_value_type)),
            _mm_castsi128_ps(static_cast<__m128i>(m_mask)))));
  }
};

template <>
class KOKKOS_DEPRECATED const_where_expression<
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using value_type = basic_simd<std::int32_t, abi_type>;
  using mask_type  = basic_simd_mask<std::int32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, element_aligned_tag) const {
    _mm256_maskstore_epi32(mem, static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, vector_aligned_tag) const {
    _mm256_maskstore_epi32(mem, static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(std::int32_t* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const&
                      index) const {
    for (std::size_t lane = 0; lane < value_type::size(); ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED where_expression<
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>
    : public const_where_expression<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>,
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
 public:
  where_expression(
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const&
          mask_arg,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value =
        value_type(_mm256_maskload_epi32(mem, static_cast<__m256i>(m_mask)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, vector_aligned_tag) {
    m_value =
        value_type(_mm256_maskload_epi32(mem, static_cast<__m256i>(m_mask)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int32_t const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_epi32(
        static_cast<__m256i>(m_value), mem, static_cast<__m256i>(index),
        static_cast<__m256i>(m_mask), 4));
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(
        _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(static_cast<__m256i>(m_value)),
            _mm256_castsi256_ps(static_cast<__m256i>(x_as_value_type)),
            _mm256_castsi256_ps(static_cast<__m256i>(m_mask)))));
  }
};

template <>
class KOKKOS_DEPRECATED const_where_expression<
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = basic_simd<std::int64_t, abi_type>;
  using mask_type  = basic_simd_mask<std::int64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      std::int64_t* mem, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(std::int64_t* mem,
                                                     vector_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(std::int64_t* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
                      index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED where_expression<
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>,
          basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const&
          mask_arg,
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::int64_t const* mem,
                                                       element_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::int64_t const* mem,
                                                       vector_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int64_t const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_epi64(
        static_cast<__m256i>(m_value), reinterpret_cast<long long const*>(mem),
        static_cast<__m128i>(index), static_cast<__m256i>(m_mask), 8));
  }
  template <class u,
            std::enable_if_t<
                std::is_convertible_v<
                    u, basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(u&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<u>(x));
    m_value = basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(static_cast<__m256i>(m_value)),
            _mm256_castsi256_pd(static_cast<__m256i>(x_as_value_type)),
            _mm256_castsi256_pd(static_cast<__m256i>(m_mask)))));
  }
};

template <>
class KOKKOS_DEPRECATED const_where_expression<
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using value_type = basic_simd<std::uint64_t, abi_type>;
  using mask_type  = basic_simd_mask<std::uint64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      std::uint64_t* mem, element_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(std::uint64_t* mem,
                                                     vector_aligned_tag) const {
    _mm256_maskstore_epi64(reinterpret_cast<long long*>(mem),
                           static_cast<__m256i>(m_mask),
                           static_cast<__m256i>(m_value));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(std::uint64_t* mem,
                  basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const&
                      index) const {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      if (m_mask[lane]) mem[index[lane]] = m_value[lane];
    }
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const& impl_get_value()
      const {
    return m_value;
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const& impl_get_mask() const {
    return m_mask;
  }
};

template <>
class KOKKOS_DEPRECATED where_expression<
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>
    : public const_where_expression<
          basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>,
          basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  where_expression(
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const&
          mask_arg,
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::uint64_t const* mem,
                                                       element_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(std::uint64_t const* mem,
                                                       vector_aligned_tag) {
    m_value = value_type(_mm256_maskload_epi64(
        reinterpret_cast<long long const*>(mem), static_cast<__m256i>(m_mask)));
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint64_t const* mem,
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& index) {
    m_value = value_type(_mm256_mask_i32gather_epi64(
        static_cast<__m256i>(m_value), reinterpret_cast<long long const*>(mem),
        static_cast<__m128i>(index), static_cast<__m256i>(m_mask), 8));
  }
  template <class u,
            std::enable_if_t<
                std::is_convertible_v<
                    u, basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(u&& x) {
    auto const x_as_value_type =
        static_cast<basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>(
            std::forward<u>(x));
    m_value = basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(
        _mm256_castpd_si256(_mm256_blendv_pd(
            _mm256_castsi256_pd(static_cast<__m256i>(m_value)),
            _mm256_castsi256_pd(static_cast<__m256i>(x_as_value_type)),
            _mm256_castsi256_pd(static_cast<__m256i>(m_mask)))));
  }
};
KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
#endif

}  // namespace Experimental
}  // namespace Kokkos

#endif
