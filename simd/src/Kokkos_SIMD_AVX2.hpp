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

template <Impl::simd_size_t N>
class avx2_fixed_size {};

}  // namespace simd_abi

template <>
class basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256d;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_castsi256_pd(_mm256_set1_epi64x(-std::int64_t(value))))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
          return static_cast<double>(other[i]);
        }) {}
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
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_castsi256_pd(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int64_t(
                gen(std::integral_constant<Impl::simd_size_t, 3>())))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm256_movemask_pd(m_value) & (1 << i)) != 0;
  }

  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_pd(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm256_and_pd(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm256_or_pd(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_pd(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_pd(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_pd(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm256_movemask_pd(m_value) == _mm256_movemask_pd(rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m128;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_castsi128_ps(_mm_set1_epi32(-std::int32_t(value))))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
          return static_cast<float>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_castsi128_ps(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int32_t(
                gen(std::integral_constant<Impl::simd_size_t, 3>())))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm_movemask_ps(m_value) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm_andnot_ps(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm_and_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm_or_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm_and_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm_or_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm_xor_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm_movemask_ps(m_value) == _mm_movemask_ps(rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>> {
  using abi_vector_type = __m256;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_castsi256_ps(_mm256_set1_epi32(-std::int32_t(value))))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
          return static_cast<float>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_castsi256_ps(_mm256_setr_epi32(
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 3>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 4>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 5>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 6>())),
            -std::int32_t(
                gen(std::integral_constant<Impl::simd_size_t, 7>())))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm256_movemask_ps(m_value) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_ps(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm256_and_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm256_or_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm256_movemask_ps(m_value) == _mm256_movemask_ps(rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m128i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_set1_epi32(-std::int32_t(value)))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
          return static_cast<std::int32_t>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_setr_epi32(
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 3>()))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm_movemask_ps(_mm_castsi128_ps(m_value)) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm_andnot_si128(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm_and_si128(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm_or_si128(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm_and_si128(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm_or_si128(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm_xor_si128(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm_movemask_ps(_mm_castsi128_ps(m_value)) ==
             _mm_movemask_ps(_mm_castsi128_ps(rhs.m_value)));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi32(-std::int32_t(value)))
#endif
  {
  }
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
          return static_cast<Impl::simd_size_t>(other[i]);
        }) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi32(
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 3>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 4>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 5>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 6>())),
            -std::int32_t(gen(std::integral_constant<Impl::simd_size_t, 7>()))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm256_movemask_ps(_mm256_castsi256_ps(m_value)) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm256_movemask_ps(_mm256_castsi256_ps(m_value)) ==
             _mm256_movemask_ps(_mm256_castsi256_ps(rhs.m_value)));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return impl_operator_eq(rhs).impl_operator_lnot();
  }
};

template <>
class basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(-std::int64_t(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
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
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 3>()))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
             _mm256_movemask_pd(_mm256_castsi256_pd(rhs.m_value)));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

template <>
class basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_mask_base<
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      [[maybe_unused]] value_type value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(-std::int64_t(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      basic_simd_mask<U, abi_type> const& other) noexcept
      : basic_simd_mask([&](Impl::simd_size_t i) {
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
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 0>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 1>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 2>())),
            -std::int64_t(gen(std::integral_constant<Impl::simd_size_t, 3>()))))
#endif
  {
  }

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator implementation_type() const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    return (_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) & (1 << i)) != 0;
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_lnot() const noexcept {
    return impl_operator_bnot();
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(true).m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_lor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_land(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(m_value, rhs.m_value));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_eq(T const& rhs) const noexcept {
    return T(_mm256_movemask_pd(_mm256_castsi256_pd(m_value)) ==
             _mm256_movemask_pd(_mm256_castsi256_pd(rhs.m_value)));
  }
  template <typename T = basic_simd_mask>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
};

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtps_pd(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int32_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int64_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::uint64_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int32_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm_cvtepi32_ps(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int32_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castsi256_ps(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm_castps_si128(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castps_si256(static_cast<__m256>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int32_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<double, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castpd_si256(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::uint64_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int32_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<double, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_castpd_si256(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
    [[maybe_unused]] basic_simd_mask<std::int64_t, abi_type> const&
        other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

template <>
class basic_simd<double, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<double, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256d;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_pd(value_type(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            _mm256_setr_pd(gen(std::integral_constant<Impl::simd_size_t, 0>()),
                           gen(std::integral_constant<Impl::simd_size_t, 1>()),
                           gen(std::integral_constant<Impl::simd_size_t, 2>()),
                           gen(std::integral_constant<Impl::simd_size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm256_load_pd(ptr)
                : _mm256_loadu_pd(ptr))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_maskload_pd(
            ptr, _mm256_castpd_si256(static_cast<__m256d>(mask))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_base<
      basic_simd<double, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    value_type tmp[size()];
    _mm256_storeu_pd(tmp, m_value);
    return tmp[i];
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(_mm256_sub_pd(_mm256_set1_pd(0.0), static_cast<__m256d>(m_value)));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm256_add_pd(static_cast<__m256d>(m_value),
                           static_cast<__m256d>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm256_sub_pd(static_cast<__m256d>(m_value),
                           static_cast<__m256d>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T(_mm256_mul_pd(static_cast<__m256d>(m_value),
                           static_cast<__m256d>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_div(T const& rhs) const noexcept {
    return T(_mm256_div_pd(static_cast<__m256d>(m_value),
                           static_cast<__m256d>(rhs)));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_EQ_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_NEQ_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_GE_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_LE_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_GT_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_pd(static_cast<__m256d>(m_value),
                                   static_cast<__m256d>(rhs), _CMP_LT_OS));
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    copysign, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      __m256d const sign_mask = _mm256_set1_pd(-0.0);
      return (Experimental::basic_simd<
              double, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_xor_pd(_mm256_andnot_pd(sign_mask, static_cast<__m256d>(a)),
                        _mm256_and_pd(sign_mask, static_cast<__m256d>(b)))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      __m256d const sign_mask = _mm256_set1_pd(-0.0);
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_andnot_pd(sign_mask, static_cast<__m256d>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    floor, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_round_pd(static_cast<__m256d>(a),
                              (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    ceil, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_round_pd(static_cast<__m256d>(a),
                              (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    round, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              double, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_round_pd(static_cast<__m256d>(a),
                          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    trunc, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_round_pd(static_cast<__m256d>(a),
                              (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    sqrt, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_sqrt_pd(static_cast<__m256d>(a))));
    })

#ifdef KOKKOS_HAVE_INTEL_SVML

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    cbrt, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_cbrt_pd(static_cast<__m256d>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    exp, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_exp_pd(static_cast<__m256d>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    log, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_log_pd(static_cast<__m256d>(a))));
    })

#endif

KOKKOS_SIMD_IMPL_TERNARY_MATH_FUNCTION(
    fma, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_fmadd_pd(static_cast<__m256d>(a), static_cast<__m256d>(b),
                              static_cast<__m256d>(c))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_max_pd(static_cast<__m256d>(a), static_cast<__m256d>(b))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_min_pd(static_cast<__m256d>(a), static_cast<__m256d>(b))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, double, simd_abi::avx2_fixed_size<4>,
    { return (basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, flag)); })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, double, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, double, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, double, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, double, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, double, simd_abi::avx2_fixed_size<4>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm256_store_pd(ptr, static_cast<__m256d>(simd));
      } else {
        _mm256_storeu_pd(ptr, static_cast<__m256d>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    unchecked, double, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_pd(ptr, _mm256_castpd_si256(static_cast<__m256d>(mask)),
                          static_cast<__m256d>(simd));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    partial, double, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_pd(ptr, _mm256_castpd_si256(static_cast<__m256d>(mask)),
                          static_cast<__m256d>(simd));
    })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, double, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<double, simd_abi::avx2_fixed_size<4>>(
          _mm256_blendv_pd(static_cast<__m256d>(c), static_cast<__m256d>(b),
                           static_cast<__m256d>(a))));
    })

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m128;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value)
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_set1_ps(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<double, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            _mm_setr_ps(gen(std::integral_constant<Impl::simd_size_t, 0>()),
                        gen(std::integral_constant<Impl::simd_size_t, 1>()),
                        gen(std::integral_constant<Impl::simd_size_t, 2>()),
                        gen(std::integral_constant<Impl::simd_size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm_load_ps(ptr)
                : _mm_loadu_ps(ptr))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            _mm_maskload_ps(ptr, _mm_castps_si128(static_cast<__m128>(mask))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const noexcept {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    auto index = _mm_cvtsi32_si128(i);
    auto tmp   = _mm_permutevar_ps(m_value, index);
    return _mm_cvtss_f32(tmp);
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(_mm_sub_ps(_mm_set1_ps(0.0), m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm_add_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm_sub_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return basic_simd(_mm_mul_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_div(T const& rhs) const noexcept {
    return T(_mm_div_ps(m_value, rhs.m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm_cmpeq_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return mask_type(_mm_cmpneq_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return mask_type(_mm_cmpge_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return mask_type(_mm_cmple_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm_cmpgt_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return mask_type(_mm_cmplt_ps(m_value, rhs.m_value));
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    copysign, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      __m128 const sign_mask = _mm_set1_ps(-0.0);
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_xor_ps(_mm_andnot_ps(sign_mask, static_cast<__m128>(a)),
                         _mm_and_ps(sign_mask, static_cast<__m128>(b)))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      __m128 const sign_mask = _mm_set1_ps(-0.0);
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_andnot_ps(sign_mask, static_cast<__m128>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    floor, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_round_ps(static_cast<__m128>(a),
                           (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    ceil, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_round_ps(static_cast<__m128>(a),
                           (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    round, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_round_ps(static_cast<__m128>(a),
                           (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    trunc, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              float, Experimental::simd_abi::avx2_fixed_size<4>>(_mm_round_ps(
          static_cast<__m128>(a), (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    sqrt, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_sqrt_ps(static_cast<__m128>(a))));
    })

#ifdef KOKKOS_HAVE_INTEL_SVML

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    cbrt, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_cbrt_ps(static_cast<__m128>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    exp, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_exp_ps(static_cast<__m128>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    log, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_log_ps(static_cast<__m128>(a))));
    })

#endif

KOKKOS_SIMD_IMPL_TERNARY_MATH_FUNCTION(
    fma, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_fmadd_ps(static_cast<__m128>(a), static_cast<__m128>(b),
                           static_cast<__m128>(c))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_max_ps(static_cast<__m128>(a), static_cast<__m128>(b))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_min_ps(static_cast<__m128>(a), static_cast<__m128>(b))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<4>,
    { return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, flag)); })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, float, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, float, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, float, simd_abi::avx2_fixed_size<4>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm_store_ps(ptr, static_cast<__m128>(simd));
      } else {
        _mm_storeu_ps(ptr, static_cast<__m128>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    unchecked, float, simd_abi::avx2_fixed_size<4>, {
      _mm_maskstore_ps(ptr, _mm_castps_si128(static_cast<__m128>(mask)),
                       static_cast<__m128>(simd));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    partial, float, simd_abi::avx2_fixed_size<4>, {
      _mm_maskstore_ps(ptr, _mm_castps_si128(static_cast<__m128>(mask)),
                       static_cast<__m128>(simd));
    })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, float, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<4>>(
          _mm_blendv_ps(static_cast<__m128>(c), static_cast<__m128>(b),
                        static_cast<__m128>(a))));
    })

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<8>>> {
  using abi_vector_type = __m256;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_ps(value_type(value)))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] G&& gen)
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            _mm256_setr_ps(gen(std::integral_constant<Impl::simd_size_t, 0>()),
                           gen(std::integral_constant<Impl::simd_size_t, 1>()),
                           gen(std::integral_constant<Impl::simd_size_t, 2>()),
                           gen(std::integral_constant<Impl::simd_size_t, 3>()),
                           gen(std::integral_constant<Impl::simd_size_t, 4>()),
                           gen(std::integral_constant<Impl::simd_size_t, 5>()),
                           gen(std::integral_constant<Impl::simd_size_t, 6>()),
                           gen(std::integral_constant<Impl::simd_size_t, 7>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm256_load_ps(ptr)
                : _mm256_loadu_ps(ptr))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_maskload_ps(
            ptr, _mm256_castps_si256(static_cast<__m256>(mask))))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<8>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    auto index = _mm256_set1_epi32(i);
    auto tmp   = _mm256_permutevar8x32_ps(m_value, index);
    return _mm256_cvtss_f32(tmp);
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(_mm256_sub_ps(_mm256_set1_ps(0.0), m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm256_add_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm256_sub_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T(_mm256_mul_ps(m_value, rhs.m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_div(T const& rhs) const noexcept {
    return T(_mm256_div_ps(m_value, rhs.m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_EQ_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_NEQ_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_GE_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_LE_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_GT_OS));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmp_ps(static_cast<__m256>(m_value),
                                   static_cast<__m256>(rhs), _CMP_LT_OS));
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    copysign, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      __m256 const sign_mask = _mm256_set1_ps(-0.0);
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_xor_ps(_mm256_andnot_ps(sign_mask, static_cast<__m256>(a)),
                            _mm256_and_ps(sign_mask, static_cast<__m256>(b)))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      __m256 const sign_mask = _mm256_set1_ps(-0.0);
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_andnot_ps(sign_mask, static_cast<__m256>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    floor, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_round_ps(static_cast<__m256>(a),
                              (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    ceil, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_round_ps(static_cast<__m256>(a),
                              (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    round, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (Experimental::basic_simd<
              float, Experimental::simd_abi::avx2_fixed_size<8>>(
          _mm256_round_ps(static_cast<__m256>(a),
                          (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    trunc, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_round_ps(static_cast<__m256>(a),
                              (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    sqrt, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_sqrt_ps(static_cast<__m256>(a))));
    })

#ifdef __INTEL_COMPILER

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    cbrt, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_cbrt_ps(static_cast<__m256>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    exp, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_exp_ps(static_cast<__m256>(a))));
    })

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    log, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_log_ps(static_cast<__m256>(a))));
    })

#endif

KOKKOS_SIMD_IMPL_TERNARY_MATH_FUNCTION(
    fma, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_fmadd_ps(static_cast<__m256>(a), static_cast<__m256>(b),
                              static_cast<__m256>(c))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_max_ps(static_cast<__m256>(a), static_cast<__m256>(b))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_min_ps(static_cast<__m256>(a), static_cast<__m256>(b))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<8>,
    { return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, flag)); })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, float, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, float, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, float, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, float, simd_abi::avx2_fixed_size<8>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm256_store_ps(ptr, static_cast<__m256>(simd));
      } else {
        _mm256_storeu_ps(ptr, static_cast<__m256>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    unchecked, float, simd_abi::avx2_fixed_size<8>, {
      _mm256_maskstore_ps(ptr, _mm256_castps_si256(static_cast<__m256>(mask)),
                          static_cast<__m256>(simd));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    partial, float, simd_abi::avx2_fixed_size<8>, {
      _mm256_maskstore_ps(ptr, _mm256_castps_si256(static_cast<__m256>(mask)),
                          static_cast<__m256>(simd));
    })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, float, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<float, simd_abi::avx2_fixed_size<8>>(
          _mm256_blendv_ps(static_cast<__m256>(c), static_cast<__m256>(b),
                           static_cast<__m256>(a))));
    })

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m128i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value)
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_set1_epi32(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<double, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            _mm_setr_epi32(gen(std::integral_constant<Impl::simd_size_t, 0>()),
                           gen(std::integral_constant<Impl::simd_size_t, 1>()),
                           gen(std::integral_constant<Impl::simd_size_t, 2>()),
                           gen(std::integral_constant<Impl::simd_size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm_load_si128(reinterpret_cast<__m128i const*>(ptr))
                : _mm_loadu_si128(reinterpret_cast<__m128i const*>(ptr)))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm_maskload_epi32(ptr, static_cast<__m128i>(mask)))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    switch (i) {
      case 0: return _mm_extract_epi32(m_value, 0x0);
      case 1: return _mm_extract_epi32(m_value, 0x1);
      case 2: return _mm_extract_epi32(m_value, 0x2);
      case 3: return _mm_extract_epi32(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
// missing return statement warning with cuda >= 12.9
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1290) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(_mm_sub_epi32(_mm_set1_epi32(0), static_cast<__m128i>(m_value)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm_andnot_si128(m_value, T(~value_type(0)).m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm_add_epi32(static_cast<__m128i>(m_value),
                           static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm_sub_epi32(static_cast<__m128i>(m_value),
                           static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T(_mm_mullo_epi32(static_cast<__m128i>(m_value),
                             static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm_and_si128(static_cast<__m128i>(m_value),
                           static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(
        _mm_or_si128(static_cast<__m128i>(m_value), static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm_xor_si128(static_cast<__m128i>(m_value),
                           static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(T const& rhs) const noexcept {
    return T(_mm_sllv_epi32(static_cast<__m128i>(m_value),
                            static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(T const& rhs) const noexcept {
    return T(_mm_srav_epi32(static_cast<__m128i>(m_value),
                            static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(int rhs) const noexcept {
    return T(_mm_slli_epi32(static_cast<__m128i>(m_value), rhs));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(int rhs) const noexcept {
    return T(_mm_srai_epi32(static_cast<__m128i>(m_value), rhs));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm_cmpeq_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return mask_type(_mm_cmplt_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm_cmpgt_epi32(static_cast<__m128i>(m_value),
                                     static_cast<__m128i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return impl_operator_gt(rhs) || impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return impl_operator_lt(rhs) || impl_operator_eq(rhs);
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      __m128i const rhs = static_cast<__m128i>(a);
      return (
          Experimental::basic_simd<std::int32_t,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_abs_epi32(rhs)));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    floor, double, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_cvtepi32_pd(static_cast<__m128i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    ceil, double, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_cvtepi32_pd(static_cast<__m128i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    round, double, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_cvtepi32_pd(static_cast<__m128i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    trunc, double, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_cvtepi32_pd(static_cast<__m128i>(a))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<std::int32_t,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_max_epi32(static_cast<__m128i>(a), static_cast<__m128i>(b))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<std::int32_t,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm_min_epi32(static_cast<__m128i>(a), static_cast<__m128i>(b))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<4>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm_store_si128(reinterpret_cast<__m128i*>(ptr),
                        static_cast<__m128i>(simd));
      } else {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr),
                         static_cast<__m128i>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(unchecked, std::int32_t,
                                         simd_abi::avx2_fixed_size<4>, {
                                           _mm_maskstore_epi32(
                                               ptr, static_cast<__m128i>(mask),
                                               static_cast<__m128i>(simd));
                                         })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(partial, std::int32_t,
                                         simd_abi::avx2_fixed_size<4>, {
                                           _mm_maskstore_epi32(
                                               ptr, static_cast<__m128i>(mask),
                                               static_cast<__m128i>(simd));
                                         })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(
          _mm_castps_si128(
              _mm_blendv_ps(_mm_castsi128_ps(static_cast<__m128i>(c)),
                            _mm_castsi128_ps(static_cast<__m128i>(b)),
                            _mm_castsi128_ps(static_cast<__m128i>(a))))));
    })

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<8>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi32(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<float, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi32(
            gen(std::integral_constant<Impl::simd_size_t, 0>()),
            gen(std::integral_constant<Impl::simd_size_t, 1>()),
            gen(std::integral_constant<Impl::simd_size_t, 2>()),
            gen(std::integral_constant<Impl::simd_size_t, 3>()),
            gen(std::integral_constant<Impl::simd_size_t, 4>()),
            gen(std::integral_constant<Impl::simd_size_t, 5>()),
            gen(std::integral_constant<Impl::simd_size_t, 6>()),
            gen(std::integral_constant<Impl::simd_size_t, 7>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm256_load_si256(reinterpret_cast<__m256i const*>(ptr))
                : _mm256_loadu_si256(reinterpret_cast<__m256i const*>(ptr)))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_maskload_epi32(ptr, static_cast<__m256i>(mask)))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
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

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(
        _mm256_sub_epi32(_mm256_set1_epi32(0), static_cast<__m256i>(m_value)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(~value_type(0)).m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm256_add_epi32(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm256_sub_epi32(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T(_mm256_mullo_epi32(static_cast<__m256i>(m_value),
                                static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(T const& rhs) const noexcept {
    return T(_mm256_sllv_epi32(static_cast<__m256i>(m_value),
                               static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(T const& rhs) const noexcept {
    return T(_mm256_srav_epi32(static_cast<__m256i>(m_value),
                               static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(Impl::simd_size_t rhs) const noexcept {
    return T(_mm256_slli_epi32(static_cast<__m256i>(m_value), rhs));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(Impl::simd_size_t rhs) const noexcept {
    return T(_mm256_srai_epi32(static_cast<__m256i>(m_value), rhs));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm256_cmpeq_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return impl_operator_gt(rhs) || impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return impl_operator_lt(rhs) || impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmpgt_epi32(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return !impl_operator_ge(rhs);
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      __m256i const rhs = static_cast<__m256i>(a);
      return (
          Experimental::basic_simd<std::int32_t,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_abs_epi32(rhs)));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    floor, float, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_cvtepi32_ps(static_cast<__m256i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    ceil, float, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_cvtepi32_ps(static_cast<__m256i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    round, float, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_cvtepi32_ps(static_cast<__m256i>(a))));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    trunc, float, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (
          Experimental::basic_simd<float,
                                   Experimental::simd_abi::avx2_fixed_size<8>>(
              _mm256_cvtepi32_ps(static_cast<__m256i>(a))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (Experimental::basic_simd<
              std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>(
          _mm256_max_epi32(static_cast<__m256i>(a), static_cast<__m256i>(b))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (Experimental::basic_simd<
              std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>(
          _mm256_min_epi32(static_cast<__m256i>(a), static_cast<__m256i>(b))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      return (
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, std::int32_t, simd_abi::avx2_fixed_size<8>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                           static_cast<__m256i>(simd));
      } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                            static_cast<__m256i>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(unchecked, std::int32_t,
                                         simd_abi::avx2_fixed_size<8>, {
                                           _mm256_maskstore_epi32(
                                               ptr, static_cast<__m256i>(mask),
                                               static_cast<__m256i>(simd));
                                         })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(partial, std::int32_t,
                                         simd_abi::avx2_fixed_size<8>, {
                                           _mm256_maskstore_epi32(
                                               ptr, static_cast<__m256i>(mask),
                                               static_cast<__m256i>(simd));
                                         })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, {
      return (basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(
          _mm256_castps_si256(
              _mm256_blendv_ps(_mm256_castsi256_ps(static_cast<__m256i>(c)),
                               _mm256_castsi256_ps(static_cast<__m256i>(b)),
                               _mm256_castsi256_ps(static_cast<__m256i>(a))))));
    })

template <>
class basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_base<
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

  static_assert(sizeof(long long) == 8);

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(value_type(value)))
#endif
  {
  }
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        })) {}
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::uint64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            gen(std::integral_constant<Impl::simd_size_t, 0>()),
            gen(std::integral_constant<Impl::simd_size_t, 1>()),
            gen(std::integral_constant<Impl::simd_size_t, 2>()),
            gen(std::integral_constant<Impl::simd_size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))
                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                      static_cast<__m256i>(mask)))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class basic_simd_base<
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    switch (i) {
      case 0: return _mm256_extract_epi64(m_value, 0x0);
      case 1: return _mm256_extract_epi64(m_value, 0x1);
      case 2: return _mm256_extract_epi64(m_value, 0x2);
      case 3: return _mm256_extract_epi64(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(
        _mm256_sub_epi64(_mm256_set1_epi64x(0), static_cast<__m256i>(m_value)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(~value_type(0)).m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm256_add_epi64(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm256_sub_epi64(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit signed integers is not available in AVX2
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T([&](Impl::simd_size_t i) { return m_value[i] * rhs[i]; });
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(T const& rhs) const noexcept {
    return T(_mm256_sllv_epi64(static_cast<__m256i>(m_value),
                               static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(T const& rhs) const noexcept {
    return T([&](Impl::simd_size_t i) { return m_value[i] >> rhs[i]; });
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(Impl::simd_size_t rhs) const noexcept {
    return T(_mm256_slli_epi64(static_cast<__m256i>(m_value), rhs));
  }
  // fallback basic_simd shift right arithmetic using generator constructor
  // Shift right arithmetic for 64bit packed ints is not availalbe in AVX2
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(Impl::simd_size_t rhs) const noexcept {
    return T([&](Impl::simd_size_t i) { return m_value[i] >> rhs; });
  }

  // AVX2 only has eq and gt comparisons for int64
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return impl_operator_gt(rhs) || impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return impl_operator_lt(rhs) || impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    return mask_type(_mm256_cmpgt_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return !impl_operator_ge(rhs);
  }
};

}  // namespace Experimental

// Manually computing absolute values, because _mm256_abs_epi64
// is not in AVX2; it's available in AVX512.
KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(
    abs, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<std::int64_t,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              [&](Experimental::Impl::simd_size_t i) {
                return (a[i] < 0) ? -a[i] : a[i];
              }));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    floor, double, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    ceil, double, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    round, double, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    trunc, double, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_blendv_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b),
                             static_cast<__m256i>(b > a))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_blendv_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b),
                             static_cast<__m256i>(b < a))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                     flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                           static_cast<__m256i>(simd));
      } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                            static_cast<__m256i>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    unchecked, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                             static_cast<__m256i>(mask),
                             static_cast<__m256i>(simd));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    partial, std::int64_t, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                             static_cast<__m256i>(mask),
                             static_cast<__m256i>(simd));
    })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(
          _mm256_castpd_si256(
              _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                               _mm256_castsi256_pd(static_cast<__m256i>(b)),
                               _mm256_castsi256_pd(static_cast<__m256i>(a))))));
    })

template <>
class basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_base<
          basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
  using abi_vector_type = __m256i;
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
  using implementation_type = abi_vector_type;
#else
  using implementation_type = Kokkos::Array<char, sizeof(abi_vector_type)>;
#endif
  using base_type = Impl::basic_simd_base<
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;
  alignas(alignof(abi_vector_type)) implementation_type m_value;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd([[maybe_unused]] U&& value) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_set1_epi64x(
            Kokkos::bit_cast<std::int64_t>(value_type(value))))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd(
      implementation_type const& value_in) noexcept
      : m_value(value_in) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd([[maybe_unused]] basic_simd<U, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(basic_simd([&](Impl::simd_size_t i) {
          return static_cast<value_type>(other[i]);
        }))
#endif
  {
  }
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int32_t, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
      basic_simd<std::int64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, Kokkos::Impl::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] G&& gen) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_setr_epi64x(
            gen(std::integral_constant<Impl::simd_size_t, 0>()),
            gen(std::integral_constant<Impl::simd_size_t, 1>()),
            gen(std::integral_constant<Impl::simd_size_t, 2>()),
            gen(std::integral_constant<Impl::simd_size_t, 3>())))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(
            std::is_same_v<FlagType, simd_flags<simd_alignment_vector_aligned>>
                ? _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))
                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)))
#endif
  {
  }
  template <typename FlagType>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      [[maybe_unused]] const value_type* ptr, mask_type const& mask,
      FlagType) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
      : m_value(_mm256_maskload_epi64(reinterpret_cast<long long const*>(ptr),
                                      static_cast<__m256i>(mask)))
#endif
  {
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit operator implementation_type()
      const {
    return m_value;
  }

#ifdef KOKKOS_IMPL_FRIEND_BASE_ACCESS_RESTRICTION
 private:
  friend class basic_simd_base<
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;
#endif

  template <typename T = value_type>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_subscript_operator(Impl::simd_size_t i) const {
    switch (i) {
      case 0: return _mm256_extract_epi64(m_value, 0x0);
      case 1: return _mm256_extract_epi64(m_value, 0x1);
      case 2: return _mm256_extract_epi64(m_value, 0x2);
      case 3: return _mm256_extract_epi64(m_value, 0x3);
      default: Kokkos::abort("Index out of bound"); break;
    }
#if defined(KOKKOS_COMPILER_NVCC) && (KOKKOS_COMPILER_NVCC >= 1290) && \
    defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
    return value_type{};
#endif
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_neg() const noexcept {
    return T(static_cast<__m256i>(m_value));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T impl_operator_bnot() const noexcept {
    return T(_mm256_andnot_si256(m_value, T(~value_type(0)).m_value));
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_plus(T const& rhs) const noexcept {
    return T(_mm256_add_epi64(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_minus(T const& rhs) const noexcept {
    return T(_mm256_sub_epi64(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  // fallback basic_simd multiplication using generator constructor
  // multiplying vectors of 64-bit unsigned integers is not available in AVX2
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_mul(T const& rhs) const noexcept {
    return T([&](Impl::simd_size_t i) { return m_value[i] * rhs[i]; });
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_band(T const& rhs) const noexcept {
    return T(_mm256_and_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_bor(T const& rhs) const noexcept {
    return T(_mm256_or_si256(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_xor(T const& rhs) const noexcept {
    return T(_mm256_xor_si256(static_cast<__m256i>(m_value),
                              static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(T const& rhs) const noexcept {
    return _mm256_sllv_epi64(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(T const& rhs) const noexcept {
    return _mm256_srlv_epi64(static_cast<__m256i>(m_value),
                             static_cast<__m256i>(rhs));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sll(Impl::simd_size_t rhs) const noexcept {
    return _mm256_slli_epi64(static_cast<__m256i>(m_value), rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION T
  impl_operator_sra(Impl::simd_size_t rhs) const noexcept {
    return _mm256_srli_epi64(static_cast<__m256i>(m_value), rhs);
  }

  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_eq(T const& rhs) const noexcept {
    return mask_type(_mm256_cmpeq_epi64(static_cast<__m256i>(m_value),
                                        static_cast<__m256i>(rhs)));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ne(T const& rhs) const noexcept {
    return !impl_operator_eq(rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_ge(T const& rhs) const noexcept {
    return !(m_value < rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_le(T const& rhs) const noexcept {
    return !(m_value > rhs);
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_gt(T const& rhs) const noexcept {
    // We use the following bit trick to compute the unsigned comparison from
    // the signed values since there is no intrinsic for unsigned int comparison
    // in AVX2: (a < 0) ^ (b < 0) ^ (a > b)
    // If a and b have the same sign, the signed and unsigned comparison will
    // give the same result. In this case:
    //  (a < 0) == (b < 0) => (a < 0) ^ (b < 0) = 0, and 0 ^ (a > b) == (a > b)
    // If they have different signs, the signed and unsigned comparisons will
    // give opposite results (negative values are higher than positive values
    // when interpreting them as unsigned). In that case:
    //  (a < 0) != (b < 0) => (a < 0) ^ (b < 0) = 1, and 1 ^ (a > b) == !(a > b)
    basic_simd<std::int64_t, abi_type> signed_lhs{
        static_cast<__m256i>(m_value)};
    basic_simd<std::int64_t, abi_type> signed_rhs{static_cast<__m256i>(rhs)};
    return static_cast<mask_type>((signed_lhs < 0) ^ (signed_rhs < 0) ^
                                  (signed_lhs > signed_rhs));
  }
  template <typename T = basic_simd>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type
  impl_operator_lt(T const& rhs) const noexcept {
    return rhs > m_value;
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_UNARY_MATH_FUNCTION(abs, std::uint64_t,
                                     Experimental::simd_abi::avx2_fixed_size<4>,
                                     { return a; })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    floor, double, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    ceil, double, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    round, double, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_ROUNDING_FUNCTION(
    trunc, double, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (
          Experimental::basic_simd<double,
                                   Experimental::simd_abi::avx2_fixed_size<4>>(
              _mm256_setr_pd(a[0], a[1], a[2], a[3])));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    max, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_blendv_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b),
                             static_cast<__m256i>(b > a))));
    })

KOKKOS_SIMD_IMPL_BINARY_MATH_FUNCTION(
    min, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (Experimental::basic_simd<
              std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>(
          _mm256_blendv_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b),
                             static_cast<__m256i>(b < a))));
    })

namespace Experimental {

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_LOAD(
    unchecked, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      return (
          basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    unchecked, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                      flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    unchecked, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                      flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_LOAD(
    partial, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                      flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_NATIVE_LOAD(
    partial, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                      flag));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_NATIVE_STORE(
    unchecked, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      if constexpr (std::is_same_v<decltype(flag),
                                   simd_flags<simd_alignment_vector_aligned>>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(ptr),
                           static_cast<__m256i>(simd));
      } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr),
                            static_cast<__m256i>(simd));
      }
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    unchecked, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                             static_cast<__m256i>(mask),
                             static_cast<__m256i>(simd));
    })

KOKKOS_SIMD_IMPL_LOAD_STORE_MASKED_STORE(
    partial, std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      _mm256_maskstore_epi64(reinterpret_cast<long long int*>(ptr),
                             static_cast<__m256i>(mask),
                             static_cast<__m256i>(simd));
    })

KOKKOS_SIMD_IMPL_MASKED_BINARY_MATH_FUNCTION(
    condition, std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, {
      return (basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(
          _mm256_castpd_si256(
              _mm256_blendv_pd(_mm256_castsi256_pd(static_cast<__m256i>(c)),
                               _mm256_castsi256_pd(static_cast<__m256i>(b)),
                               _mm256_castsi256_pd(static_cast<__m256i>(a))))));
    })

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtps_pd(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<double, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtpd_ps(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm_cvtepi32_ps(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<8>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_ps(static_cast<__m256i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm_cvtps_epi32(static_cast<__m128>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<double, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtpd_epi32(static_cast<__m256d>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd(
    [[maybe_unused]] basic_simd<float, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtps_epi32(static_cast<__m256>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::uint64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int32_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(_mm256_cvtepi32_epi64(static_cast<__m128i>(other)))
#endif
{
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
    [[maybe_unused]] basic_simd<std::int64_t, abi_type> const& other) noexcept
#ifndef KOKKOS_SIMD_IMPL_DEVICE_SIMD
    : m_value(static_cast<__m256i>(other))
#endif
{
}

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    double, simd_abi::avx2_fixed_size<4>, {
      __m128i idx = static_cast<__m128i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>{indices});
      return V(_mm256_i32gather_pd(Impl::Ranges::data(in), idx, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    double, simd_abi::avx2_fixed_size<4>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m128i idx =
          static_cast<__m128i>(basic_simd<std::int32_t, abi_type>{indices});
      __m256d mmask =
          static_cast<__m256d>(basic_simd_mask<double, abi_type>{mask});
      return V(_mm256_mask_i32gather_pd(_mm256_set1_pd(value_type{}),
                                        Impl::Ranges::data(in), idx, mmask, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    double, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    double, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    double, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    double, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    double, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    double, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    float, simd_abi::avx2_fixed_size<4>, {
      __m128i idx = static_cast<__m128i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>{indices});
      return V(_mm_i32gather_ps(Impl::Ranges::data(in), idx, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    float, simd_abi::avx2_fixed_size<4>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m128i idx =
          static_cast<__m128i>(basic_simd<std::int32_t, abi_type>{indices});
      __m128 mmask =
          static_cast<__m128>(basic_simd_mask<float, abi_type>{mask});
      return V(_mm_mask_i32gather_ps(_mm_set1_ps(value_type{}),
                                     Impl::Ranges::data(in), idx, mmask, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    float, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    float, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    float, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    float, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    float, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    float, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    float, simd_abi::avx2_fixed_size<8>, {
      __m256i idx = static_cast<__m256i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>{indices});
      return V(_mm256_i32gather_ps(Impl::Ranges::data(in), idx, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    float, simd_abi::avx2_fixed_size<8>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m256i idx =
          static_cast<__m256i>(basic_simd<std::int32_t, abi_type>{indices});
      __m256 mmask =
          static_cast<__m256>(basic_simd_mask<float, abi_type>{mask});
      return V(_mm256_mask_i32gather_ps(_mm256_set1_ps(value_type{}),
                                        Impl::Ranges::data(in), idx, mmask, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    float, simd_abi::avx2_fixed_size<8>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    float, simd_abi::avx2_fixed_size<8>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    float, simd_abi::avx2_fixed_size<8>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    float, simd_abi::avx2_fixed_size<8>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    float, simd_abi::avx2_fixed_size<8>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    float, simd_abi::avx2_fixed_size<8>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    std::int32_t, simd_abi::avx2_fixed_size<4>, {
      __m128i idx = static_cast<__m128i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>{indices});
      return V(_mm_i32gather_epi32(Impl::Ranges::data(in), idx, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<4>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m128i idx =
          static_cast<__m128i>(basic_simd<std::int32_t, abi_type>{indices});
      __m128i mmask =
          static_cast<__m128i>(basic_simd_mask<std::int32_t, abi_type>{mask});
      return V(_mm_mask_i32gather_epi32(_mm_set1_epi32(value_type{}),
                                        Impl::Ranges::data(in), idx, mmask, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    std::int32_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    std::int32_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    std::int32_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    std::int32_t, simd_abi::avx2_fixed_size<8>, {
      __m256i idx = static_cast<__m256i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>{indices});
      return V(_mm256_i32gather_epi32(Impl::Ranges::data(in), idx, 4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<8>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m256i idx =
          static_cast<__m256i>(basic_simd<std::int32_t, abi_type>{indices});
      __m256i mmask =
          static_cast<__m256i>(basic_simd_mask<std::int32_t, abi_type>{mask});
      return V(_mm256_mask_i32gather_epi32(_mm256_set1_epi32(value_type{}),
                                           Impl::Ranges::data(in), idx, mmask,
                                           4));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    std::int32_t, simd_abi::avx2_fixed_size<8>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<8>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    std::int32_t, simd_abi::avx2_fixed_size<8>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<8>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    std::int32_t, simd_abi::avx2_fixed_size<8>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    std::int32_t, simd_abi::avx2_fixed_size<8>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    std::int64_t, simd_abi::avx2_fixed_size<4>, {
      __m128i idx = static_cast<__m128i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>{indices});
      return V(_mm256_i32gather_epi64(
          reinterpret_cast<long long const*>(Impl::Ranges::data(in)), idx, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    std::int64_t, simd_abi::avx2_fixed_size<4>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m128i idx =
          static_cast<__m128i>(basic_simd<std::int32_t, abi_type>{indices});
      __m256i mmask =
          static_cast<__m256i>(basic_simd_mask<std::int64_t, abi_type>{mask});
      return V(_mm256_mask_i32gather_epi64(
          _mm256_set1_epi64x(value_type{}),
          reinterpret_cast<long long const*>(Impl::Ranges::data(in)), idx,
          mmask, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    std::int64_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    std::int64_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    std::int64_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    std::int64_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    std::int64_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    std::int64_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM(
    std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      __m128i idx = static_cast<__m128i>(
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>{indices});
      return V(_mm256_i32gather_epi64(
          reinterpret_cast<long long const*>(Impl::Ranges::data(in)), idx, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_GATHER_FROM_WITH_MASK(
    std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      using value_type = typename V::value_type;
      using abi_type   = typename V::abi_type;
      __m128i idx =
          static_cast<__m128i>(basic_simd<std::int32_t, abi_type>{indices});
      __m256i mmask =
          static_cast<__m256i>(basic_simd_mask<std::int64_t, abi_type>{mask});
      return V(_mm256_mask_i32gather_epi64(
          _mm256_set1_epi64x(value_type{}),
          reinterpret_cast<long long const*>(Impl::Ranges::data(in)), idx,
          mmask, 8));
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM(
    std::uint64_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_GATHER_FROM_WITH_MASK(
    std::uint64_t, simd_abi::avx2_fixed_size<4>,
    { return unchecked_gather_from<V>(in, mask, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO(
    std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_UNCHECKED_SCATTER_TO_WITH_MASK(
    std::uint64_t, simd_abi::avx2_fixed_size<4>, {
      for (Impl::simd_size_t lane = 0; lane < v.size(); ++lane) {
        if (mask[lane]) out[indices[lane]] = v[lane];
      }
    })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO(
    std::uint64_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, indices, flag); })

KOKKOS_SIMD_IMPL_MEMORY_PERMUTE_PARTIAL_SCATTER_TO_WITH_MASK(
    std::uint64_t, simd_abi::avx2_fixed_size<4>,
    { unchecked_scatter_to<V>(v, out, mask, indices, flag); })

}  // namespace Experimental
}  // namespace Kokkos

#endif
