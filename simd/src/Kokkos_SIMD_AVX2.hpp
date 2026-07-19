// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_AVX2_HPP
#define KOKKOS_SIMD_AVX2_HPP

#include <functional>
#include <type_traits>

#include <Kokkos_SIMD_Common.hpp>
#include <impl/Kokkos_SIMD_Impl_AVX2.hpp>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_AVX2.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

namespace Kokkos {

namespace Experimental {

template <>
class basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>
    : public Impl::basic_simd_mask_base<
          basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>; 

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_mask_native_ops<double, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  // template <class U>
  // KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<double>(other[i]);
  //       }) {}
  // KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
  //     basic_simd_mask<float, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::int64_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, double>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_mask_native_ops<float, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       }) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, float>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;

  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops = Impl::simd_mask_native_ops<float, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, float>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       }) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_mask_native_ops<std::int32_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  // template <class U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<std::int32_t>(other[i]);
  //       }) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, std::int32_t>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;

  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<8>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops = Impl::simd_mask_native_ops<std::int32_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, std::int32_t>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;
  // template <class U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<Impl::simd_size_t>(other[i]);
  //       }) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<float, abi_type> const& other) noexcept;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> 
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_mask_native_ops<std::int64_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  // template <class U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<std::int64_t>(other[i]);
  //       }) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, std::int64_t>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;

  // KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
  //     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<double, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept;
  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

template <>
class basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>
  : public Impl::basic_simd_mask_base<
          basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = bool;
  using abi_type   = simd_abi::avx2_fixed_size<4>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_mask_base<basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_mask_native_ops<std::uint64_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;
  
  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, std::uint64_t>)
  basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept;

  // template <class U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<U, abi_type> const& other) noexcept
  //     : basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<std::uint64_t>(other[i]);
  //       }) {}
  // KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask(
  //     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<double, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
  //     basic_simd_mask<std::int64_t, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<float, abi_type> const& other) noexcept
//     : m_value(_mm256_cvtps_pd(static_cast<__m128>(other))) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
//     : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other))) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int64_t, abi_type> const& other) noexcept
//     : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other))) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept
//     : m_value(_mm256_castsi256_pd(static_cast<__m256i>(other))) {}

// TODO: this could be converted to use initializer_lsit + helper function if compressed to be used for all types (as opposed to just double)
template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<double>(other[i]);
  //       });
  // }
}

// TODO
// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm_cvtepi32_ps(static_cast<__m128i>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
//     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_castsi256_ps(static_cast<__m256i>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<8>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<float, abi_type> const& other) noexcept
//     : m_value(/*_mm_castps_si128(static_cast<__m128>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd_mask(
//     basic_simd_mask<float, abi_type> const& other) noexcept
//     : m_value(/*_mm256_castps_si256(static_cast<__m256>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<8>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtepi32_epi64(static_cast<__m128i>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<double, abi_type> const& other) noexcept
//     : m_value(/*_mm256_castpd_si256(static_cast<__m256d>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::uint64_t, abi_type> const& other) noexcept
//     : m_value(/*static_cast<__m256i>(other)*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtepi32_epi64(static_cast<__m128i>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<double, abi_type> const& other) noexcept
//     : m_value(/*_mm256_castpd_si256(static_cast<__m256d>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd_mask(
//     basic_simd_mask<std::int64_t, abi_type> const& other) noexcept
//     : m_value(/*static_cast<__m256i>(other)*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>::
basic_simd_mask(basic_simd_mask<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd_mask([&](Impl::simd_size_t i) {
  //         return static_cast<float>(other[i]);
  //       });
  // }
}

template <>
class basic_simd<double, simd_abi::avx2_fixed_size<4>> 
    : public Impl::basic_simd_base<
          basic_simd<double, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = double;
  using abi_type   = simd_abi::avx2_fixed_size<4>; 
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<double, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  // KOKKOS_FORCEINLINE_FUNCTION basic_simd(
  //     basic_simd<float, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::int32_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}
  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f))
  {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

}  // namespace Experimental

// TODO: these can evfentually just call be calling impl_ops:: ... (templated on T and abi)
KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
copysign(Experimental::basic_simd<
             double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
         Experimental::basic_simd<
             double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::copysign(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::abs(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
sqrt(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::sqrt(a));
}

#ifdef KOKKOS_HAVE_INTEL_SVML

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
cbrt(Experimental::basic_simd<
     double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::cbrt(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
exp(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::exp(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
log(Experimental::basic_simd<
    double, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::log(a));
}

#endif

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
fma(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::fma(a, b, c));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
max(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
min(Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        double, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

// TODO: these could prob be consolidated too (T, abi)
template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const double* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<double, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const double* ptr,
        basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<double, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& simd, double* ptr,
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<double, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<double, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<double, simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<4>>
      : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<float, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
      : m_value(impl_ops::set1(value)) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<double, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}
  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}
  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

}  // namespace Experimental

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
copysign(Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
         Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::copysign(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> abs(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::abs(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
sqrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::sqrt(a));
}

#ifdef KOKKOS_HAVE_INTEL_SVML

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>
cbrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::cbrt(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> exp(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::exp(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> log(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<double, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::log(a));
}

#endif

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> fma(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::fma(a, b, c));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> max(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>> min(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const float* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<4>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<float, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<float, simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<float, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<float, simd_abi::avx2_fixed_size<8>>> {
 public:
  using value_type = float;
  using abi_type   = simd_abi::avx2_fixed_size<8>; 
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<float, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
 
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::int32_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f))
  {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const {
    return m_value;
  }
};

}  // namespace Experimental

// TODO make these into macro functions
KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
copysign(Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
         Experimental::basic_simd<
             float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::copysign(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> abs(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::abs(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
floor(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
ceil(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
round(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
trunc(Experimental::basic_simd<
      float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
sqrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::sqrt(a));
}

#ifdef __INTEL_COMPILER

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
cbrt(Experimental::basic_simd<
     float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::cbrt(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> exp(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::exp(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> log(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::exp(a));
}

#endif

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> fma(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& c) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::fma(a, b, c));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> max(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>> min(
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        float, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<float, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(const float* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<float, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const float* ptr,
        basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<float, simd_abi::avx2_fixed_size<8>>(ptr, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& simd, float* ptr,
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<float, simd_abi::avx2_fixed_size<8>> condition(
    basic_simd_mask<float, simd_abi::avx2_fixed_size<8>> const& a,
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& b,
    basic_simd<float, simd_abi::avx2_fixed_size<8>> const& c) {
  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
      : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
      : m_value(impl_ops::set1(value)) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<float, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<double, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const {
    return m_value;
  }
};

}  // namespace Experimental

// TODO
KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::abs(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int32_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
max(Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int32_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
min(Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::int32_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using simd_type = basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    : public Impl::basic_simd_base<
          basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>> {

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::avx2_fixed_size<8>; 
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const {
    return m_value;
  }
};

}  // namespace Experimental

KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>
abs(Experimental::basic_simd<
    std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::abs(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
floor(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
ceil(Experimental::basic_simd<
     std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
round(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>
trunc(Experimental::basic_simd<
      std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<float, Experimental::simd_abi::avx2_fixed_size<8>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int32_t,
                         Experimental::simd_abi::avx2_fixed_size<8>>
max(Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int32_t,
                         Experimental::simd_abi::avx2_fixed_size<8>>
min(Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& a,
    Experimental::basic_simd<
        std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int32_t, Experimental::simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<8>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(const std::int32_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_unchecked_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>
    simd_partial_load(
        const std::int32_t* ptr,
        basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr, simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& simd,
    std::int32_t* ptr,
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> condition(
    basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>> const& a,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& b,
    basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>> const& c) {
  using impl_ops = Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>, Impl::simd_backend_t>;
  using simd_type = basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
  : public Impl::basic_simd_base<
          basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}

  // KOKKOS_FORCEINLINE_FUNCTION basic_simd(
  //     basic_simd<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::uint64_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}
  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const {
    return m_value;
  }
};

}  // namespace Experimental

// Manually computing absolute values, because _mm256_abs_epi64
// is not in AVX2; it's available in AVX512.
KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::abs(a));
}

// TODO
KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int64_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
max(Experimental::basic_simd<
        std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::int64_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
min(Experimental::basic_simd<
        std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::int64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::int64_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION
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
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>
    simd_partial_load(
        const std::int64_t* ptr,
        basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
        simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr,
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::int64_t* ptr,
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using simd_type = basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::condition(a, b, c));
}

template <>
class basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
  : public Impl::basic_simd_base<
          basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>> {
 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::avx2_fixed_size<4>;
  using mask_type  = basic_simd_mask<value_type, abi_type>;

#ifdef KOKKOS_IMPL_BASE_FRIEND_FN_DERIVED_ACCESS_RESTRICTION_FIXED
 private:
#endif
  friend class Impl::basic_simd_base<basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using impl_vector_type = typename impl_ops::vector_type;

  impl_vector_type m_value;

 public:
  static constexpr std::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_FORCEINLINE_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  // template <typename U>
  // KOKKOS_FORCEINLINE_FUNCTION explicit(
  //     Impl::needs_explicit_conversion_v<U, value_type>)
  //     basic_simd(basic_simd<U, abi_type> const& other) noexcept
  //     : m_value(basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       })) {}
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::int32_t, abi_type> const& other) noexcept;
  // KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd(
  //     basic_simd<std::int64_t, abi_type> const& other) noexcept;

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(Impl::needs_explicit_conversion_v<U, value_type>)
  basic_simd(basic_simd<U, abi_type> const& other) noexcept;

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask, simd_flags<Flags...> f = {}) noexcept
    : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask), f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const {
    return m_value;
  }
};

}  // namespace Experimental

KOKKOS_FORCEINLINE_FUNCTION Experimental::basic_simd<
    std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>
abs(Experimental::basic_simd<
    std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::abs(a));
}

// TODO
KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
floor(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::floor(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
ceil(Experimental::basic_simd<
     std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::ceil(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
round(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::round(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>
trunc(Experimental::basic_simd<
      std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using rounded_type = Experimental::basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  return rounded_type(impl_ops::trunc(a));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::uint64_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
max(Experimental::basic_simd<
        std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::max(a, b));
}

KOKKOS_FORCEINLINE_FUNCTION
Experimental::basic_simd<std::uint64_t,
                         Experimental::simd_abi::avx2_fixed_size<4>>
min(Experimental::basic_simd<
        std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& a,
    Experimental::basic_simd<
        std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>> const& b) {
  using impl_ops = Experimental::Impl::simd_native_ops<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>, Experimental::Impl::simd_backend_t>;
  using simd_type = Experimental::basic_simd<std::uint64_t, Experimental::simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::min(a, b));
}

namespace Experimental {

template <typename SimdType, typename... Flags>
  requires std::same_as<typename SimdType::abi_type,
                        simd_abi::avx2_fixed_size<4>>
KOKKOS_FORCEINLINE_FUNCTION
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>
    simd_unchecked_load(const std::uint64_t* ptr,
                        simd_flags<Flags...> flag = simd_flag_default) {
                          return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, flag);
                        }

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
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
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_unchecked_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
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
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::uint64_t,
                                                 simd_abi::avx2_fixed_size<4>>
simd_partial_load(
    const std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = simd_flag_default) {
  return basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>(ptr, mask,
                                                                 flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::store(ptr, simd, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_unchecked_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

template <typename... Flags>
KOKKOS_FORCEINLINE_FUNCTION void simd_partial_store(
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& simd,
    std::uint64_t* ptr,
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& mask,
    simd_flags<Flags...> flag = {}) {
  using impl_ops = Impl::simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;

  impl_ops::masked_store(ptr, simd, mask, flag);
}

KOKKOS_FORCEINLINE_FUNCTION
basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> condition(
    basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& a,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& b,
    basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>> const& c) {
  using impl_ops = Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  using simd_type = basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>;

  return simd_type(impl_ops::condition(a, b, c));
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<float, abi_type> const& other) noexcept
//     : m_value(_mm256_cvtps_pd(static_cast<__m128>(other))) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<double, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::int32_t, abi_type> const& other) noexcept
//     : m_value(_mm256_cvtepi32_pd(static_cast<__m128i>(other))) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<double, simd_abi::avx2_fixed_size<4>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}

// TODO
// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<double, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtpd_ps(static_cast<__m256d>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<float, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm_cvtepi32_ps(static_cast<__m128i>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<float, simd_abi::avx2_fixed_size<4>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<float, simd_abi::avx2_fixed_size<8>>::basic_simd(
//     basic_simd<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtepi32_ps(static_cast<__m256i>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<float, simd_abi::avx2_fixed_size<8>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<8>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<float, abi_type> const& other) noexcept
//     : m_value(/*_mm_cvtps_epi32(static_cast<__m128>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<double, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtpd_epi32(static_cast<__m256d>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}


// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>::basic_simd(
//     basic_simd<float, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtps_epi32(static_cast<__m256>(other))*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<8>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}


// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtepi32_epi64(static_cast<__m128i>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::uint64_t, abi_type> const& other) noexcept
//     : m_value(/*static_cast<__m256i>(other)*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::int32_t, abi_type> const& other) noexcept
//     : m_value(/*_mm256_cvtepi32_epi64(static_cast<__m128i>(other))*/) {}

// KOKKOS_FORCEINLINE_FUNCTION
// basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::basic_simd(
//     basic_simd<std::int64_t, abi_type> const& other) noexcept
//     : m_value(/*static_cast<__m256i>(other)*/) {}

template <typename U>
KOKKOS_FORCEINLINE_FUNCTION basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>::
basic_simd(basic_simd<U, simd_abi::avx2_fixed_size<4>> const& other) noexcept
  : m_value(impl_ops::convert_from<U>(other)) {
  // if constexpr (requires { impl_ops::convert_from(other); }) {
  //   m_value = impl_ops::convert_from(other);
  // } else {
  //   m_value = basic_simd([&](Impl::simd_size_t i) {
  //         return static_cast<value_type>(other[i]);
  //       });
  // }
}


// TODO: uncomment/enable when all avx2 simd specializations are implmeneted
//       since all functions are essentially call-through now, should be all templatable to
//         one function
// template <typename T, typename Abi>
// KOKKOS_FORCEINLINE_FUNCTION
// Experimental::basic_simd<T, Abi>
// copysign(Experimental::basic_simd<T, Abi> const& a,
//          Experimental::basic_simd<T, Abi> const& b) {
//   using impl_ops = Experimental::Impl::simd_native_ops<T, Abi, Impl::simd_backend_t>;
//   using simd_type = Experimental::basic_simd<T, Abi>;

//   return simd_type(impl_ops::copysign(a, b));
// }



// TODO
// these prob could also be consolidated into one function each, call-throughs
  // template <Impl::SimdVecType V, Impl::Ranges::contiguous_range R,        
  //           Impl::SimdIntegral I, typename... Flags>                      
  //   requires Impl::Ranges::sized_range<R> &&                              
  //            std::same_as<V, basic_simd<double, simd_abi::avx2_fixed_size<4>>>             
  // KOKKOS_FORCEINLINE_FUNCTION constexpr V unchecked_gather_from( 
  //     R&& in, const I& indices,                                           
  //     simd_flags<Flags...> flag = simd_flag_default) {   

  //   using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>, Impl::simd_backend_t>;
  //   // using simd_type = basic_simd<double, Experimental::simd_abi::avx2_fixed_size<4>>;

  //   return V(impl_ops::unchecked_gather_from(in, static_cast<I::impl_vector_type>(indices), flag));
  // }

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
