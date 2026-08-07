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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<double, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<double, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, double>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<float, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, float>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<float, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<float, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, float>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<std::int32_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, std::int32_t>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<std::int32_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, std::int32_t>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<std::int64_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, std::int64_t>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_mask_base<
      basic_simd_mask<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_mask_native_ops<std::uint64_t, abi_type, Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd_mask() noexcept = default;

  KOKKOS_FORCEINLINE_FUNCTION explicit basic_simd_mask(
      value_type value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd_mask(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, std::uint64_t>)
      basic_simd_mask(basic_simd_mask<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
  friend class Impl::basic_simd_base<
      basic_simd<double, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<double, simd_abi::avx2_fixed_size<4>,
                                         Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}
  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<4>,
                                         Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type()
      const noexcept {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<float, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops = Impl::simd_native_ops<float, simd_abi::avx2_fixed_size<8>,
                                         Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;

  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type() const {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<4>,
                            Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value)
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type() const {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<std::int32_t, simd_abi::avx2_fixed_size<8>>>;

  using impl_ops =
      Impl::simd_native_ops<std::int32_t, simd_abi::avx2_fixed_size<8>,
                            Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 8> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

  template <class G>
    requires Impl::InvocableWithReturnType<
        G, value_type, std::integral_constant<Impl::simd_size_t, 0>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(G&& gen) noexcept
      : m_value(impl_ops::gen(gen)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::load(ptr, f)) {}

  template <typename... Flags>
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type() const {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<std::int64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_native_ops<std::int64_t, simd_abi::avx2_fixed_size<4>,
                            Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  KOKKOS_FORCEINLINE_FUNCTION constexpr explicit basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type() const {
    return m_value;
  }
};

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
  friend class Impl::basic_simd_base<
      basic_simd<std::uint64_t, simd_abi::avx2_fixed_size<4>>>;

  using impl_ops =
      Impl::simd_native_ops<std::uint64_t, simd_abi::avx2_fixed_size<4>,
                            Impl::simd_backend_t>;
  using impl_vector_type      = typename impl_ops::vector_type;
  using impl_host_vector_type = typename impl_ops::host_vector_type;

  alignas(alignof(impl_host_vector_type)) impl_vector_type m_value;

 public:
  static constexpr Kokkos::Impl::integral_constant<Impl::simd_size_t, 4> size{};

  KOKKOS_DEFAULTED_FUNCTION basic_simd() noexcept = default;
  template <class U>
    requires std::convertible_to<U, value_type>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  KOKKOS_FORCEINLINE_FUNCTION basic_simd(U&& value) noexcept
      : m_value(impl_ops::set1(value)) {}
  KOKKOS_FORCEINLINE_FUNCTION constexpr basic_simd(
      impl_vector_type const& value_in) noexcept
      : m_value(value_in) {}

  template <typename U>
  KOKKOS_FORCEINLINE_FUNCTION explicit(
      Impl::needs_explicit_conversion_v<U, value_type>)
      basic_simd(basic_simd<U, abi_type> const& other) noexcept
      : m_value(impl_ops::convert_from<U>(other)) {}

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
      const value_type* ptr, mask_type const& mask,
      simd_flags<Flags...> f = {}) noexcept
      : m_value(impl_ops::masked_load(ptr, static_cast<impl_vector_type>(mask),
                                      f)) {}

  KOKKOS_FORCEINLINE_FUNCTION constexpr operator impl_vector_type() const {
    return m_value;
  }
};

}  // namespace Experimental

KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(double,
                                Experimental::simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(float,
                                Experimental::simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(float,
                                Experimental::simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(std::int32_t,
                                Experimental::simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(std::int32_t,
                                Experimental::simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(std::int64_t,
                                Experimental::simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MATH_FN(std::uint64_t,
                                Experimental::simd_abi::avx2_fixed_size<4>)

namespace Experimental {

KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(double, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(float, simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(float, simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(std::int32_t,
                                           simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(std::int32_t,
                                           simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(std::int64_t,
                                           simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_LOAD_STORE_FN_AVX2(std::uint64_t,
                                           simd_abi::avx2_fixed_size<4>)

KOKKOS_SIMD_IMPL_DEFINE_MASKED_FN_AVX2(condition)

KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(double, double,
                                               simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(float, float,
                                               simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(float, float,
                                               simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(std::int32_t, std::int32_t,
                                               simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(std::int32_t, std::int32_t,
                                               simd_abi::avx2_fixed_size<8>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(std::int64_t, std::int64_t,
                                               simd_abi::avx2_fixed_size<4>)
KOKKOS_SIMD_IMPL_DEFINE_MEMORY_PERMUTE_FN_AVX2(std::uint64_t, std::int64_t,
                                               simd_abi::avx2_fixed_size<4>)

}  // namespace Experimental
}  // namespace Kokkos

#endif
