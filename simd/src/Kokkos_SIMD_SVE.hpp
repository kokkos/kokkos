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

#ifndef KOKKOS_SIMD_sve_HPP
#define KOKKOS_SIMD_sve_HPP

#include <functional>
#include <type_traits>

#include <Kokkos_SIMD_Common.hpp>

#include <arm_sve.h>

#ifdef KOKKOS_SIMD_COMMON_MATH_HPP
#error \
    "Kokkos_SIMD_SVE.hpp must be included before Kokkos_SIMD_Common_Math.hpp!"
#endif

#if __ARM_FEATURE_SVE_BITS == 512
typedef svint32_t vls_int32_t __attribute__((arm_sve_vector_bits(512)));
typedef svuint32_t vls_uint32_t __attribute__((arm_sve_vector_bits(512)));
typedef svfloat32_t vls_float32_t __attribute__((arm_sve_vector_bits(512)));
typedef svint64_t vls_int64_t __attribute__((arm_sve_vector_bits(512)));
typedef svuint64_t vls_uint64_t __attribute__((arm_sve_vector_bits(512)));
typedef svfloat64_t vls_float64_t __attribute__((arm_sve_vector_bits(512)));
typedef svbool_t vls_bool_t __attribute__((arm_sve_vector_bits(512)));
#else
#error \
    "Kokkos_SIMD_SVE.hpp: Only vector length of 512 is supported at this time"
#endif

namespace Kokkos {

namespace Experimental {

namespace simd_abi {

template <int N>
class sve_fixed_size {};

}  // namespace simd_abi

namespace Impl {

template <class Derived, int Bits>
class sve_mask;

template <class Derived>
class sve_mask<Derived, 64> {
  vls_bool_t m_value;

 public:
  class reference {
    vls_bool_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_bool_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b64(pg, op);
      auto rep  = svdup_b64(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b64(pg, op);

      return svtest_any(svand_x(pred, m_mask));
    }

    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator vls_bool_t() const {
      return m_mask;
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b64(VL1);
        case 1: return svptrue_pat_b64(VL2);
        case 2: return svptrue_pat_b64(VL3);
        case 3: return svptrue_pat_b64(VL4);
        case 4: return svptrue_pat_b64(VL5);
        case 5: return svptrue_pat_b64(VL6);
        case 6: return svptrue_pat_b64(VL7);
        case 7: return svptrue_pat_b64(VL8);
        default: return svpfalse();
      }
      return svptrue_pat_b64(VL16);
    }
  };
  using value_type          = bool;
  using abi_type            = simd_abi::sve_fixed_size<8>;
  using implementation_type = vls_bool_t;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION sve_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit sve_mask(value_type value)
      : m_value(svdup_b64(value)) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit sve_mask(
      G&& gen) noexcept
      : m_value(svdupq_b64(gen(std::integral_constant<std::size_t, 0>()),
                           gen(std::integral_constant<std::size_t, 1>()),
                           gen(std::integral_constant<std::size_t, 2>()),
                           gen(std::integral_constant<std::size_t, 3>()),
                           gen(std::integral_constant<std::size_t, 4>()),
                           gen(std::integral_constant<std::size_t, 5>()),
                           gen(std::integral_constant<std::size_t, 6>()),
                           gen(std::integral_constant<std::size_t, 7>()))) {}
  template <class U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION sve_mask(sve_mask<U, 64> const& other)
      : m_value(other) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 8;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit sve_mask(
      implementation_type const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator vls_bool_t()
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
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived
  operator||(sve_mask const& other) const {
    return Derived(svorr_z(svptrue_b64(), m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived
  operator&&(sve_mask const& other) const {
    return Derived(svand_z(svptrue_b64(), m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived operator!() const {
    return Derived(svnot_z(svptrue_b64(), m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      sve_mask const& other) const {
    return svptest_any(svptrue_b64(), sveor_z(svptrue_b64(), m_value, other));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      sve_mask const& other) const {
    return !operator==(other);
  }
};

template <class Derived>
class sve_mask<Derived, 32> {
  vls_bool_t m_value;

 public:
  class reference {
    vls_bool_t& m_mask;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_bool_t& mask_arg,
                                                    int lane_arg)
        : m_mask(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(bool value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b32(pg, op);
      auto rep  = svdup_b32(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator bool() const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b32(pg, op);

      return svtest_any(svand_x(pred, m_mask));
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator vls_bool_t() const {
      return m_mask;
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b32(VL1);
        case 1: return svptrue_pat_b32(VL2);
        case 2: return svptrue_pat_b32(VL3);
        case 3: return svptrue_pat_b32(VL4);
        case 4: return svptrue_pat_b32(VL5);
        case 5: return svptrue_pat_b32(VL6);
        case 6: return svptrue_pat_b32(VL7);
        case 7: return svptrue_pat_b32(VL8);
        case 8: return svptrue_pat_b32(VL9);
        case 9: return svptrue_pat_b32(VL10);
        case 10: return svptrue_pat_b32(VL11);
        case 11: return svptrue_pat_b32(VL12);
        case 12: return svptrue_pat_b32(VL13);
        case 13: return svptrue_pat_b32(VL14);
        case 14: return svptrue_pat_b32(VL15);
        case 15: return svptrue_pat_b32(VL16);
        default: return svpfalse();
      }
      return svpfalse();
    }
  };
  using value_type          = bool;
  using abi_type            = simd_abi::sve_fixed_size<16>;
  using implementation_type = vls_bool_t;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION sve_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit sve_mask(value_type value)
      : m_value(svdup_b32(value)) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit sve_mask(
      G&& gen) noexcept
      : m_value(svdupq_b32(gen(std::integral_constant<std::size_t, 0>()),
                           gen(std::integral_constant<std::size_t, 1>()),
                           gen(std::integral_constant<std::size_t, 2>()),
                           gen(std::integral_constant<std::size_t, 3>()),
                           gen(std::integral_constant<std::size_t, 4>()),
                           gen(std::integral_constant<std::size_t, 5>()),
                           gen(std::integral_constant<std::size_t, 6>()),
                           gen(std::integral_constant<std::size_t, 7>()),
                           gen(std::integral_constant<std::size_t, 8>()),
                           gen(std::integral_constant<std::size_t, 9>()),
                           gen(std::integral_constant<std::size_t, 10>()),
                           gen(std::integral_constant<std::size_t, 11>()),
                           gen(std::integral_constant<std::size_t, 12>()),
                           gen(std::integral_constant<std::size_t, 13>()),
                           gen(std::integral_constant<std::size_t, 14>()),
                           gen(std::integral_constant<std::size_t, 15>()))) {}
  template <class U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION sve_mask(sve_mask<U, 32> const& other)
      : m_value(other) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 16;
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit sve_mask(
      implementation_type const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit operator vls_bool_t()
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
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived
  operator||(sve_mask const& other) const {
    return Derived(svorr_z(svptrue_b32(), m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived
  operator&&(sve_mask const& other) const {
    return Derived(svand_z(svptrue_b32(), m_value, other.m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Derived operator!() const {
    return Derived(svnot_z(svptrue_b32(), m_value));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator==(
      sve_mask const& other) const {
    return svptest_any(svptrue_b32(), sveor_z(svptrue_b32(), m_value, other));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION bool operator!=(
      sve_mask const& other) const {
    return !operator==(other);
  }
};

}  // namespace Impl

template <class T>
class simd_mask<T, simd_abi::sve_fixed_size<8>>
    : public Impl::sve_mask<simd_mask<T, simd_abi::sve_fixed_size<8>>,
                            sizeof(T) * 8> {
  using base_type =
      Impl::sve_mask<simd_mask<T, simd_abi::sve_fixed_size<8>>, sizeof(T) * 8>;

 public:
  using implementation_type = typename base_type::implementation_type;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(bool value)
      : base_type(value) {}
  template <class U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<U, simd_abi::sve_fixed_size<8>> const& other)
      : base_type(other) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      implementation_type const& value)
      : base_type(value) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<typename base_type::value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : base_type(gen) {}
};

template <class T>
class simd_mask<T, simd_abi::sve_fixed_size<16>>
    : public Impl::sve_mask<simd_mask<T, simd_abi::sve_fixed_size<8>>,
                            sizeof(T) * 8> {
  using base_type =
      Impl::sve_mask<simd_mask<T, simd_abi::sve_fixed_size<8>>, sizeof(T) * 8>;

 public:
  using implementation_type = typename base_type::implementation_type;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask() = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION explicit simd_mask(bool value)
      : base_type(value) {}
  template <class U>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd_mask(
      simd_mask<U, simd_abi::sve_fixed_size<8>> const& other)
      : base_type(other) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      implementation_type const& value)
      : base_type(value) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<typename base_type::value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd_mask(
      G&& gen) noexcept
      : base_type(gen) {}
};

template <>
class simd<double, simd_abi::sve_fixed_size<8>> {
  vls_float64_t m_value;

 public:
  using value_type = double;
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_float64_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_float64_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(double value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b64(pg, op);
      auto rep  = svdup_f64(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator double() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b64(VL1);
        case 1: return svptrue_pat_b64(VL2);
        case 2: return svptrue_pat_b64(VL3);
        case 3: return svptrue_pat_b64(VL4);
        case 4: return svptrue_pat_b64(VL5);
        case 5: return svptrue_pat_b64(VL6);
        case 6: return svptrue_pat_b64(VL7);
        case 7: return svptrue_pat_b64(VL8);
      }
      return svptrue_pat_b64(VL16);
    }
  };
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
      : m_value(svdup_f64(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>())};

    m_value = svld1(svptrue_b64(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_float64_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_float64_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b64(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                        static_cast<vls_float64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                        static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                        static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                        static_cast<vls_float64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                             static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                             static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                             static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b64(), static_cast<vls_float64_t>(lhs),
                             static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpeq(static_cast<vls_float64_t>(lhs),
                             static_cast<vls_float64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    abs(Experimental::simd<
        double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svabs_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    floor(Experimental::simd<
          double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svrintm_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    ceil(Experimental::simd<
         double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svrintp_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    round(Experimental::simd<
          double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svrinta_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    trunc(Experimental::simd<
          double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svrintz_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    double, Experimental::simd_abi::sve_fixed_size<8>>
copysign(
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        a,
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        b) {
  vls_uint64_t const sign_mask = svreinterpret_u64(svdup_f64(value_type(-0.0)));
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svreinterpret_f64(
          svor_x(svptrue_b64(), static_cast<vls_float64_t>(abs(a)),
                 svand_x(svptrue_b64(), sign_mask,
                         svreinterpret_u64(static_cast<vls_float64_t>(b))))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>
    sqrt(Experimental::simd<
         double, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svsqrt_x(svptrue_b64(), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    double, Experimental::simd_abi::sve_fixed_size<8>>
fma(Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        a,
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        b,
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        c) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svmad_x(svptrue_b64(), static_cast<vls_float64_t>(c),
              static_cast<vls_float64_t>(b), static_cast<vls_float64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    double, Experimental::simd_abi::sve_fixed_size<8>>
max(Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        a,
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        b) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svmax_x(svptrue_b64(), static_cast<vls_float64_t>(a),
              static_cast<vls_float64_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    double, Experimental::simd_abi::sve_fixed_size<8>>
min(Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        a,
    Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>> const&
        b) {
  return Experimental::simd<double, Experimental::simd_abi::sve_fixed_size<8>>(
      svmin_x(svptrue_b64(), static_cast<vls_float64_t>(a),
              static_cast<vls_float64_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<double, simd_abi::sve_fixed_size<8>>
    condition(simd_mask<double, simd_abi::sve_fixed_size<8>> const& a,
              simd<double, simd_abi::sve_fixed_size<8>> const& b,
              simd<double, simd_abi::sve_fixed_size<8>> const& c) {
  return simd<double, simd_abi::sve_fixed_size<8>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_float64_t>(b),
            static_cast<vls_float64_t>(c)));
}

template <>
class simd<float, simd_abi::sve_fixed_size<16>> {
  vls_float32_t m_value;

 public:
  using value_type = float;
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_float32_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_float32_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(float value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b32(pg, op);
      auto rep  = svdup_f32(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator float() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b32(VL1);
        case 1: return svptrue_pat_b32(VL2);
        case 2: return svptrue_pat_b32(VL3);
        case 3: return svptrue_pat_b32(VL4);
        case 4: return svptrue_pat_b32(VL5);
        case 5: return svptrue_pat_b32(VL6);
        case 6: return svptrue_pat_b32(VL7);
        case 7: return svptrue_pat_b32(VL8);
        case 8: return svptrue_pat_b32(VL9);
        case 9: return svptrue_pat_b32(VL10);
        case 10: return svptrue_pat_b32(VL11);
        case 11: return svptrue_pat_b32(VL12);
        case 12: return svptrue_pat_b32(VL13);
        case 13: return svptrue_pat_b32(VL14);
        case 14: return svptrue_pat_b32(VL15);
        case 15: return svptrue_pat_b32(VL16);
      }
      return svpfalse();
    }
  };
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 16;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(svdup_f32(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>()),
                         gen(std::integral_constant<std::size_t, 8>()),
                         gen(std::integral_constant<std::size_t, 9>()),
                         gen(std::integral_constant<std::size_t, 10>()),
                         gen(std::integral_constant<std::size_t, 11>()),
                         gen(std::integral_constant<std::size_t, 12>()),
                         gen(std::integral_constant<std::size_t, 13>()),
                         gen(std::integral_constant<std::size_t, 14>()),
                         gen(std::integral_constant<std::size_t, 15>())

    };

    m_value = svld1(svptrue_b32(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_float32_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_float32_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b32(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                        static_cast<vls_float32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                        static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                        static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                        static_cast<vls_float32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                             static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                             static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                             static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b32(), static_cast<vls_float32_t>(lhs),
                             static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpeq(static_cast<vls_float32_t>(lhs),
                             static_cast<vls_float32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    abs(Experimental::simd<
        float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svabs_x(svptrue_b32(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    floor(Experimental::simd<
          float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svrintm_x(svptrue_b32(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    ceil(Experimental::simd<
         float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svrintp_x(svptrue_b32(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    round(Experimental::simd<
          float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svrinta_x(svptrue_b32(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    trunc(Experimental::simd<
          float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svrintz_x(svptrue_b64(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    float, Experimental::simd_abi::sve_fixed_size<16>>
copysign(
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        a,
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        b) {
  vls_uint32_t const sign_mask = svreinterpret_u32(svdup_f32(value_type(-0.0)));
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svreinterpret_f32(
          svor_x(svptrue_b32(), static_cast<vls_float32_t>(abs(a)),
                 svand_x(svptrue_b32(), sign_mask,
                         svreinterpret_u32(static_cast<vls_float32_t>(b))))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>
    sqrt(Experimental::simd<
         float, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svsqrt_x(svptrue_b32(), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    float, Experimental::simd_abi::sve_fixed_size<16>>
fma(Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        a,
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        b,
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        c) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svmad_x(svptrue_b32(), static_cast<vls_float32_t>(c),
              static_cast<vls_float32_t>(b), static_cast<vls_float32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    float, Experimental::simd_abi::sve_fixed_size<16>>
max(Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        a,
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        b) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svmax_x(svptrue_b32(), static_cast<vls_float32_t>(a),
              static_cast<vls_float32_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    float, Experimental::simd_abi::sve_fixed_size<16>>
min(Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        a,
    Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>> const&
        b) {
  return Experimental::simd<float, Experimental::simd_abi::sve_fixed_size<16>>(
      svmin_x(svptrue_b32(), static_cast<vls_float32_t>(a),
              static_cast<vls_float32_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<float, simd_abi::sve_fixed_size<16>>
    condition(simd_mask<float, simd_abi::sve_fixed_size<16>> const& a,
              simd<float, simd_abi::sve_fixed_size<16>> const& b,
              simd<float, simd_abi::sve_fixed_size<16>> const& c) {
  return simd<float, simd_abi::sve_fixed_size<16>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_float32_t>(b),
            static_cast<vls_float32_t>(c)));
}

template <>
class simd<std::int32_t, simd_abi::sve_fixed_size<16>> {
  vls_int32_t m_value;

 public:
  using value_type = std::int32_t;
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_int32_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_int32_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator=(value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b32(pg, op);
      auto rep  = svdup_s32(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator std::int32_t() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b32(VL1);
        case 1: return svptrue_pat_b32(VL2);
        case 2: return svptrue_pat_b32(VL3);
        case 3: return svptrue_pat_b32(VL4);
        case 4: return svptrue_pat_b32(VL5);
        case 5: return svptrue_pat_b32(VL6);
        case 6: return svptrue_pat_b32(VL7);
        case 7: return svptrue_pat_b32(VL8);
        case 8: return svptrue_pat_b32(VL9);
        case 9: return svptrue_pat_b32(VL10);
        case 10: return svptrue_pat_b32(VL11);
        case 11: return svptrue_pat_b32(VL12);
        case 12: return svptrue_pat_b32(VL13);
        case 13: return svptrue_pat_b32(VL14);
        case 14: return svptrue_pat_b32(VL15);
        case 15: return svptrue_pat_b32(VL16);
      }
      return svpfalse();
    }
  };
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 16;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(svdup_s32(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>()),
                         gen(std::integral_constant<std::size_t, 8>()),
                         gen(std::integral_constant<std::size_t, 9>()),
                         gen(std::integral_constant<std::size_t, 10>()),
                         gen(std::integral_constant<std::size_t, 11>()),
                         gen(std::integral_constant<std::size_t, 12>()),
                         gen(std::integral_constant<std::size_t, 13>()),
                         gen(std::integral_constant<std::size_t, 14>()),
                         gen(std::integral_constant<std::size_t, 15>())

    };

    m_value = svld1(svptrue_b32(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_int32_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_int32_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b32(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                        static_cast<vls_int32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                        static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                        static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                        static_cast<vls_int32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                             static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                             static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                             static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b32(), static_cast<vls_int32_t>(lhs),
                             static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(
        svcmpeq(static_cast<vls_int32_t>(lhs), static_cast<vls_int32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    abs(Experimental::simd<
        std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svabs_x(svptrue_b32(), static_cast<vls_int32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    floor(Experimental::simd<
          std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_int32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    ceil(Experimental::simd<
         std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_int32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    round(Experimental::simd<
          std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_int32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    trunc(Experimental::simd<
          std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_int32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
copysign(
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  vls_uint32_t const sign_mask = svreinterpret_u32(svdup_s32(value_type(-0.0)));
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svreinterpret_s32(
          svor_x(svptrue_b32(), static_cast<vls_int32_t>(abs(a)),
                 svand_x(svptrue_b32(), sign_mask,
                         svreinterpret_u32(static_cast<vls_int32_t>(b))))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
    sqrt(Experimental::simd<
         std::int32_t, Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svsqrt_x(svptrue_b32(), static_cast<vls_int32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
fma(Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b,
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& c) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svmad_x(svptrue_b32(), static_cast<vls_int32_t>(c),
              static_cast<vls_int32_t>(b), static_cast<vls_int32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
max(Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(svmax_x(
      svptrue_b32(), static_cast<vls_int32_t>(a), static_cast<vls_int32_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::int32_t, Experimental::simd_abi::sve_fixed_size<16>>
min(Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::int32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  return Experimental::simd<std::int32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(svmin_x(
      svptrue_b32(), static_cast<vls_int32_t>(a), static_cast<vls_int32_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int32_t, simd_abi::sve_fixed_size<16>>
    condition(simd_mask<std::int32_t, simd_abi::sve_fixed_size<16>> const& a,
              simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& b,
              simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& c) {
  return simd<std::int32_t, simd_abi::sve_fixed_size<16>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_int32_t>(b),
            static_cast<vls_int32_t>(c)));
}

template <>
class simd<std::uint32_t, simd_abi::sve_fixed_size<16>> {
  vls_int32_t m_value;

 public:
  using value_type = std::uint32_t;
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_uint32_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_uint32_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator=(value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b32(pg, op);
      auto rep  = svdup_u32(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator std::uint32_t() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b32(VL1);
        case 1: return svptrue_pat_b32(VL2);
        case 2: return svptrue_pat_b32(VL3);
        case 3: return svptrue_pat_b32(VL4);
        case 4: return svptrue_pat_b32(VL5);
        case 5: return svptrue_pat_b32(VL6);
        case 6: return svptrue_pat_b32(VL7);
        case 7: return svptrue_pat_b32(VL8);
        case 8: return svptrue_pat_b32(VL9);
        case 9: return svptrue_pat_b32(VL10);
        case 10: return svptrue_pat_b32(VL11);
        case 11: return svptrue_pat_b32(VL12);
        case 12: return svptrue_pat_b32(VL13);
        case 13: return svptrue_pat_b32(VL14);
        case 14: return svptrue_pat_b32(VL15);
        case 15: return svptrue_pat_b32(VL16);
      }
      return svpfalse();
    }
  };
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd()            = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(simd&&)      = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd const&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd& operator=(simd&&) = default;
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION static constexpr std::size_t size() {
    return 16;
  }
  template <class U, std::enable_if_t<std::is_convertible_v<U, value_type>,
                                      bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd(U&& value)
      : m_value(svdup_u32(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>()),
                         gen(std::integral_constant<std::size_t, 8>()),
                         gen(std::integral_constant<std::size_t, 9>()),
                         gen(std::integral_constant<std::size_t, 10>()),
                         gen(std::integral_constant<std::size_t, 11>()),
                         gen(std::integral_constant<std::size_t, 12>()),
                         gen(std::integral_constant<std::size_t, 13>()),
                         gen(std::integral_constant<std::size_t, 14>()),
                         gen(std::integral_constant<std::size_t, 15>())

    };

    m_value = svld1(svptrue_b32(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_uint32_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b32(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b32(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_uint32_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b32(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                        static_cast<vls_uint32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                        static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                        static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                        static_cast<vls_uint32_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                             static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                             static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                             static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b32(), static_cast<vls_uint32_t>(lhs),
                             static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpeq(static_cast<vls_uint32_t>(lhs),
                             static_cast<vls_uint32_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
abs(Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svabs_x(svptrue_b32(), static_cast<vls_uint32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
floor(Experimental::simd<std::uint32_t,
                         Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_uint32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
ceil(Experimental::simd<std::uint32_t,
                        Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_uint32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
round(Experimental::simd<std::uint32_t,
                         Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_uint32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
trunc(Experimental::simd<std::uint32_t,
                         Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      static_cast<vls_uint32_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
copysign(
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  vls_uint32_t const sign_mask = svdup_u32(value_type(-0.0));
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svor_x(svptrue_b32(), static_cast<vls_uint32_t>(abs(a)),
             svand_x(svptrue_b32(), sign_mask, static_cast<vls_uint32_t>(b))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
sqrt(Experimental::simd<std::uint32_t,
                        Experimental::simd_abi::sve_fixed_size<16>> const& a) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svsqrt_x(svptrue_b32(), static_cast<vls_uint32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
fma(Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b,
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& c) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svmad_x(svptrue_b32(), static_cast<vls_uint32_t>(c),
              static_cast<vls_uint32_t>(b), static_cast<vls_uint32_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
max(Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svmax_x(svptrue_b32(), static_cast<vls_uint32_t>(a),
              static_cast<vls_uint32_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint32_t, Experimental::simd_abi::sve_fixed_size<16>>
min(Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& a,
    Experimental::simd<std::uint32_t,
                       Experimental::simd_abi::sve_fixed_size<16>> const& b) {
  return Experimental::simd<std::uint32_t,
                            Experimental::simd_abi::sve_fixed_size<16>>(
      svmin_x(svptrue_b32(), static_cast<vls_uint32_t>(a),
              static_cast<vls_uint32_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint32_t, simd_abi::sve_fixed_size<16>>
    condition(simd_mask<std::uint32_t, simd_abi::sve_fixed_size<16>> const& a,
              simd<std::uint32_t, simd_abi::sve_fixed_size<16>> const& b,
              simd<std::uint32_t, simd_abi::sve_fixed_size<16>> const& c) {
  return simd<std::uint32_t, simd_abi::sve_fixed_size<16>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_uint32_t>(b),
            static_cast<vls_uint32_t>(c)));
}

template <>
class simd<std::int64_t, simd_abi::sve_fixed_size<8>> {
  vls_int64_t m_value;

 public:
  using value_type = std::int64_t;
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_int64_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_int64_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(std::int64_t value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b64(pg, op);
      auto rep  = svdup_s64(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator std::int64_t() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b64(VL1);
        case 1: return svptrue_pat_b64(VL2);
        case 2: return svptrue_pat_b64(VL3);
        case 3: return svptrue_pat_b64(VL4);
        case 4: return svptrue_pat_b64(VL5);
        case 5: return svptrue_pat_b64(VL6);
        case 6: return svptrue_pat_b64(VL7);
        case 7: return svptrue_pat_b64(VL8);
      }
      return svptrue_pat_b64(VL16);
    }
  };
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
      : m_value(svdup_s64(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>())};
    m_value           = svld1(svptrue_b64(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_int64_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_int64_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b64(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                        static_cast<vls_int64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                        static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                        static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                        static_cast<vls_int64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                             static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                             static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                             static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b64(), static_cast<vls_int64_t>(lhs),
                             static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(
        svcmpeq(static_cast<vls_int64_t>(lhs), static_cast<vls_int64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    abs(Experimental::simd<
        std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_int64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    floor(Experimental::simd<
          std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_int64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    ceil(Experimental::simd<
         std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_int64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    round(Experimental::simd<
          std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_int64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    trunc(Experimental::simd<
          std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_int64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    copysign(
        Experimental::simd<std::int64_t,
                           Experimental::simd_abi::sve_fixed_size<8>> const& a,
        Experimental::simd<
            std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  vls_uint64_t const sign_mask = svreinterpret_u64(svdup_s64(value_type(-0.0)));
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svreinterpret_s64(
          svor_x(svptrue_b64(), static_cast<vls_int64_t>(abs(a)),
                 svand_x(svptrue_b64(), sign_mask,
                         svreinterpret_u64(static_cast<vls_int64_t>(b))))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    sqrt(Experimental::simd<
         std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svsqrt_x(svptrue_b64(), static_cast<vls_int64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    fma(Experimental::simd<std::int64_t,
                           Experimental::simd_abi::sve_fixed_size<8>> const& a,
        Experimental::simd<std::int64_t,
                           Experimental::simd_abi::sve_fixed_size<8>> const& b,
        Experimental::simd<
            std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& c) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svmad_x(svptrue_b64(), static_cast<vls_int64_t>(c),
              static_cast<vls_int64_t>(b), static_cast<vls_int64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    max(Experimental::simd<std::int64_t,
                           Experimental::simd_abi::sve_fixed_size<8>> const& a,
        Experimental::simd<
            std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(svmax_x(
      svptrue_b64(), static_cast<vls_int64_t>(a), static_cast<vls_int64_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::int64_t, Experimental::simd_abi::sve_fixed_size<8>>
    min(Experimental::simd<std::int64_t,
                           Experimental::simd_abi::sve_fixed_size<8>> const& a,
        Experimental::simd<
            std::int64_t, Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  return Experimental::simd<std::int64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(svmin_x(
      svptrue_b64(), static_cast<vls_int64_t>(a), static_cast<vls_int64_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::int64_t, simd_abi::sve_fixed_size<8>>
    condition(simd_mask<std::int64_t, simd_abi::sve_fixed_size<8>> const& a,
              simd<std::int64_t, simd_abi::sve_fixed_size<8>> const& b,
              simd<std::int64_t, simd_abi::sve_fixed_size<8>> const& c) {
  return simd<std::int64_t, simd_abi::sve_fixed_size<8>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_int64_t>(b),
            static_cast<vls_int64_t>(c)));
}

template <>
class simd<std::uint64_t, simd_abi::sve_fixed_size<8>> {
  vls_uint64_t m_value;

 public:
  using value_type = std::uint64_t;
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using mask_type  = simd_mask<value_type, abi_type>;
  class reference {
    vls_uint64_t& m_value;
    int m_lane;

   public:
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference(vls_uint64_t& mask_arg,
                                                    int lane_arg)
        : m_value(mask_arg), m_lane(lane_arg) {}
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference
    operator=(std::uint64_t value) const {
      auto pg   = get_pred(m_lane);
      auto op   = get_pred(m_lane - 1);
      auto pred = svpnext_b64(pg, op);
      auto rep  = svdup_u64(value);

      m_value = svsel(pred, rep, m_value);

      return *this;
    }
    KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION operator std::uint64_t() const {
      return svlastb(get_pred(m_lane), m_value);
    }

   private:
    vls_bool_t get_pred(int lane) {
      switch (lane) {
        case 0: return svptrue_pat_b64(VL1);
        case 1: return svptrue_pat_b64(VL2);
        case 2: return svptrue_pat_b64(VL3);
        case 3: return svptrue_pat_b64(VL4);
        case 4: return svptrue_pat_b64(VL5);
        case 5: return svptrue_pat_b64(VL6);
        case 6: return svptrue_pat_b64(VL7);
        case 7: return svptrue_pat_b64(VL8);
      }
      return svptrue_pat_b64(VL16);
    }
  };
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
      : m_value(svdup_u64(value_type(value))) {}
  template <class G,
            std::enable_if_t<
                std::is_invocable_r_v<value_type, G,
                                      std::integral_constant<std::size_t, 0>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      G&& gen) noexcept {
    value_type temp[] = {gen(std::integral_constant<std::size_t, 0>()),
                         gen(std::integral_constant<std::size_t, 1>()),
                         gen(std::integral_constant<std::size_t, 2>()),
                         gen(std::integral_constant<std::size_t, 3>()),
                         gen(std::integral_constant<std::size_t, 4>()),
                         gen(std::integral_constant<std::size_t, 5>()),
                         gen(std::integral_constant<std::size_t, 6>()),
                         gen(std::integral_constant<std::size_t, 7>())};
    m_value           = svld1(svptrue_b64(), temp);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit simd(
      vls_uint64_t const& value_in)
      : m_value(value_in) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION reference operator[](std::size_t i) {
    return reference(m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type
  operator[](std::size_t i) const {
    return reference(const_cast<simd*>(this)->m_value, int(i));
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       element_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_from(value_type const* ptr,
                                                       vector_aligned_tag) {
    m_value = svld1(svptrue_b64(), ptr);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(
      value_type* ptr, element_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void copy_to(value_type* ptr,
                                                     vector_aligned_tag) const {
    svst1(svptrue_b64(), ptr, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION constexpr explicit
  operator vls_uint64_t() const {
    return m_value;
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION simd operator-() const
      noexcept {
    return simd(svneg_z(svptrue_b64(), m_value));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator*(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svmul_z(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                        static_cast<vls_uint64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator/(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svdiv_z(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                        static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator+(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svadd_z(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                        static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend simd operator-(
      simd const& lhs, simd const& rhs) noexcept {
    return simd(svsub_z(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                        static_cast<vls_uint64_t>(rhs)));
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmplt(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                             static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpgt(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                             static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator<=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmple(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                             static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator>=(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpge(svptrue_b64(), static_cast<vls_uint64_t>(lhs),
                             static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator==(simd const& lhs, simd const& rhs) noexcept {
    return mask_type(svcmpeq(static_cast<vls_uint64_t>(lhs),
                             static_cast<vls_uint64_t>(rhs)));
  }
  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION friend mask_type
  operator!=(simd const& lhs, simd const& rhs) noexcept {
    return !(operator==(lhs, rhs));
  }
};

}  // namespace Experimental

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    abs(Experimental::simd<
        std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_uint64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    floor(Experimental::simd<
          std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_uint64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    ceil(Experimental::simd<
         std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_uint64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    round(Experimental::simd<
          std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_uint64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    trunc(Experimental::simd<
          std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      static_cast<vls_uint64_t>(a));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
copysign(
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& a,
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  vls_uint64_t const sign_mask = svreinterpret_u64(svdup_u64(value_type(-0.0)));
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svreinterpret_u64(svor_x(
          svptrue_b64(), static_cast<vls_uint64_t>(abs(a)),
          svand_x(svptrue_b64(), sign_mask, static_cast<vls_uint64_t>(b)))));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    Experimental::simd<std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
    sqrt(Experimental::simd<
         std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>> const& a) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svsqrt_x(svptrue_b64(), static_cast<vls_uint64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
fma(Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& a,
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& b,
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& c) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svmad_x(svptrue_b64(), static_cast<vls_uint64_t>(c),
              static_cast<vls_uint64_t>(b), static_cast<vls_uint64_t>(a)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
max(Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& a,
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svmax_x(svptrue_b64(), static_cast<vls_uint64_t>(a),
              static_cast<vls_uint64_t>(b)));
}

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION Experimental::simd<
    std::uint64_t, Experimental::simd_abi::sve_fixed_size<8>>
min(Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& a,
    Experimental::simd<std::uint64_t,
                       Experimental::simd_abi::sve_fixed_size<8>> const& b) {
  return Experimental::simd<std::uint64_t,
                            Experimental::simd_abi::sve_fixed_size<8>>(
      svmin_x(svptrue_b64(), static_cast<vls_uint64_t>(a),
              static_cast<vls_uint64_t>(b)));
}

namespace Experimental {

[[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
    simd<std::uint64_t, simd_abi::sve_fixed_size<8>>
    condition(simd_mask<std::uint64_t, simd_abi::sve_fixed_size<8>> const& a,
              simd<std::uint64_t, simd_abi::sve_fixed_size<8>> const& b,
              simd<std::uint64_t, simd_abi::sve_fixed_size<8>> const& c) {
  return simd<std::uint64_t, simd_abi::sve_fixed_size<8>>(
      svsel(static_cast<vls_bool_t>(a), static_cast<vls_uint64_t>(b),
            static_cast<vls_uint64_t>(c)));
}

template <>
class const_where_expression<simd_mask<double, simd_abi::sve_fixed_size<8>>,
                             simd<double, simd_abi::sve_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using value_type = simd<double, abi_type>;
  using mask_type  = simd_mask<double, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(double* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      double* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<double, simd_abi::sve_fixed_size<8>>,
                       simd<double, simd_abi::sve_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<double, simd_abi::sve_fixed_size<8>>,
          simd<double, simd_abi::sve_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<double, simd_abi::sve_fixed_size<8>> const& mask_arg,
      simd<double, simd_abi::sve_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(double const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      double const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <class U,
            std::enable_if_t<std::is_convertible_v<
                                 U, simd<double, simd_abi::sve_fixed_size<8>>>,
                             bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<double, simd_abi::sve_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<double, simd_abi::sve_fixed_size<8>>>(
        svsel(m_mask, static_cast<vls_float64_t>(x_as_value_type),
              static_cast<vls_float64_t>(m_value)));
  }
};

template <>
class const_where_expression<simd_mask<float, simd_abi::sve_fixed_size<16>>,
                             simd<float, simd_abi::sve_fixed_size<16>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using value_type = simd<float, abi_type>;
  using mask_type  = simd_mask<float, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(float* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      float* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
    if (m_mask[8]) mem[index[8]] = m_value[8];
    if (m_mask[9]) mem[index[9]] = m_value[9];
    if (m_mask[10]) mem[index[10]] = m_value[10];
    if (m_mask[11]) mem[index[11]] = m_value[11];
    if (m_mask[12]) mem[index[12]] = m_value[12];
    if (m_mask[13]) mem[index[13]] = m_value[13];
    if (m_mask[14]) mem[index[14]] = m_value[14];
    if (m_mask[15]) mem[index[15]] = m_value[15];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<float, simd_abi::sve_fixed_size<16>>,
                       simd<float, simd_abi::sve_fixed_size<16>>>
    : public const_where_expression<
          simd_mask<float, simd_abi::sve_fixed_size<16>>,
          simd<float, simd_abi::sve_fixed_size<16>>> {
 public:
  where_expression(
      simd_mask<float, simd_abi::sve_fixed_size<16>> const& mask_arg,
      simd<float, simd_abi::sve_fixed_size<16>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(float const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      float const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <class U,
            std::enable_if_t<std::is_convertible_v<
                                 U, simd<float, simd_abi::sve_fixed_size<16>>>,
                             bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<float, simd_abi::sve_fixed_size<16>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<float, simd_abi::sve_fixed_size<16>>>(
        svsel(m_mask, static_cast<vls_float32_t>(x_as_value_type),
              static_cast<vls_float32_t>(m_value)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int32_t, simd_abi::sve_fixed_size<16>>,
    simd<std::int32_t, simd_abi::sve_fixed_size<16>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using value_type = simd<std::int32_t, abi_type>;
  using mask_type  = simd_mask<std::int32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int32_t* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int32_t* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
    if (m_mask[8]) mem[index[8]] = m_value[8];
    if (m_mask[9]) mem[index[9]] = m_value[9];
    if (m_mask[10]) mem[index[10]] = m_value[10];
    if (m_mask[11]) mem[index[11]] = m_value[11];
    if (m_mask[12]) mem[index[12]] = m_value[12];
    if (m_mask[13]) mem[index[13]] = m_value[13];
    if (m_mask[14]) mem[index[14]] = m_value[14];
    if (m_mask[15]) mem[index[15]] = m_value[15];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::int32_t, simd_abi::sve_fixed_size<16>>,
                       simd<std::int32_t, simd_abi::sve_fixed_size<16>>>
    : public const_where_expression<
          simd_mask<std::int32_t, simd_abi::sve_fixed_size<16>>,
          simd<std::int32_t, simd_abi::sve_fixed_size<16>>> {
 public:
  where_expression(
      simd_mask<std::int32_t, simd_abi::sve_fixed_size<16>> const& mask_arg,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int32_t const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int32_t const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, simd<std::int32_t, simd_abi::sve_fixed_size<16>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int32_t, simd_abi::sve_fixed_size<16>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<std::int32_t, simd_abi::sve_fixed_size<16>>>(
        svsel(m_mask, static_cast<vls_int32_t>(x_as_value_type),
              static_cast<vls_int32_t>(m_value)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::uint32_t, simd_abi::sve_fixed_size<16>>,
    simd<std::uint32_t, simd_abi::sve_fixed_size<16>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<16>;
  using value_type = simd<std::uint32_t, abi_type>;
  using mask_type  = simd_mask<std::uint32_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint32_t* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint32_t* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::uint32_t* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
    if (m_mask[8]) mem[index[8]] = m_value[8];
    if (m_mask[9]) mem[index[9]] = m_value[9];
    if (m_mask[10]) mem[index[10]] = m_value[10];
    if (m_mask[11]) mem[index[11]] = m_value[11];
    if (m_mask[12]) mem[index[12]] = m_value[12];
    if (m_mask[13]) mem[index[13]] = m_value[13];
    if (m_mask[14]) mem[index[14]] = m_value[14];
    if (m_mask[15]) mem[index[15]] = m_value[15];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::uint32_t, simd_abi::sve_fixed_size<16>>,
                       simd<std::uint32_t, simd_abi::sve_fixed_size<16>>>
    : public const_where_expression<
          simd_mask<std::uint32_t, simd_abi::sve_fixed_size<16>>,
          simd<std::uint32_t, simd_abi::sve_fixed_size<16>>> {
 public:
  where_expression(
      simd_mask<std::uint32_t, simd_abi::sve_fixed_size<16>> const& mask_arg,
      simd<std::uint32_t, simd_abi::sve_fixed_size<16>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint32_t const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint32_t const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint32_t const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<16>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <class U,
            std::enable_if_t<
                std::is_convertible_v<
                    U, simd<std::uint32_t, simd_abi::sve_fixed_size<16>>>,
                bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::uint32_t, simd_abi::sve_fixed_size<16>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<std::uint32_t, simd_abi::sve_fixed_size<16>>>(
        svsel(m_mask, static_cast<vls_uint32_t>(x_as_value_type),
              static_cast<vls_uint32_t>(m_value)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::int64_t, simd_abi::sve_fixed_size<8>>,
    simd<std::int64_t, simd_abi::sve_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using value_type = simd<std::int64_t, abi_type>;
  using mask_type  = simd_mask<std::int64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int64_t* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::int64_t* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::int64_t* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::int64_t, simd_abi::sve_fixed_size<8>>,
                       simd<std::int64_t, simd_abi::sve_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::int64_t, simd_abi::sve_fixed_size<8>>,
          simd<std::int64_t, simd_abi::sve_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::int64_t, simd_abi::sve_fixed_size<8>> const& mask_arg,
      simd<std::int64_t, simd_abi::sve_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int64_t const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::int64_t const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::int64_t const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, simd<std::int64_t, simd_abi::sve_fixed_size<8>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::int64_t, simd_abi::sve_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<std::int64_t, simd_abi::sve_fixed_size<8>>>(
        svsel(m_mask, static_cast<vls_int64_t>(x_as_value_type),
              static_cast<vls_int64_t>(m_value)));
  }
};

template <>
class const_where_expression<
    simd_mask<std::uint64_t, simd_abi::sve_fixed_size<8>>,
    simd<std::uint64_t, simd_abi::sve_fixed_size<8>>> {
 public:
  using abi_type   = simd_abi::sve_fixed_size<8>;
  using value_type = simd<std::uint64_t, abi_type>;
  using mask_type  = simd_mask<std::uint64_t, abi_type>;

 protected:
  value_type& m_value;
  mask_type const& m_mask;

 public:
  const_where_expression(mask_type const& mask_arg, value_type const& value_arg)
      : m_value(const_cast<value_type&>(value_arg)), m_mask(mask_arg) {}

  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint64_t* mem, element_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_to(std::uint64_t* mem, vector_aligned_tag) const {
    svst1(m_mask, mem, m_value);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void scatter_to(
      std::uint64_t* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) const {
    if (m_mask[0]) mem[index[0]] = m_value[0];
    if (m_mask[1]) mem[index[1]] = m_value[1];
    if (m_mask[2]) mem[index[2]] = m_value[2];
    if (m_mask[3]) mem[index[3]] = m_value[3];
    if (m_mask[4]) mem[index[4]] = m_value[4];
    if (m_mask[5]) mem[index[5]] = m_value[5];
    if (m_mask[6]) mem[index[6]] = m_value[6];
    if (m_mask[7]) mem[index[7]] = m_value[7];
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION value_type const&
  impl_get_value() const {
    return m_value;
  }

  [[nodiscard]] KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION mask_type const&
  impl_get_mask() const {
    return m_mask;
  }
};

template <>
class where_expression<simd_mask<std::uint64_t, simd_abi::sve_fixed_size<8>>,
                       simd<std::uint64_t, simd_abi::sve_fixed_size<8>>>
    : public const_where_expression<
          simd_mask<std::uint64_t, simd_abi::sve_fixed_size<8>>,
          simd<std::uint64_t, simd_abi::sve_fixed_size<8>>> {
 public:
  where_expression(
      simd_mask<std::uint64_t, simd_abi::sve_fixed_size<8>> const& mask_arg,
      simd<std::uint64_t, simd_abi::sve_fixed_size<8>>& value_arg)
      : const_where_expression(mask_arg, value_arg) {}
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint64_t const* mem, element_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void copy_from(std::uint64_t const* mem, vector_aligned_tag) {
    m_value = svld1(m_mask, mem);
  }
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION
  void gather_from(
      std::uint64_t const* mem,
      simd<std::int32_t, simd_abi::sve_fixed_size<8>> const& index) {
    m_value = svld1_gather_index(m_mask, mem, index);
  }
  template <
      class U,
      std::enable_if_t<std::is_convertible_v<
                           U, simd<std::uint64_t, simd_abi::sve_fixed_size<8>>>,
                       bool> = false>
  KOKKOS_IMPL_HOST_FORCEINLINE_FUNCTION void operator=(U&& x) {
    auto const x_as_value_type =
        static_cast<simd<std::uint64_t, simd_abi::sve_fixed_size<8>>>(
            std::forward<U>(x));
    m_value = static_cast<simd<std::uint64_t, simd_abi::sve_fixed_size<8>>>(
        svsel(m_mask, static_cast<vls_uint64_t>(x_as_value_type),
              static_cast<vls_uint64_t>(m_value)));
  }
};

}  // namespace Experimental
}  // namespace Kokkos

#endif
