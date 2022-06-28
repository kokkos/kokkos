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

#ifndef KOKKOS_SIMD_COMMON_HPP
#define KOKKOS_SIMD_COMMON_HPP

#include <cmath>
#include <cstring>

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

template <class To, class From>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr To bit_cast(
    From const& src) {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <class T, class Abi>
class simd;

template <class T, class Abi>
class simd_mask;

struct element_aligned_tag {};

// class template declarations for const_where_expression and where_expression

template <class M, class T>
class const_where_expression {
 protected:
  T& m_value;
  M const& m_mask;

 public:
  const_where_expression(M const& mask_arg, T const& value_arg)
      : m_value(const_cast<T&>(value_arg)), m_mask(mask_arg) {}
};

template <class M, class T>
class where_expression : public const_where_expression<M, T> {
  using base_type = const_where_expression<M, T>;

 public:
  where_expression(M const& mask_arg, T& value_arg)
      : base_type(mask_arg, value_arg) {}
};

// specializations of where expression templates for the case when the
// mask type is bool, to allow generic code to use where() on both
// SIMD types and non-SIMD builtin arithmetic types

template <class T>
class const_where_expression<bool, T> {
 protected:
  T& m_value;
  bool m_mask;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const_where_expression(bool mask_arg, T const& value_arg)
      : m_value(const_cast<T&>(value_arg)), m_mask(mask_arg) {}
};

template <class T>
class where_expression<bool, T> : public const_where_expression<bool, T> {
  using base_type = const_where_expression<bool, T>;

 public:
  KOKKOS_FORCEINLINE_FUNCTION
  where_expression(bool mask_arg, T& value_arg)
      : base_type(mask_arg, value_arg) {}
  template <class U,
            std::enable_if_t<std::is_convertible_v<U, T>, bool> = false>
  KOKKOS_FORCEINLINE_FUNCTION void operator=(U const& x) {
    if (this->m_mask) this->m_value = x;
  }
};

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(typename simd<T, Abi>::mask_type const& mask, simd<T, Abi>& value) {
  return where_expression(mask, value);
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(typename simd<T, Abi>::mask_type const& mask,
          simd<T, Abi> const& value) {
  return const_where_expression(mask, value);
}

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION where_expression<bool, T> where(
    bool mask, T& value) {
  return where_expression(mask, value);
}

template <class T>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION const_where_expression<bool, T> where(
    bool mask, T const& value) {
  return const_where_expression(mask, value);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator+(
    simd<T, Abi> const& lhs,
    U const& rhs) {
  using result_type = decltype(T() + U());
  return simd<result_type, Abi>(lhs) + simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator+(
    U const& lhs,
    simd<T, Abi> const& rhs) {
  using result_type = decltype(U() + T());
  return simd<result_type, Abi>(lhs) + simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator-(
    simd<T, Abi> const& lhs,
    U const& rhs) {
  using result_type = decltype(T() - U());
  return simd<result_type, Abi>(lhs) - simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator-(
    U const& lhs,
    simd<T, Abi> const& rhs) {
  using result_type = decltype(U() - T());
  return simd<result_type, Abi>(lhs) - simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator*(
    simd<T, Abi> const& lhs,
    U const& rhs) {
  using result_type = decltype(T() * U());
  return simd<result_type, Abi>(lhs) * simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator*(
    U const& lhs,
    simd<T, Abi> const& rhs) {
  using result_type = decltype(U() * T());
  return simd<result_type, Abi>(lhs) * simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator/(
    simd<T, Abi> const& lhs,
    U const& rhs) {
  using result_type = decltype(T() / U());
  return simd<result_type, Abi>(lhs) / simd<result_type, Abi>(rhs);
}

template <class T, class U, class Abi,
         std::enable_if_t<std::is_arithmetic_v<U>, bool> = false>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
auto operator/(
    U const& lhs,
    simd<T, Abi> const& rhs) {
  using result_type = decltype(U() / T());
  return simd<result_type, Abi>(lhs) / simd<result_type, Abi>(rhs);
}

template <class T, class Abi>
KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi>& operator+=(
    simd<T, Abi>& a, Kokkos::Impl::identity_t<simd<T, Abi>> const& b) {
  a = a + b;
  return a;
}

template <class T, class Abi>
KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi>& operator-=(
    simd<T, Abi>& a, Kokkos::Impl::identity_t<simd<T, Abi>> const& b) {
  a = a - b;
  return a;
}

template <class T, class Abi>
KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi>& operator*=(
    simd<T, Abi>& a, Kokkos::Impl::identity_t<simd<T, Abi>> const& b) {
  a = a * b;
  return a;
}

template <class T, class Abi>
KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi>& operator/=(
    simd<T, Abi>& a, Kokkos::Impl::identity_t<simd<T, Abi>> const& b) {
  a = a / b;
  return a;
}

// implement mask reductions for type bool to allow generic code to accept
// both simd<double, Abi> and just double

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr bool all_of(bool a) {
  return a;
}

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr bool any_of(bool a) {
  return a;
}

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr bool none_of(bool a) {
  return !a;
}

// fallback implementations of reductions across simd_mask:

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool all_of(
    simd_mask<T, Abi> const& a) {
  return a == simd_mask<T, Abi>(true);
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool any_of(
    simd_mask<T, Abi> const& a) {
  return a != simd_mask<T, Abi>(false);
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool none_of(
    simd_mask<T, Abi> const& a) {
  return a == simd_mask<T, Abi>(false);
}

// fallback implementations of transcendental functions.
// individual Abi types may provide overloads with more efficient
// implementations.

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi> exp(simd<T, Abi> a) {
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = Kokkos::exp(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi> pow(
    simd<T, Abi> a, simd<T, Abi> const& b) {
  T a_array[simd<T, Abi>::size()];
  T b_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  b.copy_to(b_array, element_aligned_tag());
  for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = Kokkos::pow(a_array[i], b_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi> sin(simd<T, Abi> a) {
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = Kokkos::sin(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION simd<T, Abi> cos(simd<T, Abi> a) {
  T a_array[simd<T, Abi>::size()];
  a.copy_to(a_array, element_aligned_tag());
  for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) {
    a_array[i] = Kokkos::cos(a_array[i]);
  }
  a.copy_from(a_array, element_aligned_tag());
  return a;
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
