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

template <class Mask, class Value>
class const_where_expression {
 protected:
  Value& m_value;
  Mask const& m_mask;

 public:
  const_where_expression(Mask const& mask_arg, Value const& value_arg)
      : m_value(const_cast<Value&>(value_arg)), m_mask(mask_arg) {}
};

template <class Mask, class Value>
class where_expression : public const_where_expression<Mask, Value> {
  using base_type = const_where_expression<Mask, Value>;

 public:
  where_expression(Mask const& mask_arg, Value& value_arg)
      : base_type(mask_arg, value_arg) {}
};

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(Kokkos::Impl::identity_t<simd_mask<T, Abi>> const& mask,
          simd<T, Abi>& value) {
  return where_expression(mask, value);
}

template <class T, class Abi>
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(Kokkos::Impl::identity_t<simd_mask<T, Abi>> const& mask,
          simd<T, Abi> const& value) {
  return const_where_expression(mask, value);
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

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr bool all_of(bool a) {
  return a;
}

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION constexpr bool any_of(bool a) {
  return a;
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
