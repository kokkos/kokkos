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

#ifndef KOKKOS_STD_MIN_MAX_OPERATIONS_HPP
#define KOKKOS_STD_MIN_MAX_OPERATIONS_HPP

#include <Kokkos_Core.hpp>

namespace Kokkos {
namespace Experimental {

// clamp
template <class T>
constexpr KOKKOS_INLINE_FUNCTION const T& clamp(const T& value, const T& lo,
                                                const T& hi) {
  return (value < lo) ? lo : (hi < value) ? hi : value;
}

template <class T, class ComparatorType>
constexpr KOKKOS_INLINE_FUNCTION const T& clamp(const T& value, const T& lo,
                                                const T& hi,
                                                ComparatorType comp) {
  return comp(value, lo) ? lo : comp(hi, value) ? hi : value;
}

// max
template <class T>
constexpr KOKKOS_INLINE_FUNCTION const T& max(const T& a, const T& b) {
  return (a < b) ? b : a;
}

template <class T, class ComparatorType>
constexpr KOKKOS_INLINE_FUNCTION const T& max(const T& a, const T& b,
                                              ComparatorType comp) {
  return comp(a, b) ? b : a;
}

// min
template <class T>
constexpr KOKKOS_INLINE_FUNCTION const T& min(const T& a, const T& b) {
  return (b < a) ? b : a;
}

template <class T, class ComparatorType>
constexpr KOKKOS_INLINE_FUNCTION const T& min(const T& a, const T& b,
                                              ComparatorType comp) {
  return comp(b, a) ? b : a;
}

// minmax
template <class T>
constexpr KOKKOS_INLINE_FUNCTION auto minmax(const T& a, const T& b) {
  using return_t = ::Kokkos::pair<const T&, const T&>;
  return (b < a) ? return_t{b, a} : return_t{a, b};
}

template <class T, class ComparatorType>
constexpr KOKKOS_INLINE_FUNCTION auto minmax(const T& a, const T& b,
                                             ComparatorType comp) {
  using return_t = ::Kokkos::pair<const T&, const T&>;
  return comp(b, a) ? return_t{b, a} : return_t{a, b};
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
