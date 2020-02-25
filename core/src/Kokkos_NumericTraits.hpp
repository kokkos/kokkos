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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_NUMERICTRAITS_HPP
#define KOKKOS_NUMERICTRAITS_HPP

#include <limits>
#include <type_traits>

namespace Kokkos {

template <class T>
struct reduction_identity {
  static_assert(
      std::is_arithmetic<T>::value,
      "reduction_identity can only be instantiated for arithmetic types");

  KOKKOS_FORCEINLINE_FUNCTION constexpr static T sum() {
    return static_cast<T>(0);
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr static T prod() {
    return static_cast<T>(1);
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr static T max() {
    return std::numeric_limits<T>::lowest();
  }

  KOKKOS_FORCEINLINE_FUNCTION constexpr static T min() {
    return std::numeric_limits<T>::max();
  }

  template <class Dummy = T>
  KOKKOS_FORCEINLINE_FUNCTION constexpr static typename std::enable_if<
      std::is_same<Dummy, T>::value && std::is_integral<T>::value, T>::type
  bor() {
    return static_cast<T>(0x0);
  }

  template <class Dummy = T>
  KOKKOS_FORCEINLINE_FUNCTION constexpr static typename std::enable_if<
      std::is_same<Dummy, T>::value && std::is_integral<T>::value, T>::type
  band() {
    return ~static_cast<T>(0x0);
  }

  template <class Dummy = T>
  KOKKOS_FORCEINLINE_FUNCTION constexpr static typename std::enable_if<
      std::is_same<Dummy, T>::value && std::is_integral<T>::value, T>::type
  lor() {
    return static_cast<T>(0);
  }

  template <class Dummy = T>
  KOKKOS_FORCEINLINE_FUNCTION constexpr static typename std::enable_if<
      std::is_same<Dummy, T>::value && std::is_integral<T>::value, T>::type
  land() {
    return static_cast<T>(1);
  }
};

}  // namespace Kokkos

#endif
