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

#ifndef KOKKOS_QUAD_PRECISION_MATH_HPP
#define KOKKOS_QUAD_PRECISION_MATH_HPP

#include <Kokkos_NumericTraits.hpp>

#include <quadmath.h>

//<editor-fold desc="numeric traits __float128 specializations">
#if defined(KOKKOS_ENABLE_CXX17)
#define KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(TRAIT, TYPE, VALUE_TYPE, VALUE) \
  template <>                                                                \
  struct Kokkos::Experimental::TRAIT<TYPE> {                                 \
    static constexpr VALUE_TYPE value = VALUE;                               \
  };                                                                         \
  template <>                                                                \
  inline constexpr auto Kokkos::Experimental::TRAIT##_v<TYPE> =              \
      Kokkos::Experimental::TRAIT<TYPE>::value;
#else
#define KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(TRAIT, TYPE, VALUE_TYPE, VALUE) \
  template <>                                                                \
  struct Kokkos::Experimental::TRAIT<TYPE> {                                 \
    static constexpr VALUE_TYPE value = VALUE;                               \
  };
#endif

// clang-format off
// Numeric distinguished value traits
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(infinity,       __float128, __float128, HUGE_VALQ)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(finite_min,     __float128, __float128, -FLT128_MAX)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(finite_max,     __float128, __float128, FLT128_MAX)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(epsilon,        __float128, __float128, FLT128_EPSILON)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(round_error,    __float128, __float128, static_cast<__float128>(0.5))
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(norm_min,       __float128, __float128, FLT128_MIN)

KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(digits,         __float128,        int, FLT128_MANT_DIG)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(digits10,       __float128,        int, FLT128_DIG)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_digits10,   __float128,        int, 36)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(radix,          __float128,        int, 2)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(min_exponent,   __float128,        int, FLT128_MIN_EXP)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_exponent,   __float128,        int, FLT128_MAX_EXP)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(min_exponent10, __float128,        int, FLT128_MIN_10_EXP)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(max_exponent10, __float128,        int, FLT128_MAX_10_EXP)
// clang-format on

#undef KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT
//</editor-fold>

#endif
