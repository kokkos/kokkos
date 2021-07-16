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

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_LIBQUADMATH)

#include <Kokkos_NumericTraits.hpp>

#include <quadmath.h>

#if !(defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__))
#error __float128 not supported on this host
#endif

//<editor-fold desc="numeric traits __float128 specializations">
namespace Kokkos {
namespace Experimental {
#if defined(KOKKOS_ENABLE_CXX17)
#define KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(TRAIT, TYPE, VALUE_TYPE, VALUE) \
  template <>                                                                \
  struct TRAIT<TYPE> {                                                       \
    static constexpr VALUE_TYPE value = VALUE;                               \
  };                                                                         \
  template <>                                                                \
  inline constexpr auto TRAIT##_v<TYPE> = TRAIT<TYPE>::value;
#else
#define KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(TRAIT, TYPE, VALUE_TYPE, VALUE) \
  template <>                                                                \
  struct TRAIT<TYPE> {                                                       \
    static constexpr VALUE_TYPE value = VALUE;                               \
  };
#endif

// clang-format off
// Numeric distinguished value traits
// Workaround GCC bug https://godbolt.org/z/qWb5oe4dx
// error: '__builtin_huge_valq()' is not a constant expression
#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU >= 710)
KOKKOS_IMPL_SPECIALIZE_NUMERIC_TRAIT(infinity,       __float128, __float128, HUGE_VALQ)
#endif
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
}  // namespace Experimental
}  // namespace Kokkos
//</editor-fold>

namespace Kokkos {
template <>
struct reduction_identity<__float128> {
  KOKKOS_FORCEINLINE_FUNCTION constexpr static __float128 sum() {
    return static_cast<__float128>(0.0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static __float128 prod() {
    return static_cast<__float128>(1.0);
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static __float128 max() {
    return -FLT128_MAX;
  }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static __float128 min() {
    return FLT128_MAX;
  }
};
}  // namespace Kokkos

//<editor-fold desc="Common mathematical functions __float128 overloads">
namespace Kokkos {
namespace Experimental {
// clang-format off
// Basic operations
inline __float128 fabs(__float128 x) { return ::fabsq(x); }
inline __float128 fmod(__float128 x, __float128 y) { return ::fmodq(x, y); }
inline __float128 remainder(__float128 x, __float128 y) { return ::remainderq(x, y); }
inline __float128 fmin(__float128 x, __float128 y) { return ::fminq(x, y); }
inline __float128 fmax(__float128 x, __float128 y) { return ::fmaxq(x, y); }
inline __float128 fdim(__float128 x, __float128 y) { return ::fdimq(x, y); }
inline __float128 nanq(char const* arg) { return ::nanq(arg); }
// Power functions
inline __float128 pow(__float128 x, __float128 y) { return ::powq(x, y); }
inline __float128 sqrt(__float128 x) { return ::sqrtq(x); }
inline __float128 cbrt(__float128 x) { return ::cbrtq(x); }
inline __float128 hypot(__float128 x, __float128 y) { return ::hypotq(x, y); }
// Exponential functions
inline __float128 exp(__float128 x) { return ::expq(x); }
#if defined(KOKKOS_COMPILER_GNU) && (KOKKOS_COMPILER_GNU >= 910)
inline __float128 exp2(__float128 x) { return ::exp2q(x); }
#endif
inline __float128 expm1(__float128 x) { return ::expm1q(x); }
inline __float128 log(__float128 x) { return ::logq(x); }
inline __float128 log10(__float128 x) { return ::log10q(x); }
inline __float128 log2(__float128 x) { return ::log2q(x); }
inline __float128 log1p(__float128 x) { return ::log1pq(x); }
// Trigonometric functions
inline __float128 sin(__float128 x) { return ::sinq(x); }
inline __float128 cos(__float128 x) { return ::cosq(x); }
inline __float128 tan(__float128 x) { return ::tanq(x); }
inline __float128 asin(__float128 x) { return ::asinq(x); }
inline __float128 acos(__float128 x) { return ::acosq(x); }
inline __float128 atan(__float128 x) { return ::atanq(x); }
inline __float128 atan2(__float128 x, __float128 y) { return ::atan2q(x, y); }
// Hyperbolic functions
inline __float128 sinh(__float128 x) { return ::sinhq(x); }
inline __float128 cosh(__float128 x) { return ::coshq(x); }
inline __float128 tanh(__float128 x) { return ::tanhq(x); }
inline __float128 asinh(__float128 x) { return ::asinhq(x); }
inline __float128 acosh(__float128 x) { return ::acoshq(x); }
inline __float128 atanh(__float128 x) { return ::atanhq(x); }
// Error and gamma functions
inline __float128 erf(__float128 x) { return ::erfq(x); }
inline __float128 erfc(__float128 x) { return ::erfcq(x); }
inline __float128 tgamma(__float128 x) { return ::tgammaq(x); }
inline __float128 lgamma(__float128 x) { return ::lgammaq(x); }
// Nearest integer floating point operations
inline __float128 ceil(__float128 x) { return ::ceilq(x); }
inline __float128 floor(__float128 x) { return ::floorq(x); }
inline __float128 trunc(__float128 x) { return ::truncq(x); }
inline __float128 nearbyint(__float128 x) { return ::nearbyintq(x); }
// Classification and comparison
inline bool isfinite(__float128 x) { return !::isinfq(x); }  // isfiniteq not provided
inline bool isinf(__float128 x) { return ::isinfq(x); }
inline bool isnan(__float128 x) { return ::isnanq(x); }
}  // namespace Experimental
}  // namespace Kokkos
//</editor-fold>

#endif

#endif
