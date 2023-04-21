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

#ifndef KOKKOS_HALF_MATHEMATICAL_FUNCTIONS_HPP_
#define KOKKOS_HALF_MATHEMATICAL_FUNCTIONS_HPP_

#include <Kokkos_MathematicalFunctions.hpp>  // For the float overloads

namespace Kokkos {
////////////// BEGIN HALF_T (float16) MATH FNS ////////////
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
#define KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(FUNC)          \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t FUNC( \
      Kokkos::Experimental::half_t x) {                     \
    return static_cast<Kokkos::Experimental::half_t>(       \
        Kokkos::FUNC(static_cast<float>(x)));               \
  }

KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(fabs)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(exp)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(exp2)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(expm1)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(log)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(log10)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(log2)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(log1p)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(sqrt)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(cbrt)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(sin)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(cos)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(tan)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(asin)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(acos)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(atan)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(sinh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(cosh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(tanh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(asinh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(acosh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(atanh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(erf)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(erfc)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(tgamma)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(lgamma)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(ceil)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(floor)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(trunc)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(round)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(nearbyint)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF(logb)

#undef KOKKOS_IMPL_MATH_UNARY_FUNCTION_HALF

#define KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(FUNC)                     \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t FUNC(             \
      Kokkos::Experimental::half_t x, Kokkos::Experimental::half_t y) { \
    return static_cast<Kokkos::Experimental::half_t>(                   \
        Kokkos::FUNC(static_cast<float>(x), static_cast<float>(y)));    \
  }

KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(fmod)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(remainder)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(fmax)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(fmin)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(fdim)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(pow)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(hypot)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(atan2)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(nextafter)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF(copysign)

#undef KOKKOS_IMPL_MATH_BINARY_FUNCTION_HALF

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t abs(
    Kokkos::Experimental::half_t x) {
  return static_cast<Kokkos::Experimental::half_t>(
      Kokkos::abs(static_cast<float>(x)));
}

#define KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF(FUNC)                  \
  KOKKOS_INLINE_FUNCTION bool FUNC(Kokkos::Experimental::half_t x) { \
    return Kokkos::FUNC(static_cast<float>(x));                      \
  }

KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF(isfinite)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF(isinf)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF(isnan)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF(signbit)

#undef KOKKOS_IMPL_MATH_UNARY_PREDICATE_HALF

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t real(
    Kokkos::Experimental::half_t x) {
  return x;
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t imag(
    Kokkos::Experimental::half_t) {
  return 0;
}
#endif  // defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
////////////// END HALF_T (float16) MATH FNS ////////////

////////////// BEGIN BHALF_T (bfloat16) MATH FNS ////////////
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
#define KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(FUNC)          \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t FUNC( \
      Kokkos::Experimental::bhalf_t x) {                     \
    return static_cast<Kokkos::Experimental::bhalf_t>(       \
        Kokkos::FUNC(static_cast<float>(x)));                \
  }

KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(fabs)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(exp)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(exp2)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(expm1)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(log)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(log10)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(log2)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(log1p)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(sqrt)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(cbrt)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(sin)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(cos)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(tan)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(asin)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(acos)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(atan)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(sinh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(cosh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(tanh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(asinh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(acosh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(atanh)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(erf)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(erfc)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(tgamma)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(lgamma)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(ceil)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(floor)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(trunc)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(round)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(nearbyint)
KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF(logb)

#undef KOKKOS_IMPL_MATH_UNARY_FUNCTION_BHALF

#define KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(FUNC)                      \
  KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t FUNC(              \
      Kokkos::Experimental::bhalf_t x, Kokkos::Experimental::bhalf_t y) { \
    return static_cast<Kokkos::Experimental::bhalf_t>(                    \
        Kokkos::FUNC(static_cast<float>(x), static_cast<float>(y)));      \
  }

KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(fmod)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(remainder)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(fmax)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(fmin)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(fdim)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(pow)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(hypot)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(atan2)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(nextafter)
KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF(copysign)

#undef KOKKOS_IMPL_MATH_BINARY_FUNCTION_BHALF

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t abs(
    Kokkos::Experimental::bhalf_t x) {
  return static_cast<Kokkos::Experimental::bhalf_t>(
      Kokkos::abs(static_cast<float>(x)));
}

#define KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF(FUNC)                  \
  KOKKOS_INLINE_FUNCTION bool FUNC(Kokkos::Experimental::bhalf_t x) { \
    return Kokkos::FUNC(static_cast<float>(x));                       \
  }

KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF(isfinite)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF(isinf)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF(isnan)
KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF(signbit)

#undef KOKKOS_IMPL_MATH_UNARY_PREDICATE_BHALF

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t real(
    Kokkos::Experimental::bhalf_t x) {
  return x;
}

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t imag(
    Kokkos::Experimental::bhalf_t) {
  return 0;
}
#endif  // defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
////////////// END BHALF_T (bfloat16) MATH FNS //////////
}  // namespace Kokkos

#undef KOKKOS_IMPL_MATH_FUNCTIONS_NAMESPACE

#endif  // KOKKOS_HALF_MATHEMATICAL_FUNCTIONS_HPP_
