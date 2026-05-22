// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_STD_HALF_MATHEMATICAL_FUNCTIONS_HPP_
#define KOKKOS_STD_HALF_MATHEMATICAL_FUNCTIONS_HPP_

#include <impl/Kokkos_Half_FloatingPointWrapper.hpp>

namespace Kokkos {
namespace Impl {
#ifdef KOKKOS_IMPL_HALF_TYPE_STANDARD_SUPPORT

#define KOKKOS_STD_HALF_UNARY_FUNCTION(OP)               \
  KOKKOS_INLINE_FUNCTION Experimental::half_t impl_##OP( \
      Experimental::half_t x) {                          \
    return std::OP(Experimental::half_t::impl_type(x));  \
  }

#define KOKKOS_STD_HALF_BINARY_FUNCTION(OP)              \
  KOKKOS_INLINE_FUNCTION Experimental::half_t impl_##OP( \
      Experimental::half_t x, Experimental::half_t y) {  \
    return static_cast<Experimental::half_t>(            \
        std::OP(Experimental::half_t::impl_type(x),      \
                Experimental::half_t::impl_type(y)));    \
  }

#define KOKKOS_STD_HALF_TERNARY_INT_PTR_FUNCTION(OP)            \
  KOKKOS_INLINE_FUNCTION Experimental::half_t impl_##OP(        \
      Experimental::half_t x, Experimental::half_t y, int* z) { \
    return static_cast<Experimental::half_t>(                   \
        std::OP(Experimental::half_t::impl_type(x),             \
                Experimental::half_t::impl_type(y), z));        \
  }

#define KOKKOS_STD_HALF_UNARY_PREDICATE(OP)                       \
  KOKKOS_INLINE_FUNCTION bool impl_##OP(Experimental::half_t x) { \
    return std::OP(Experimental::half_t::impl_type(x));           \
  }

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::half_t impl_test_fallback_half(
    Kokkos::Experimental::half_t) {
  return Kokkos::Experimental::half_t(0.f);
}

#else
#define KOKKOS_STD_HALF_UNARY_FUNCTION(OP)
#define KOKKOS_STD_HALF_BINARY_FUNCTION(OP)
#define KOKKOS_STD_HALF_TERNARY_INT_PTR_FUNCTION(OP)
#define KOKKOS_STD_HALF_UNARY_PREDICATE(OP)
#endif

#ifdef KOKKOS_IMPL_BHALF_TYPE_STANDARD_SUPPORT

#define KOKKOS_STD_BHALF_UNARY_FUNCTION(OP)               \
  KOKKOS_INLINE_FUNCTION Experimental::bhalf_t impl_##OP( \
      Experimental::bhalf_t x) {                          \
    return std::OP(Experimental::bhalf_t::impl_type(x));  \
  }

#define KOKKOS_STD_BHALF_BINARY_FUNCTION(OP)              \
  KOKKOS_INLINE_FUNCTION Experimental::bhalf_t impl_##OP( \
      Experimental::bhalf_t x, Experimental::bhalf_t y) { \
    return static_cast<Experimental::bhalf_t>(            \
        std::OP(Experimental::bhalf_t::impl_type(x),      \
                Experimental::bhalf_t::impl_type(y)));    \
  }

#define KOKKOS_STD_BHALF_TERNARY_INT_PTR_FUNCTION(OP)             \
  KOKKOS_INLINE_FUNCTION Experimental::bhalf_t impl_##OP(         \
      Experimental::bhalf_t x, Experimental::bhalf_t y, int* z) { \
    return static_cast<Experimental::bhalf_t>(                    \
        std::OP(Experimental::bhalf_t::impl_type(x),              \
                Experimental::bhalf_t::impl_type(y), z));         \
  }

#define KOKKOS_STD_BHALF_UNARY_PREDICATE(OP)                       \
  KOKKOS_INLINE_FUNCTION bool impl_##OP(Experimental::bhalf_t x) { \
    return std::OP(Experimental::bhalf_t::impl_type(x));           \
  }

KOKKOS_INLINE_FUNCTION Kokkos::Experimental::bhalf_t impl_test_fallback_bhalf(
    Kokkos::Experimental::bhalf_t) {
  return Kokkos::Experimental::bhalf_t(0.f);
}
#else
#define KOKKOS_STD_BHALF_UNARY_FUNCTION(OP)
#define KOKKOS_STD_BHALF_BINARY_FUNCTION(OP)
#define KOKKOS_STD_BHALF_TERNARY_INT_PTR_FUNCTION(OP)
#define KOKKOS_STD_BHALF_UNARY_PREDICATE(OP)
#endif

#define KOKKOS_STD_UNARY_FUNCTION(OP) \
  KOKKOS_STD_HALF_UNARY_FUNCTION(OP) KOKKOS_STD_BHALF_UNARY_FUNCTION(OP)
#define KOKKOS_STD_BINARY_FUNCTION(OP) \
  KOKKOS_STD_HALF_BINARY_FUNCTION(OP) KOKKOS_STD_BHALF_BINARY_FUNCTION(OP)
#define KOKKOS_STD_TERNARY_INT_PTR_FUNCTION(OP) \
  KOKKOS_STD_HALF_TERNARY_INT_PTR_FUNCTION(OP)  \
  KOKKOS_STD_BHALF_TERNARY_INT_PTR_FUNCTION(OP)
#define KOKKOS_STD_UNARY_PREDICATE(OP) \
  KOKKOS_STD_HALF_UNARY_PREDICATE(OP) KOKKOS_STD_BHALF_UNARY_PREDICATE(OP)

// Basic operations
// abs
KOKKOS_STD_UNARY_FUNCTION(fabs)
KOKKOS_STD_BINARY_FUNCTION(fmod)
KOKKOS_STD_BINARY_FUNCTION(remainder)
KOKKOS_STD_BINARY_FUNCTION(fmax)
KOKKOS_STD_BINARY_FUNCTION(fmin)
KOKKOS_STD_BINARY_FUNCTION(fdim)
KOKKOS_STD_TERNARY_INT_PTR_FUNCTION(remquo)
// Exponential functions
KOKKOS_STD_UNARY_FUNCTION(exp)
KOKKOS_STD_UNARY_FUNCTION(exp2)
KOKKOS_STD_UNARY_FUNCTION(expm1)
KOKKOS_STD_UNARY_FUNCTION(log)
KOKKOS_STD_UNARY_FUNCTION(log10)
KOKKOS_STD_UNARY_FUNCTION(log2)
KOKKOS_STD_UNARY_FUNCTION(log1p)
// Power functions
KOKKOS_STD_BINARY_FUNCTION(pow)
KOKKOS_STD_UNARY_FUNCTION(sqrt)
KOKKOS_STD_UNARY_FUNCTION(cbrt)
KOKKOS_STD_BINARY_FUNCTION(hypot)
// Trigonometric functions
KOKKOS_STD_UNARY_FUNCTION(sin)
KOKKOS_STD_UNARY_FUNCTION(cos)
KOKKOS_STD_UNARY_FUNCTION(tan)
KOKKOS_STD_UNARY_FUNCTION(asin)
KOKKOS_STD_UNARY_FUNCTION(acos)
KOKKOS_STD_UNARY_FUNCTION(atan)
KOKKOS_STD_BINARY_FUNCTION(atan2)
// Hyperbolic functions
KOKKOS_STD_UNARY_FUNCTION(sinh)
KOKKOS_STD_UNARY_FUNCTION(cosh)
KOKKOS_STD_UNARY_FUNCTION(tanh)
KOKKOS_STD_UNARY_FUNCTION(asinh)
KOKKOS_STD_UNARY_FUNCTION(acosh)
KOKKOS_STD_UNARY_FUNCTION(atanh)
// Error and gamma functions
KOKKOS_STD_UNARY_FUNCTION(erf)
KOKKOS_STD_UNARY_FUNCTION(erfc)
KOKKOS_STD_UNARY_FUNCTION(tgamma)
KOKKOS_STD_UNARY_FUNCTION(lgamma)
// Nearest integer floating point functions
KOKKOS_STD_UNARY_FUNCTION(ceil)
KOKKOS_STD_UNARY_FUNCTION(floor)
KOKKOS_STD_UNARY_FUNCTION(trunc)
KOKKOS_STD_UNARY_FUNCTION(round)
KOKKOS_STD_UNARY_FUNCTION(nearbyint)
KOKKOS_STD_UNARY_FUNCTION(rint)
KOKKOS_STD_UNARY_FUNCTION(logb)
KOKKOS_STD_HALF_BINARY_FUNCTION(nextafter)
KOKKOS_STD_HALF_BINARY_FUNCTION(copysign)
KOKKOS_STD_HALF_UNARY_PREDICATE(isfinite)
KOKKOS_STD_HALF_UNARY_PREDICATE(isinf)
KOKKOS_STD_HALF_UNARY_PREDICATE(isnan)
KOKKOS_STD_HALF_UNARY_PREDICATE(signbit)
// Non-standard functions
// KOKKOS_STD_HALF_UNARY_FUNCTION(rsqrt)

#undef KOKKOS_STD_HALF_UNARY_FUNCTION
#undef KOKKOS_STD_HALF_BINARY_FUNCTION
#undef KOKKOS_STD_HALF_TERNARY_INT_PTR_FUNCTION
#undef KOKKOS_STD_HALF_UNARY_PREDICATE
}  // namespace Impl
}  // namespace Kokkos
#endif
