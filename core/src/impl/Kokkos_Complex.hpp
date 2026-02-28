// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOS_IMPL_COMPLEX_HPP
#define KOKKOS_IMPL_COMPLEX_HPP

#include <complex>
#include <concepts>
#include <type_traits>

namespace Kokkos {
template <class RealType>
  requires std::same_as<RealType, std::remove_cv_t<RealType>>
class
#ifdef KOKKOS_ENABLE_COMPLEX_ALIGN
    alignas(2 * sizeof(RealType))
#endif
        complex;

template <typename RealType>
KOKKOS_FUNCTION constexpr RealType norm(const complex<RealType>& z) noexcept;

template <class RealType>
constexpr KOKKOS_INLINE_FUNCTION complex<RealType> conj(
    const complex<RealType>& x) noexcept;
}  // namespace Kokkos

namespace Kokkos::Impl {

/**
 * @brief Naive complex division implementation.
 *
 * References:
 * https://github.com/gcc-mirror/gcc/blob/8758503918a91dacff4dbc7126eced21787fbfc9/libstdc%2B%2B-v3/include/std/complex#L355-L367
 * https://github.com/llvm/llvm-project/blob/5b64aeb409ecd9f8e9ff95ffe5fde7eb6a26146c/libcxx/include/complex#L821-L831
 */
template <typename RealType>
KOKKOS_FUNCTION constexpr complex<RealType> complex_naive_div(
    const complex<RealType>& x, const complex<RealType>& y) noexcept {
  const RealType norm = Kokkos::norm(y);
  return {(x.real() * y.real() + x.imag() * y.imag()) / norm,
          (x.imag() * y.real() - x.real() * y.imag()) / norm};
}

/**
 * @brief Complex division using max-norm scaling to reduce the risk of
 *        overflow.
 *
 * The standard approach to complex division is:
 *   (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
 * where the squared denominator c²+d² can overflow for large but finite
 * inputs (e.g. c = d = 1e154 for double).
 *
 * LLVM specializes this operation, see:
 * https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/clang/lib/CodeGen/CGExprComplex.cpp#L1065-L1079
 * https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/compiler-rt/lib/builtins/divdc3.c#L20
 *
 * They allow a few different ways to treat it, see
 * https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/clang/lib/CodeGen/CGExprComplex.cpp#L1029.
 *
 * GCC also specializes, see
 * https://github.com/gcc-mirror/gcc/blob/8758503918a91dacff4dbc7126eced21787fbfc9/libstdc%2B%2B-v3/include/std/complex#L1734-L1741
 * https://github.com/gcc-mirror/gcc/blob/8758503918a91dacff4dbc7126eced21787fbfc9/gcc/tree-complex.cc#L1352
 *
 * There is the following attempt
 * https://github.com/jtravs/cuda_complex/blob/master/cuda_complex.hpp#L553
 * at porting the algorithm from
 * https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/compiler-rt/lib/builtins/divdc3.c#L20.
 *
 * However, we think it has too many expensive operations and branches
 * to lead to acceptable performance on devices such as GPUs.
 *
 * Instead, this implementation scales the inputs, and therefore has:
 *  * zero branch
 *  * uniform execution across all threads (of a warp)
 *
 * The template parameter @p EnableBranching controls the behavior when
 * the denominator is zero.
 * For performance reasons (branching induces divergence), branching can be
 * disabled.
 */
template <bool EnableBranching, std::floating_point RealType>
KOKKOS_FUNCTION constexpr complex<RealType> complex_scaling_div(
    const complex<RealType>& x, const complex<RealType>& y) noexcept {
  const RealType scale =
      Kokkos::fmax(Kokkos::fabs(y.real()), Kokkos::fabs(y.imag()));
  if constexpr (EnableBranching) {
    if (scale == RealType(0)) {
      return {x.real() / scale, x.imag() / scale};
    }
  }
  const complex<RealType> x_scaled      = x / scale;
  const complex<RealType> y_conj_scaled = Kokkos::conj(y) / scale;
  const RealType y_conj_scaled_norm     = Kokkos::norm(y_conj_scaled);
  return (x_scaled * y_conj_scaled) / y_conj_scaled_norm;
}

template <std::floating_point RealType>
KOKKOS_FUNCTION constexpr complex<RealType> complex_div_choice(
    const complex<RealType>& x, const complex<RealType>& y) noexcept {
#if !defined(KOKKOS_ENABLE_COMPLEX_OVERFLOW_GUARD)
  return complex_naive_div(x, y);
#else
#if defined(KOKKOS_ENABLE_IMPL_COMPLEX_OVERFLOW_GUARD_ZERO_BRANCH)
  return complex_scaling_div<true>(x, y);
#else
  return complex_scaling_div<false>(x, y);
#endif
#endif
}

template <std::floating_point RealType>
KOKKOS_FUNCTION constexpr complex<RealType> complex_div(
    const complex<RealType>& x, const complex<RealType>& y) noexcept {
#if defined(KOKKOS_ENABLE_COMPLEX_ON_HOST_NATIVE)
  KOKKOS_IF_ON_HOST(
      (const auto tmp = std::complex<RealType>{x.real(), x.imag()} /
                        std::complex<RealType>{y.real(), y.imag()};
       return {tmp.real(), tmp.imag()};))
#else
  KOKKOS_IF_ON_HOST(return complex_div_choice(x, y);)
#endif

  KOKKOS_IF_ON_DEVICE(return complex_div_choice(x, y);)
}

// Need to scale both numerator and denormintor to avoid over and underflow
template <typename T>
KOKKOS_FUNCTION complex<T> complex_iec559_div(const complex<T>& x,
                                              const complex<T>& y) {
  int __ilogbw = 0;
  T __a        = x.real();
  T __b        = x.imag();
  T __c        = y.real();
  T __d        = y.imag();
  T __logbw = Kokkos::logb(Kokkos::fmax(Kokkos::fabs(__c), Kokkos::fabs(__d)));
  if (Kokkos::isfinite(__logbw)) {
    __ilogbw = static_cast<int>(__logbw);
    // Scale all four components by the same factor so the ratio is preserved
    // exactly, and cross-products a*d / b*d no longer underflow when a,b are
    // subnormal and the denominator exponent is very negative.
    __a = Kokkos::scalbn(__a, -__ilogbw);
    __b = Kokkos::scalbn(__b, -__ilogbw);
    __c = Kokkos::scalbn(__c, -__ilogbw);
    __d = Kokkos::scalbn(__d, -__ilogbw);
  }
  T __denom = __c * __c + __d * __d;
  // No scalbn needed: ilogbw cancelled out in numerator and denominator.
  T __x = (__a * __c + __b * __d) / __denom;
  T __y = (__b * __c - __a * __d) / __denom;
  if (Kokkos::isnan(__x) && Kokkos::isnan(__y)) {
    if ((__denom == T(0)) && (!Kokkos::isnan(__a) || !Kokkos::isnan(__b))) {
      __x = Kokkos::copysign(T(INFINITY), __c) * __a;
      __y = Kokkos::copysign(T(INFINITY), __c) * __b;
    } else if ((Kokkos::isinf(__a) || Kokkos::isinf(__b)) &&
               Kokkos::isfinite(__c) && Kokkos::isfinite(__d)) {
      __a = Kokkos::copysign(Kokkos::isinf(__a) ? T(1) : T(0), __a);
      __b = Kokkos::copysign(Kokkos::isinf(__b) ? T(1) : T(0), __b);
      __x = T(INFINITY) * (__a * __c + __b * __d);
      __y = T(INFINITY) * (__b * __c - __a * __d);
    } else if (Kokkos::isinf(__logbw) && __logbw > T(0) &&
               Kokkos::isfinite(x.real()) && Kokkos::isfinite(x.imag())) {
      __c = Kokkos::copysign(Kokkos::isinf(__c) ? T(1) : T(0), __c);
      __d = Kokkos::copysign(Kokkos::isinf(__d) ? T(1) : T(0), __d);
      __x = T(0) * (__a * __c + __b * __d);
      __y = T(0) * (__b * __c - __a * __d);
    }
  }
  return {__x, __y};
}

/**
 * Adapted from
 * https://github.com/jtravs/cuda_complex/blob/master/cuda_complex.hpp#L553,
 * similar to
 * https://github.com/llvm/llvm-project/blob/2342db00ab4d0305580814fb00f477b4b5cebec6/compiler-rt/lib/builtins/divdc3.c#L20.
 *
 * Though it's cheaper on gpus, it seems to fail more often then the scaling.
 */
template <typename T>
KOKKOS_FUNCTION complex<T> complex_iec559_div_old(const complex<T>& x,
                                              const complex<T>& y) {
  int __ilogbw = 0;
  T __a        = x.real();
  T __b        = x.imag();
  T __c        = y.real();
  T __d        = y.imag();
  T __logbw = Kokkos::logb(Kokkos::fmax(Kokkos::fabs(__c), Kokkos::fabs(__d)));
  if (Kokkos::isfinite(__logbw)) {
    __ilogbw = static_cast<int>(__logbw);
    __c      = Kokkos::scalbn(__c, -__ilogbw);
    __d      = Kokkos::scalbn(__d, -__ilogbw);
  }
  T __denom = __c * __c + __d * __d;
  T __x     = Kokkos::scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
  T __y     = Kokkos::scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (Kokkos::isnan(__x) && Kokkos::isnan(__y)) {
    if ((__denom == T(0)) && (!Kokkos::isnan(__a) || !Kokkos::isnan(__b))) {
      __x = Kokkos::copysign(T(INFINITY), __c) * __a;
      __y = Kokkos::copysign(T(INFINITY), __c) * __b;
    } else if ((Kokkos::isinf(__a) || Kokkos::isinf(__b)) &&
               Kokkos::isfinite(__c) && Kokkos::isfinite(__d)) {
      __a = Kokkos::copysign(Kokkos::isinf(__a) ? T(1) : T(0), __a);
      __b = Kokkos::copysign(Kokkos::isinf(__b) ? T(1) : T(0), __b);
      __x = T(INFINITY) * (__a * __c + __b * __d);
      __y = T(INFINITY) * (__b * __c - __a * __d);
    } else if (Kokkos::isinf(__logbw) && __logbw > T(0) &&
               Kokkos::isfinite(__a) && Kokkos::isfinite(__b)) {
      __c = Kokkos::copysign(Kokkos::isinf(__c) ? T(1) : T(0), __c);
      __d = Kokkos::copysign(Kokkos::isinf(__d) ? T(1) : T(0), __d);
      __x = T(0) * (__a * __c + __b * __d);
      __y = T(0) * (__b * __c - __a * __d);
    }
  }
  return {__x, __y};
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_COMPLEX_HPP
