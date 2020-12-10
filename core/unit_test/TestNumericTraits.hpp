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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include "Kokkos_ArithmeticTraits.hpp"
#include "Kokkos_ExecPolicy.hpp"

struct extrema {
#define DEFINE_EXTREMA(T, m, M)                 \
  KOKKOS_FUNCTION static T min(T) { return m; } \
  KOKKOS_FUNCTION static T max(T) { return M; }

  DEFINE_EXTREMA(char, CHAR_MIN, CHAR_MAX);
  DEFINE_EXTREMA(signed char, SCHAR_MIN, SCHAR_MAX);
  DEFINE_EXTREMA(unsigned char, 0, UCHAR_MAX);
  DEFINE_EXTREMA(short, SHRT_MIN, SHRT_MAX);
  DEFINE_EXTREMA(unsigned short, 0, USHRT_MAX);
  DEFINE_EXTREMA(int, INT_MIN, INT_MAX);
  DEFINE_EXTREMA(unsigned, 0U, UINT_MAX);
  DEFINE_EXTREMA(long, LONG_MIN, LONG_MAX);
  DEFINE_EXTREMA(unsigned long, 0UL, ULONG_MAX);
  DEFINE_EXTREMA(long long, LLONG_MIN, LLONG_MAX);
  DEFINE_EXTREMA(unsigned long long, 0ULL, ULLONG_MAX);

  DEFINE_EXTREMA(float, -FLT_MAX, FLT_MAX);
  DEFINE_EXTREMA(double, -DBL_MAX, DBL_MAX);
  DEFINE_EXTREMA(long double, -LDBL_MAX, LDBL_MAX);

#undef DEFINE_EXTREMA
};

struct Infinity {};
struct Epsilon {};
struct FiniteMinMax {};
struct RoundError {};
struct NormMin {};
struct Digits {};
struct Digits10 {};
struct MaxDigits10 {};
struct Radix {};
struct MinMaxExponent {};
struct MinMaxExponent10 {};

template <class T>
KOKKOS_FUNCTION T* take_address_of(T& arg) {
  return &arg;
}

template <class Space, class T, class Tag>
struct TestNumericTraits {
  TestNumericTraits() { run(); }

  void run() const {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space, Tag>(0, 1), *this,
                            errors);
    ASSERT_EQ(errors, 0);
  }

  KOKKOS_FUNCTION void operator()(Infinity, int, int& e) const {
    using Kokkos::Experimental::infinity;
    auto const inf  = infinity<T>::value;
    auto const zero = T(0);
    e += (int)!(inf + inf == inf);
    e += (int)!(inf != zero);
    (void)take_address_of(infinity<T>::value);
  }

  KOKKOS_FUNCTION void operator()(Epsilon, int, int& e) const {
    using Kokkos::Experimental::epsilon;
    auto const eps = epsilon<T>::value;
    auto const one = T(1);
    e += (int)!(one + eps != one);
    e += (int)!(one + eps / 2 == one);
    (void)take_address_of(epsilon<T>::value);
  }

  KOKKOS_FUNCTION void operator()(FiniteMinMax, int, int& e) const {
    using Kokkos::Experimental::finite_max;
    using Kokkos::Experimental::finite_min;
    auto const min = finite_min<T>::value;
    auto const max = finite_max<T>::value;
    e += (int)!(min == extrema::min(T{}));
    e += (int)!(max == extrema::max(T{}));
    (void)take_address_of(finite_min<T>::value);
    (void)take_address_of(finite_max<T>::value);
  }

  KOKKOS_FUNCTION void operator()(RoundError, int, int&) const {
    using Kokkos::Experimental::round_error;
    (void)take_address_of(round_error<T>::value);
  }

  KOKKOS_FUNCTION void operator()(NormMin, int, int&) const {
    using Kokkos::Experimental::norm_min;
    (void)take_address_of(norm_min<T>::value);
  }

  KOKKOS_FUNCTION void operator()(Digits, int, int&) const {
    using Kokkos::Experimental::digits;
    (void)take_address_of(digits<T>::value);
  }

  KOKKOS_FUNCTION void operator()(Digits10, int, int&) const {
    using Kokkos::Experimental::digits10;
    (void)take_address_of(digits10<T>::value);
  }

  KOKKOS_FUNCTION void operator()(MaxDigits10, int, int&) const {
    using Kokkos::Experimental::max_digits10;
    (void)take_address_of(max_digits10<T>::value);
  }

  KOKKOS_FUNCTION void operator()(Radix, int, int&) const {
    using Kokkos::Experimental::radix;
    (void)take_address_of(radix<T>::value);
  }

  KOKKOS_FUNCTION void operator()(MinMaxExponent, int, int&) const {
    using Kokkos::Experimental::max_exponent;
    using Kokkos::Experimental::min_exponent;
    (void)take_address_of(min_exponent<T>::value);
    (void)take_address_of(max_exponent<T>::value);
  }

  KOKKOS_FUNCTION void operator()(MinMaxExponent10, int, int&) const {
    using Kokkos::Experimental::max_exponent10;
    using Kokkos::Experimental::min_exponent10;
    (void)take_address_of(min_exponent10<T>::value);
    (void)take_address_of(max_exponent10<T>::value);
  }
};

TEST(TEST_CATEGORY, numeric_traits_infinity) {
  TestNumericTraits<TEST_EXECSPACE, float, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, double, Infinity>();
  TestNumericTraits<TEST_EXECSPACE, long double, Infinity>();
}

TEST(TEST_CATEGORY, numeric_traits_epsilon) {
  TestNumericTraits<TEST_EXECSPACE, float, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, double, Epsilon>();
  TestNumericTraits<TEST_EXECSPACE, long double, Epsilon>();
}

TEST(TEST_CATEGORY, numeric_traits_round_error) {
  TestNumericTraits<TEST_EXECSPACE, float, RoundError>();
  TestNumericTraits<TEST_EXECSPACE, double, RoundError>();
  TestNumericTraits<TEST_EXECSPACE, long double, RoundError>();
}

TEST(TEST_CATEGORY, numeric_traits_norm_min) {
  TestNumericTraits<TEST_EXECSPACE, float, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, double, NormMin>();
  TestNumericTraits<TEST_EXECSPACE, long double, NormMin>();
}

TEST(TEST_CATEGORY, numeric_traits_finite_min_max) {
  TestNumericTraits<TEST_EXECSPACE, char, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, signed char, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, FiniteMinMax>();

  TestNumericTraits<TEST_EXECSPACE, short, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, FiniteMinMax>();

  TestNumericTraits<TEST_EXECSPACE, int, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, FiniteMinMax>();

  TestNumericTraits<TEST_EXECSPACE, long, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long, FiniteMinMax>();

  TestNumericTraits<TEST_EXECSPACE, long long, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long, FiniteMinMax>();

  TestNumericTraits<TEST_EXECSPACE, float, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, double, FiniteMinMax>();
  TestNumericTraits<TEST_EXECSPACE, long double, FiniteMinMax>();
}

TEST(TEST_CATEGORY, numeric_traits_digits) {
  TestNumericTraits<TEST_EXECSPACE, bool, Digits>();
  TestNumericTraits<TEST_EXECSPACE, char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Digits>();
  TestNumericTraits<TEST_EXECSPACE, short, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Digits>();
  TestNumericTraits<TEST_EXECSPACE, int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Digits>();
  TestNumericTraits<TEST_EXECSPACE, float, Digits>();
  TestNumericTraits<TEST_EXECSPACE, double, Digits>();
  TestNumericTraits<TEST_EXECSPACE, long double, Digits>();
}

TEST(TEST_CATEGORY, numeric_traits_digits10) {
  TestNumericTraits<TEST_EXECSPACE, bool, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, short, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, float, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, double, Digits10>();
  TestNumericTraits<TEST_EXECSPACE, long double, Digits10>();
}

TEST(TEST_CATEGORY, numeric_traits_max_digits10) {
  TestNumericTraits<TEST_EXECSPACE, float, MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, double, MaxDigits10>();
  TestNumericTraits<TEST_EXECSPACE, long double, MaxDigits10>();
}

TEST(TEST_CATEGORY, numeric_traits_radix) {
  TestNumericTraits<TEST_EXECSPACE, bool, Radix>();
  TestNumericTraits<TEST_EXECSPACE, char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, signed char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned char, Radix>();
  TestNumericTraits<TEST_EXECSPACE, short, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned short, Radix>();
  TestNumericTraits<TEST_EXECSPACE, int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, unsigned long long int, Radix>();
  TestNumericTraits<TEST_EXECSPACE, float, Radix>();
  TestNumericTraits<TEST_EXECSPACE, double, Radix>();
  TestNumericTraits<TEST_EXECSPACE, long double, Radix>();
}

TEST(TEST_CATEGORY, numeric_traits_min_max_exponent) {
  TestNumericTraits<TEST_EXECSPACE, float, MinMaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, double, MinMaxExponent>();
  TestNumericTraits<TEST_EXECSPACE, long double, MinMaxExponent>();
}

TEST(TEST_CATEGORY, numeric_traits_min_max_exponent10) {
  TestNumericTraits<TEST_EXECSPACE, float, MinMaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, double, MinMaxExponent10>();
  TestNumericTraits<TEST_EXECSPACE, long double, MinMaxExponent10>();
}
