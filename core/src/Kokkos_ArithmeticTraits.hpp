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

#ifndef KOKKOS_ARITHMETIC_TRAITS_HPP
#define KOKKOS_ARITHMETIC_TRAITS_HPP

#include <Kokkos_Macros.hpp>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace Kokkos {
namespace Experimental {
namespace Impl {
// clang-format off
template <class> struct infinity_helper;
template <> struct infinity_helper<float> { static constexpr float value = HUGE_VALF; };
template <> struct infinity_helper<double> { static constexpr double value = HUGE_VAL; };
template <> struct infinity_helper<long double> { static constexpr long double value = HUGE_VALL; };
template <class> struct finite_min_helper;
template <> struct finite_min_helper<bool> { static constexpr bool value = false; };
template <> struct finite_min_helper<char> { static constexpr char value = CHAR_MIN; };
template <> struct finite_min_helper<unsigned char> { static constexpr unsigned char value = 0; };
template <> struct finite_min_helper<short> { static constexpr short value = SHRT_MIN; };
template <> struct finite_min_helper<unsigned short> { static constexpr unsigned short value = 0; };
template <> struct finite_min_helper<int> { static constexpr int value = INT_MIN; };
template <> struct finite_min_helper<unsigned int> { static constexpr unsigned int value = 0; };
template <> struct finite_min_helper<long int> { static constexpr long int value = LONG_MIN; };
template <> struct finite_min_helper<unsigned long int> { static constexpr unsigned long int value = 0; };
template <> struct finite_min_helper<long long int> { static constexpr long long int value = LONG_MIN; };
template <> struct finite_min_helper<unsigned long long int> { static constexpr unsigned long long int value = 0; };
template <> struct finite_min_helper<float> { static constexpr float value = -FLT_MAX; };
template <> struct finite_min_helper<double> { static constexpr double value = -DBL_MAX; };
template <> struct finite_min_helper<long double> { static constexpr long double value = -LDBL_MAX; };
template <class> struct finite_max_helper;
template <> struct finite_max_helper<bool> { static constexpr bool value = true; };
template <> struct finite_max_helper<char> { static constexpr char value = CHAR_MAX; };
template <> struct finite_max_helper<unsigned char> { static constexpr unsigned char value = UCHAR_MAX; };
template <> struct finite_max_helper<short> { static constexpr short value = SHRT_MAX; };
template <> struct finite_max_helper<unsigned short> { static constexpr unsigned short value = USHRT_MAX; };
template <> struct finite_max_helper<int> { static constexpr int value = INT_MAX; };
template <> struct finite_max_helper<unsigned int> { static constexpr unsigned int value = UINT_MAX; };
template <> struct finite_max_helper<long int> { static constexpr long int value = LONG_MAX; };
template <> struct finite_max_helper<unsigned long int> { static constexpr unsigned long int value = ULONG_MAX; };
template <> struct finite_max_helper<long long int> { static constexpr long long int value = LLONG_MAX; };
template <> struct finite_max_helper<unsigned long long int> { static constexpr unsigned long long int value = ULLONG_MAX; };
template <> struct finite_max_helper<float> { static constexpr float value = FLT_MAX; };
template <> struct finite_max_helper<double> { static constexpr double value = DBL_MAX; };
template <> struct finite_max_helper<long double> { static constexpr long double value = LDBL_MAX; };
template <class> struct epsilon_helper;
template <> struct epsilon_helper<float> { static constexpr float value = FLT_EPSILON; };
template <> struct epsilon_helper<double> { static constexpr double value = DBL_EPSILON; };
template <> struct epsilon_helper<long double> { static constexpr long double value = LDBL_EPSILON; };
template <class> struct round_error_helper;
template <> struct round_error_helper<float> { static constexpr float value = 0.5F; };
template <> struct round_error_helper<double> { static constexpr double value = 0.5; };
template <> struct round_error_helper<long double> { static constexpr long double value = 0.5L; };
template <class> struct norm_min_helper;
template <> struct norm_min_helper<float> { static constexpr float value = FLT_MIN; };
template <> struct norm_min_helper<double> { static constexpr double value = DBL_MIN; };
template <> struct norm_min_helper<long double> { static constexpr long double value = LDBL_MIN; };
template <class> struct digits_helper;
template <> struct digits_helper<bool> { static constexpr int value = 1; };
template <> struct digits_helper<char> { static constexpr int value = CHAR_BIT - std::is_signed<char>::value; };
template <> struct digits_helper<unsigned char> { static constexpr int value = CHAR_BIT; };
template <> struct digits_helper<short> { static constexpr int value = CHAR_BIT*sizeof(short)-1; };
template <> struct digits_helper<unsigned short> { static constexpr int value = CHAR_BIT*sizeof(short); };
template <> struct digits_helper<int> { static constexpr int value = CHAR_BIT*sizeof(int)-1; };
template <> struct digits_helper<unsigned int> { static constexpr int value = CHAR_BIT*sizeof(int); };
template <> struct digits_helper<long int> { static constexpr int value = CHAR_BIT*sizeof(long int)-1; };
template <> struct digits_helper<unsigned long int> { static constexpr int value = CHAR_BIT*sizeof(long int); };
template <> struct digits_helper<long long int> { static constexpr int value = CHAR_BIT*sizeof(long long int)-1; };
template <> struct digits_helper<unsigned long long int> { static constexpr int value = CHAR_BIT*sizeof(long long int); };
template <> struct digits_helper<float> { static constexpr int value = FLT_MANT_DIG; };
template <> struct digits_helper<double> { static constexpr int value = DBL_MANT_DIG; };
template <> struct digits_helper<long double> { static constexpr int value = LDBL_MANT_DIG; };
template <class> struct radix_helper;
template <> struct radix_helper<bool> { static constexpr int value = 2; };
template <> struct radix_helper<char> { static constexpr int value = 2; };
template <> struct radix_helper<unsigned char> { static constexpr int value = 2; };
template <> struct radix_helper<short> { static constexpr int value = 2; };
template <> struct radix_helper<unsigned short> { static constexpr int value = 2; };
template <> struct radix_helper<int> { static constexpr int value = 2; };
template <> struct radix_helper<unsigned int> { static constexpr int value = 2; };
template <> struct radix_helper<long int> { static constexpr int value = 2; };
template <> struct radix_helper<unsigned long int> { static constexpr int value = 2; };
template <> struct radix_helper<long long int> { static constexpr int value = 2; };
template <> struct radix_helper<unsigned long long int> { static constexpr int value = 2; };
template <> struct radix_helper<float> { static constexpr int value = FLT_RADIX; };
template <> struct radix_helper<double> { static constexpr int value = FLT_RADIX; };
template <> struct radix_helper<long double> { static constexpr int value = FLT_RADIX; };
template <class> struct min_exponent_helper;
template <> struct min_exponent_helper<float> { static constexpr int value = FLT_MIN_EXP; };
template <> struct min_exponent_helper<double> { static constexpr int value = DBL_MIN_EXP; };
template <> struct min_exponent_helper<long double> { static constexpr int value = LDBL_MIN_EXP; };
template <class> struct min_exponent10_helper;
template <> struct min_exponent10_helper<float> { static constexpr int value = FLT_MIN_10_EXP; };
template <> struct min_exponent10_helper<double> { static constexpr int value = DBL_MIN_10_EXP; };
template <> struct min_exponent10_helper<long double> { static constexpr int value = LDBL_MIN_10_EXP; };
template <class> struct max_exponent_helper;
template <> struct max_exponent_helper<float> { static constexpr int value = FLT_MAX_EXP; };
template <> struct max_exponent_helper<double> { static constexpr int value = DBL_MAX_EXP; };
template <> struct max_exponent_helper<long double> { static constexpr int value = LDBL_MAX_EXP; };
template <class> struct max_exponent10_helper;
template <> struct max_exponent10_helper<float> { static constexpr int value = FLT_MAX_10_EXP; };
template <> struct max_exponent10_helper<double> { static constexpr int value = DBL_MAX_10_EXP; };
template <> struct max_exponent10_helper<long double> { static constexpr int value = LDBL_MAX_10_EXP; };
// clang-format on
}  // namespace Impl

// Numeric distinguished value traits
template <class T>
struct infinity : Impl::infinity_helper<T> {};
template <class T>
struct finite_min : Impl::finite_min_helper<T> {};
template <class T>
struct finite_max : Impl::finite_max_helper<T> {};
template <class T>
struct epsilon : Impl::epsilon_helper<T> {};
template <class T>
struct round_error : Impl::round_error_helper<T> {};
template <class T>
struct norm_min : Impl::norm_min_helper<T> {};

// Numeric characteristics traits
template <class T>
struct digits : Impl::digits_helper<T> {};
template <class T>
struct radix : Impl::radix_helper<T> {};
template <class T>
struct min_exponent : Impl::min_exponent_helper<T> {};
template <class T>
struct min_exponent10 : Impl::min_exponent10_helper<T> {};
template <class T>
struct max_exponent : Impl::max_exponent_helper<T> {};
template <class T>
struct max_exponent10 : Impl::max_exponent10_helper<T> {};

}  // namespace Experimental
}  // namespace Kokkos

#endif
