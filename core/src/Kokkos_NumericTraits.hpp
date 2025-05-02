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

#ifndef KOKKOS_NUMERIC_TRAITS_HPP
#define KOKKOS_NUMERIC_TRAITS_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_NUMERIC_TRAITS
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
#include <Kokkos_ReductionIdentity.hpp>
#endif
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace Kokkos::Experimental {
namespace Impl {
// clang-format off
template <class T> struct infinity_helper       { static constexpr T value  = std::numeric_limits<T>::infinity();      };
template <class T> struct finite_min_helper     { static constexpr T value  = std::numeric_limits<T>::lowest();        };
template <class T> struct finite_max_helper     { static constexpr T value  = std::numeric_limits<T>::max();           };
template <class T> struct epsilon_helper        { static constexpr T value  = std::numeric_limits<T>::epsilon();       };
template <class T> struct round_error_helper    { static constexpr T value  = std::numeric_limits<T>::round_error();   };
template <class T> struct norm_min_helper       { static constexpr T value  = std::numeric_limits<T>::min();           };
template <class T> struct denorm_min_helper     { static constexpr T value  = std::numeric_limits<T>::denorm_min();    };
template <class T> struct quiet_NaN_helper      { static constexpr T value  = std::numeric_limits<T>::quiet_NaN();     };
template <class T> struct signaling_NaN_helper  { static constexpr T value  = std::numeric_limits<T>::signaling_NaN(); };
template <class T> struct digits_helper         { static constexpr int value  = std::numeric_limits<T>::digits;         };
template <class T> struct digits10_helper       { static constexpr int value  = std::numeric_limits<T>::digits10;       };
template <class T> struct max_digits10_helper   { static constexpr int value  = std::numeric_limits<T>::max_digits10;   };
template <class T> struct radix_helper          { static constexpr int value  = std::numeric_limits<T>::radix;          };
template <class T> struct min_exponent_helper   { static constexpr int value  = std::numeric_limits<T>::min_exponent;   };
template <class T> struct min_exponent10_helper { static constexpr int value  = std::numeric_limits<T>::min_exponent10; };
template <class T> struct max_exponent_helper   { static constexpr int value  = std::numeric_limits<T>::max_exponent;   };
template <class T> struct max_exponent10_helper { static constexpr int value  = std::numeric_limits<T>::max_exponent10; };
// clang-format on
}  // namespace Impl

#define KOKKOS_IMPL_DEFINE_TRAIT(TRAIT)                        \
  template <class T>                                           \
  struct TRAIT : Impl::TRAIT##_helper<std::remove_cv_t<T>> {}; \
  template <class T>                                           \
  inline constexpr auto TRAIT##_v = TRAIT<T>::value;

// Numeric distinguished value traits
KOKKOS_IMPL_DEFINE_TRAIT(infinity)
KOKKOS_IMPL_DEFINE_TRAIT(finite_min)
KOKKOS_IMPL_DEFINE_TRAIT(finite_max)
KOKKOS_IMPL_DEFINE_TRAIT(epsilon)
KOKKOS_IMPL_DEFINE_TRAIT(round_error)
KOKKOS_IMPL_DEFINE_TRAIT(norm_min)
KOKKOS_IMPL_DEFINE_TRAIT(denorm_min)
KOKKOS_IMPL_DEFINE_TRAIT(quiet_NaN)
KOKKOS_IMPL_DEFINE_TRAIT(signaling_NaN)

// Numeric characteristics traits
KOKKOS_IMPL_DEFINE_TRAIT(digits)
KOKKOS_IMPL_DEFINE_TRAIT(digits10)
KOKKOS_IMPL_DEFINE_TRAIT(max_digits10)
KOKKOS_IMPL_DEFINE_TRAIT(radix)
KOKKOS_IMPL_DEFINE_TRAIT(min_exponent)
KOKKOS_IMPL_DEFINE_TRAIT(min_exponent10)
KOKKOS_IMPL_DEFINE_TRAIT(max_exponent)
KOKKOS_IMPL_DEFINE_TRAIT(max_exponent10)

#undef KOKKOS_IMPL_DEFINE_TRAIT

}  // namespace Kokkos::Experimental

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_NUMERIC_TRAITS
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_NUMERIC_TRAITS
#endif
#endif
