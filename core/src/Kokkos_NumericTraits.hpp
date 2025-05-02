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
#include <type_traits>
#include <limits>

namespace Kokkos::Experimental {
namespace Impl {

#define KOKKOS_IMPL_DEFINE_TRAITS_HELPER(TRAIT, STD_TRAIT, RETURN_TYPE,     \
                                         CONSTRAINT)                        \
  template <class T, class Enable = void>                                   \
  struct TRAIT##_helper {};                                                 \
  template <class T>                                                        \
  struct TRAIT##_helper<T, std::enable_if_t<std::is_##CONSTRAINT##_v<T>>> { \
    static constexpr RETURN_TYPE value = std::numeric_limits<T>::STD_TRAIT; \
  };

KOKKOS_IMPL_DEFINE_TRAITS_HELPER(infinity, infinity(), T, floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(finite_min, lowest(), T, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(finite_max, max(), T, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(epsilon, epsilon(), T, floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(round_error, round_error(), T, floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(norm_min, min(), T, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(denorm_min, denorm_min(), T, floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(quiet_NaN, quiet_NaN(), T, floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(signaling_NaN, signaling_NaN(), T,
                                 floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(digits, digits, int, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(digits10, digits10, int, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(max_digits10, max_digits10, int,
                                 floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(radix, radix, int, arithmetic)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(min_exponent, min_exponent, int,
                                 floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(min_exponent10, min_exponent10, int,
                                 floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(max_exponent, max_exponent, int,
                                 floating_point)
KOKKOS_IMPL_DEFINE_TRAITS_HELPER(max_exponent10, max_exponent10, int,
                                 floating_point)

#undef KOKKOS_IMPL_DEFINE_TRAITS_HELPER

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
