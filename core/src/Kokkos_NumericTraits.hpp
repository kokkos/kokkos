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

#define KOKKOS_IMPL_DEFINE_TRAIT(TRAIT, STD_TRAIT, RETURN_TYPE, CONSTRAINT) \
  namespace Impl {                                                          \
  template <class T, class Enable = void>                                   \
  struct TRAIT##_helper {};                                                 \
  template <class T>                                                        \
  struct TRAIT##_helper<T, std::enable_if_t<std::is_##CONSTRAINT##_v<T>>> { \
    static constexpr RETURN_TYPE value = std::numeric_limits<T>::STD_TRAIT; \
  };                                                                        \
  }                                                                         \
  template <class T>                                                        \
  struct TRAIT : Impl::TRAIT##_helper<std::remove_cv_t<T>> {};              \
  template <class T>                                                        \
  inline constexpr auto TRAIT##_v = TRAIT<T>::value;

// clang-format off
// Numeric distinguished value traits
KOKKOS_IMPL_DEFINE_TRAIT(infinity,       infinity(),      T,   floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(finite_min,     lowest(),        T,   arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(finite_max,     max(),           T,   arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(epsilon,        epsilon(),       T,   floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(round_error,    round_error(),   T,   floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(norm_min,       min(),           T,   arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(denorm_min,     denorm_min(),    T,   floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(quiet_NaN,      quiet_NaN(),     T,   floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(signaling_NaN,  signaling_NaN(), T,   floating_point)

// Numeric characteristics traits
KOKKOS_IMPL_DEFINE_TRAIT(digits,         digits,          int, arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(digits10,       digits10,        int, arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(max_digits10,   max_digits10,    int, floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(radix,          radix,           int, arithmetic    )
KOKKOS_IMPL_DEFINE_TRAIT(min_exponent,   min_exponent,    int, floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(min_exponent10, min_exponent10,  int, floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(max_exponent,   max_exponent,    int, floating_point)
KOKKOS_IMPL_DEFINE_TRAIT(max_exponent10, max_exponent10,  int, floating_point)
// clang-format on

#undef KOKKOS_IMPL_DEFINE_TRAIT

}  // namespace Kokkos::Experimental

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_NUMERIC_TRAITS
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_NUMERIC_TRAITS
#endif
#endif
