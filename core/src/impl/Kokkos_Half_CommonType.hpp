// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HALF_COMMON_TYPE_HPP_
#define KOKKOS_HALF_COMMON_TYPE_HPP_

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace std {

#if !KOKKOS_HALF_T_IS_FLOAT
template <>
struct common_type<Kokkos::Experimental::half_t, Kokkos::Experimental::half_t> {
  using type = Kokkos::Experimental::half_t;
};

template <>
struct common_type<Kokkos::Experimental::half_t, float> {
  using type = float;
};

template <>
struct common_type<Kokkos::Experimental::half_t, double> {
  using type = double;
};

template <>
struct common_type<Kokkos::Experimental::half_t, long double> {
  using type = long double;
};

template <typename T>
struct common_type<T, Kokkos::Experimental::half_t>
    : std::common_type<Kokkos::Experimental::half_t, T> {};
#endif

#if !KOKKOS_BHALF_T_IS_FLOAT
template <>
struct common_type<Kokkos::Experimental::bhalf_t,
                   Kokkos::Experimental::bhalf_t> {
  using type = Kokkos::Experimental::bhalf_t;
};

template <>
struct common_type<Kokkos::Experimental::bhalf_t, float> {
  using type = float;
};

template <>
struct common_type<Kokkos::Experimental::bhalf_t, double> {
  using type = double;
};

template <>
struct common_type<Kokkos::Experimental::bhalf_t, long double> {
  using type = long double;
};

template <typename T>
struct common_type<T, Kokkos::Experimental::bhalf_t>
    : std::common_type<Kokkos::Experimental::bhalf_t, T> {};
#endif

#if !KOKKOS_BHALF_T_IS_FLOAT && !KOKKOS_HALF_T_IS_FLOAT
template <>
struct common_type<Kokkos::Experimental::bhalf_t,
                   Kokkos::Experimental::half_t> {
  using type = float;
};

template <>
struct common_type<Kokkos::Experimental::half_t,
                   Kokkos::Experimental::bhalf_t> {
  using type = float;
};
#endif

}  // namespace std

#endif
