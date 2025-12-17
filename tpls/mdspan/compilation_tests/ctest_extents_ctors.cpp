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
#include "ctest_common.hpp"

#include <mdspan/mdspan.hpp>



MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,1, 2, Kokkos::dynamic_extent>,
    int
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_copy_constructible<
    Kokkos::extents<size_t,1, 2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_copy_constructible<
    Kokkos::extents<size_t,1, 2>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_copy_constructible<
    Kokkos::extents<size_t,Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_move_constructible<
    Kokkos::extents<size_t,1, 2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_default_constructible<
    Kokkos::extents<size_t,1, 2, 3>
  >::value
);

// constructibility test from integrals
MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
    int, size_t, int
  >::value
);

MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
    int, int
  >::value
);

MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
    int
  >::value
);

// conversion construction from extents
MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>,
    Kokkos::extents<size_t,Kokkos::dynamic_extent, 2, 3>
  >::value
);


MDSPAN_STATIC_TEST(
  std::is_convertible<
    Kokkos::extents<size_t,2, 3>,
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    Kokkos::extents<size_t,2, 3>
  >::value
);

#if MDSPAN_HAS_CXX_20
// GNU gets the is_convertible with conditional explicit
// wrong in some older versions.
#if !defined(__GNUC__) || (__GNUC__ > 10)
MDSPAN_STATIC_TEST(
  !std::is_convertible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    Kokkos::extents<size_t,2, 3>
  >::value
);
#endif
#endif


MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t>,
    Kokkos::extents<int>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_convertible<
    Kokkos::extents<int>,
    Kokkos::extents<size_t>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<int>,
    Kokkos::extents<size_t>
  >::value
);

#if MDSPAN_HAS_CXX_20
// GNU gets the is_convertible with conditional explicit
// wrong in some older versions.
#if !defined(__GNUC__) || (__GNUC__ > 10)
MDSPAN_STATIC_TEST(
  !std::is_convertible<
    Kokkos::extents<size_t>,
    Kokkos::extents<int>
  >::value
);
#endif
#endif

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::array<int, 1>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_convertible<
    std::array<int, 1>,
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::array<int, 2>
  >::value
);

#if MDSPAN_HAS_CXX_20
// GNU gets the is_convertible with conditional explicit
// wrong in some older versions.
#if !defined(__GNUC__) || (__GNUC__ > 10)
MDSPAN_STATIC_TEST(
  !std::is_convertible<
    std::array<int, 2>,
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>
  >::value
);
#endif
#endif

MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::array<int,3>
  >::value
);

#ifdef __cpp_lib_span
MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::span<int,1>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_convertible<
    std::span<int,1>,
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::span<int,2>
  >::value
);

#if MDSPAN_HAS_CXX_20
// GNU gets the is_convertible with conditional explicit
// wrong in some older versions.
#if !defined(__GNUC__) || (__GNUC__ > 10)
MDSPAN_STATIC_TEST(
  !std::is_convertible<
    std::span<int, 2>,
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>
  >::value
);
#endif
#endif

// this is not supported in the standard
MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::span<int,std::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    std::span<int,3>
  >::value
);
#endif // __cpp_lib_span

MDSPAN_STATIC_TEST(
  !std::is_constructible<
    Kokkos::extents<size_t,3, Kokkos::dynamic_extent>,
    Kokkos::extents<size_t,2, 3>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_assignable<
    Kokkos::extents<size_t,2, Kokkos::dynamic_extent>,
    Kokkos::extents<size_t,2, 3>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_same<
    Kokkos::dextents<size_t,0>,
    Kokkos::extents<size_t>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_same<
    Kokkos::dextents<size_t,1>,
    Kokkos::extents<size_t,Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_same<
    Kokkos::dextents<size_t,2>,
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_same<
    Kokkos::dextents<size_t,3>,
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_constructible<
    Kokkos::extents<size_t, Kokkos::dynamic_extent, 4>,
    int, int
  >::value
);

MDSPAN_STATIC_TEST(
  Kokkos::extents<size_t, Kokkos::dynamic_extent, 4>{ 7, 4 }.extent( 0 ) == 7
);

MDSPAN_STATIC_TEST(
  Kokkos::dextents<size_t, 1>{std::array<std::size_t, 1>{1}}.extent(0) == 1
);

MDSPAN_STATIC_TEST(
  Kokkos::extents<size_t, Kokkos::dynamic_extent, 1>{Kokkos::extents<size_t, 0, 1>{}}.extent(0) == 0
);
