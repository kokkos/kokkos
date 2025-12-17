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

#include <type_traits>



//==============================================================================
// <editor-fold desc="extents"> {{{1

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,1, 2, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,Kokkos::dynamic_extent, 1, 2, 45>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,45, Kokkos::dynamic_extent, 1>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,1, 2, 3>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::extents<size_t,42>
  >::value
);

// </editor-fold> end extents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="layouts"> {{{1

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_left::template mapping<
      Kokkos::extents<size_t,42, Kokkos::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_right::template mapping<
      Kokkos::extents<size_t,42, Kokkos::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_right::template mapping<
      Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_stride::template mapping<
      Kokkos::extents<size_t,42, Kokkos::dynamic_extent, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_stride::template mapping<
      Kokkos::extents<size_t,42, 27, 73>
    >
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::layout_stride::template mapping<
      Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent>
    >
  >::value
);

// TODO: Remove this test alltogether?
// CT: Fails with GCC too after I removed the template parameter
// I guess there is padding added after foo?
#if 0
struct layout_stride_as_member_should_be_standard_layout :
  Kokkos::layout_stride::template mapping<
    Kokkos::extents<size_t,1, 2, 3>>
{
  int foo;
};

// Fails with MSVC which adds some padding
#ifndef MDSPAN_IMPL_COMPILER_MSVC
MDSPAN_STATIC_TEST(
  std::is_standard_layout<layout_stride_as_member_should_be_standard_layout>::value
);
#endif
#endif

// </editor-fold> end layouts }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="mdspan"> {{{1

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::mdspan<double, Kokkos::extents<size_t,1, 2, 3>>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::mdspan<int, Kokkos::dextents<size_t,2>>
  >::value
);

MDSPAN_STATIC_TEST(
  std::is_standard_layout<
    Kokkos::mdspan<
      double, Kokkos::dextents<size_t,2>,
      Kokkos::layout_left, Kokkos::default_accessor<double>
    >
  >::value
);

// </editor-fold> end mdspan }}}1
//==============================================================================
