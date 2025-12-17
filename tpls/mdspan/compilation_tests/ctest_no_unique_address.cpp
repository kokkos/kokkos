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
  sizeof(Kokkos::extents<size_t,1, 2, Kokkos::dynamic_extent>) == sizeof(ptrdiff_t)
);

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::extents<size_t,Kokkos::dynamic_extent>) == sizeof(ptrdiff_t)
);

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::extents<size_t,Kokkos::dynamic_extent, Kokkos::dynamic_extent>) == 2 * sizeof(ptrdiff_t)
);

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::extents<size_t,Kokkos::dynamic_extent, 1, 2, 45>) == sizeof(ptrdiff_t)
);

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::extents<size_t,45, Kokkos::dynamic_extent, 1>) == sizeof(ptrdiff_t)
);

#ifdef MDSPAN_IMPL_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS
MDSPAN_STATIC_TEST(
  std::is_empty<Kokkos::extents<size_t,1, 2, 3>>::value
);

MDSPAN_STATIC_TEST(
  std::is_empty<Kokkos::extents<size_t,42>>::value
);
#endif

// </editor-fold> end extents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="layouts"> {{{1

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::layout_left::template mapping<
    Kokkos::extents<size_t,42, Kokkos::dynamic_extent, 73>
  >) == sizeof(size_t)
);

#if defined(MDSPAN_IMPL_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS) && !defined(MDSPAN_IMPL_USE_FAKE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
MDSPAN_STATIC_TEST(
  std::is_empty<Kokkos::layout_right::template mapping<
    Kokkos::extents<size_t,42, 123, 73>
  >>::value
);
#endif

MDSPAN_STATIC_TEST(
  sizeof(Kokkos::layout_stride::template mapping<
    Kokkos::extents<size_t,42, Kokkos::dynamic_extent, 73>
  >) == 4 * sizeof(size_t)
);


// </editor-fold> end layouts }}}1
//==============================================================================
