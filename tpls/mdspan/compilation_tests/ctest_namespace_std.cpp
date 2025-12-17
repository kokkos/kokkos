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

// These need to be ordered first or the test, well, fails
#include <experimental/mdspan>
#include <experimental/mdarray>

#include "ctest_common.hpp"

using test_extents_type = std::extents<int, std::dynamic_extent>;
using test_mdspan_type = std::mdspan<double, test_extents_type>;
using test_mdarray_type = std::experimental::mdarray<double, test_extents_type>;
