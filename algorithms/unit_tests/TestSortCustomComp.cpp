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

#ifndef KOKKOS_ALGORITHMS_SORTING_CUSTOM_COMP_HPP
#define KOKKOS_ALGORITHMS_SORTING_CUSTOM_COMP_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

struct MyComp{
  KOKKOS_INLINE_FUNCTION
  bool operator()(double const & a, double const & b) const{
    return a < b;
  }
};

TEST(sorting, custom_comp)
{
  std::cout << "GIGIG\n";

  using view_t = Kokkos::View<double*>;
  view_t v{"v", 5};
  auto f = Kokkos::Experimental::begin(v);
  auto l = Kokkos::Experimental::end(v);
  v[0] = 15.;
  v[1] = 5.;
  v[2] = -15.;
  v[3] = 45.;
  v[4] = 115.;
  std::for_each(f, l, [](double val){ std::cout << val << " "; });
  std::cout << "\n";
  Kokkos::sort(v, MyComp{});
  Kokkos::fence();
  std::for_each(f, l, [](double val){ std::cout << val << " "; });
}

#endif
