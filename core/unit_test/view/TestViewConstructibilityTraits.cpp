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

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace {
// Relevant documentation:
// https://kokkos.org/kokkos-core-wiki/API/core/view/view.html#assignment-rules
// Not everything will be tested here as this is just testing the
// `std::is_constructible_v` trait

// Differing rank not allowed
static_assert(
    !std::is_constructible_v<Kokkos::View<double **>, Kokkos::View<double *>>);
static_assert(
    !std::is_assignable_v<Kokkos::View<double **> &, Kokkos::View<double *>>);

// Differing non const value type
static_assert(
    !std::is_constructible_v<Kokkos::View<double *>, Kokkos::View<int *>>);
static_assert(
    !std::is_assignable_v<Kokkos::View<double *> &, Kokkos::View<int *>>);

// Don't construct away constness
static_assert(!std::is_constructible_v<Kokkos::View<double *>,
                                       Kokkos::View<const double *>>);
static_assert(!std::is_assignable_v<Kokkos::View<double *> &,
                                    Kokkos::View<const double *>>);

// Construct to const is ok
static_assert(std::is_constructible_v<Kokkos::View<const double *>,
                                      Kokkos::View<double *>>);
static_assert(std::is_assignable_v<Kokkos::View<const double *> &,
                                   Kokkos::View<double *>>);

// Differing static extents
static_assert(
    !std::is_constructible_v<Kokkos::View<double[3]>, Kokkos::View<double[4]>>);
static_assert(
    !std::is_assignable_v<Kokkos::View<double[3]> &, Kokkos::View<double[4]>>);

// Differing layouts:
// Single rank is OK
static_assert(
    std::is_constructible_v<Kokkos::View<double[3], Kokkos::LayoutLeft>,
                            Kokkos::View<double[3], Kokkos::LayoutRight>>);
static_assert(
    std::is_constructible_v<Kokkos::View<double[3], Kokkos::LayoutRight>,
                            Kokkos::View<double[3], Kokkos::LayoutLeft>>);
static_assert(
    std::is_assignable_v<Kokkos::View<double[3], Kokkos::LayoutLeft> &,
                         Kokkos::View<double[3], Kokkos::LayoutRight>>);
static_assert(
    std::is_assignable_v<Kokkos::View<double[3], Kokkos::LayoutRight> &,
                         Kokkos::View<double[3], Kokkos::LayoutLeft>>);

// Rank > 1 not ok
static_assert(
    !std::is_constructible_v<Kokkos::View<double[3][5], Kokkos::LayoutRight>,
                             Kokkos::View<double[3][5], Kokkos::LayoutLeft>>);
static_assert(
    !std::is_constructible_v<Kokkos::View<double[3][5], Kokkos::LayoutLeft>,
                             Kokkos::View<double[3][5], Kokkos::LayoutRight>>);
static_assert(
    !std::is_assignable_v<Kokkos::View<double[3][5], Kokkos::LayoutRight> &,
                          Kokkos::View<double[3][5], Kokkos::LayoutLeft>>);
static_assert(
    !std::is_assignable_v<Kokkos::View<double[3][5], Kokkos::LayoutLeft> &,
                          Kokkos::View<double[3][5], Kokkos::LayoutRight>>);

}  // namespace
