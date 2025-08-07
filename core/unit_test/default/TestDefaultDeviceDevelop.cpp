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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

struct TimeTwo {};

struct Functor {
  Kokkos::View<double*> v;
  KOKKOS_FUNCTION void operator()() const { v(0) *= 3; }
  KOKKOS_FUNCTION void operator()(const TimeTwo) const { v(0) *= 2; }
};

struct PlusTen {};

struct FunctorRed {
  Kokkos::View<double*> v;
  KOKKOS_FUNCTION void operator()(int& res) const { res = v(0) - 5; }
  KOKKOS_FUNCTION void operator()(const PlusTen, int& res) const {
    res = v(0) + 10;
  }
};

void test_func() {
  {
    Kokkos::View<double*> v("v", 1);
    auto mirror = Kokkos::create_mirror_view(v);
    mirror(0)   = 5;
    Kokkos::deep_copy(v, mirror);

    Functor f;
    f.v = v;

    double res = 5;

    // Minimal
    Kokkos::single(f);
    Kokkos::deep_copy(mirror, v);
    res *= 3;
    EXPECT_EQ(res, mirror(0));

    // Minimal lambda
    Kokkos::single(KOKKOS_LAMBDA() { v(0) += 2; });
    Kokkos::deep_copy(mirror, v);
    res += 2;
    EXPECT_EQ(res, mirror(0));

    // +kernal_name +WorkTag +ExecSpace
    Kokkos::single(
        "Single",
        Kokkos::SinglePolicy<TimeTwo, Kokkos::DefaultExecutionSpace>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 2;
    EXPECT_EQ(res, mirror(0));

    // +kernel_name
    Kokkos::single("test", f);
    Kokkos::deep_copy(mirror, v);
    res *= 3;
    EXPECT_EQ(res, mirror(0));

    // +WorkTag
    Kokkos::single(Kokkos::SinglePolicy<TimeTwo>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 2;
    EXPECT_EQ(res, mirror(0));

    // +WorkTag +kernel_name
    Kokkos::single("Single", Kokkos::SinglePolicy<TimeTwo>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 2;
    EXPECT_EQ(res, mirror(0));

    // +WorkTag +ExecSpace
    Kokkos::single(
        Kokkos::SinglePolicy<TimeTwo, Kokkos::DefaultExecutionSpace>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 2;
    EXPECT_EQ(res, mirror(0));

    // +ExecSpace
    Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 3;
    EXPECT_EQ(res, mirror(0));

    // +Policy +kernel_name
    Kokkos::single("Single",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f);
    Kokkos::deep_copy(mirror, v);
    res *= 3;
    EXPECT_EQ(res, mirror(0));
  }

  // Reduce based API
  {
    Kokkos::View<double*> v("v", 1);
    auto mirror = Kokkos::create_mirror_view(v);
    mirror(0)   = 5;
    Kokkos::deep_copy(v, mirror);

    FunctorRed f;
    f.v = v;

    int val;

    //// Full signature
    //Kokkos::single(
    //    "Single Reduce",
    //    Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, PlusTen>(), f, val);
    //EXPECT_EQ(val, 15);

    //Kokkos::single(
    //    "Single Reduce", Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
    //    KOKKOS_LAMBDA(int& ret) { ret = 5; }, val);
    //EXPECT_EQ(val, 5);

    // Minimal
    Kokkos::single(f, val);
    EXPECT_EQ(val, 0);

    //// +kernel_name
    //Kokkos::single("Single", f, val);
    //EXPECT_EQ(val, 0);

    //// +Policy
    //Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f,
    //               val);
    //EXPECT_EQ(val, 0);

    //// +Worktag
    //Kokkos::single(Kokkos::SinglePolicy<PlusTen>(), f, val);
    //EXPECT_EQ(val, 15);
  }
}

namespace Test {
TEST(defaultdevicetype, development_test) { test_func(); }
}  // namespace Test
