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

struct Ten {};

struct FunctorRed {
  KOKKOS_FUNCTION void operator()(int& res) const { res = 5; }
  KOKKOS_FUNCTION void operator()(const Ten, int& res) const { res = 10; }
};

struct CombinedFunctorRed {
  KOKKOS_FUNCTION void operator()(int& res1, int& res2) const {
    res1 = 5;
    res2 = 5;
  }
  KOKKOS_FUNCTION void operator()(const Ten, int& res1, int& res2) const {
    res1 = 10;
    res2 = 10;
  }

  KOKKOS_FUNCTION void operator()(int& res1, int& res2, int& res3) const {
    res1 = 5;
    res2 = 5;
    res3 = 5;
  }
  KOKKOS_FUNCTION void operator()(const Ten, int& res1, int& res2,
                                  int& res3) const {
    res1 = 10;
    res2 = 10;
    res3 = 10;
  }
};

void test_func() {
  // ParallelFor based API
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

  // ParallelReduced based API
  {
    int val;
    FunctorRed f;

    // Full signature
    // Functor
    Kokkos::single("Single Reduce",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, Ten>(),
                   f, val);
    EXPECT_EQ(val, 10);

    // Lambda
    Kokkos::single(
        "Single Reduce", Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
        KOKKOS_LAMBDA(int& ret) { ret = 5; }, val);
    EXPECT_EQ(val, 5);

    // Minimal
    Kokkos::single(f, val);
    EXPECT_EQ(val, 5);

    // +kernel_name
    Kokkos::single("Single", f, val);
    EXPECT_EQ(val, 5);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f,
                   val);
    EXPECT_EQ(val, 5);

    // +kernel_name +Policy
    Kokkos::single("Single",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f,
                   val);
    EXPECT_EQ(val, 5);

    // +Worktag
    Kokkos::single(Kokkos::SinglePolicy<Ten>(), f, val);
    EXPECT_EQ(val, 10);

    // +Worktag +Policy
    Kokkos::single(Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, Ten>(),
                   f, val);
    EXPECT_EQ(val, 10);

    // +kernel_name +Worktag
    Kokkos::single("Single", Kokkos::SinglePolicy<Ten>(), f, val);
    EXPECT_EQ(val, 10);
  }

  // Combined Reducer
  {
    int sum1, sum2;

    auto l = KOKKOS_LAMBDA(int& sum1, int& sum2) {
      sum1 = 1;
      sum2 = 2;
    };

    // Lambda
    // Minimal
    Kokkos::single(l, sum1, sum2);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);

    // +Label
    Kokkos::single("Combined reducer", l, sum1, sum2);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), l, sum1, sum2);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);

    // Full
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy(), l, sum1, sum2);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);

    // Full with ExecSpace
    Kokkos::single("Combined reducer",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), l,
                   sum1, sum2);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);

    // Functor
    CombinedFunctorRed f{};
    // Minimal
    Kokkos::single(f, sum1, sum2);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);

    // +Label
    Kokkos::single("Combined reducer", f, sum1, sum2);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), f, sum1, sum2);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);

    // +Policy with WorkTag
    Kokkos::single(Kokkos::SinglePolicy<Ten>(), f, sum1, sum2);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);

    // Full
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy(), f, sum1, sum2);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);

    // Full with WorkTag
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy<Ten>(), f, sum1,
                   sum2);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);

    // Full with WorkTag and ExecSpace
    Kokkos::single("Combined reducer",
                   Kokkos::SinglePolicy<Ten, Kokkos::DefaultExecutionSpace>(),
                   f, sum1, sum2);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);
  }

  {
    int sum1, sum2, sum3;

    auto l = KOKKOS_LAMBDA(int& sum1, int& sum2, int& sum3) {
      sum1 = 1;
      sum2 = 2;
      sum3 = 3;
    };

    // Lambda
    // Minimal
    Kokkos::single(l, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);
    EXPECT_EQ(sum3, 3);

    // +Label
    Kokkos::single("Combined reducer", l, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);
    EXPECT_EQ(sum3, 3);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), l, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);
    EXPECT_EQ(sum3, 3);

    // Full
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy(), l, sum1, sum2,
                   sum3);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);
    EXPECT_EQ(sum3, 3);

    // Full with ExecSpace
    Kokkos::single("Combined reducer",
                   Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), l,
                   sum1, sum2, sum3);
    EXPECT_EQ(sum1, 1);
    EXPECT_EQ(sum2, 2);
    EXPECT_EQ(sum3, 3);

    //// Functor
    CombinedFunctorRed f{};
    // Minimal
    Kokkos::single(f, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);
    EXPECT_EQ(sum3, 5);

    // +Label
    Kokkos::single("Combined reducer", f, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);
    EXPECT_EQ(sum3, 5);

    // +Policy
    Kokkos::single(Kokkos::SinglePolicy(), f, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);
    EXPECT_EQ(sum3, 5);

    // +Policy with WorkTag
    Kokkos::single(Kokkos::SinglePolicy<Ten>(), f, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);
    EXPECT_EQ(sum3, 10);

    // Full
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy(), f, sum1, sum2,
                   sum3);
    EXPECT_EQ(sum1, 5);
    EXPECT_EQ(sum2, 5);
    EXPECT_EQ(sum3, 5);

    // Full with WorkTag
    Kokkos::single("Combined reducer", Kokkos::SinglePolicy<Ten>(), f, sum1,
                   sum2, sum3);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);
    EXPECT_EQ(sum3, 10);

    // Full with WorkTag and ExecSpace
    Kokkos::single("Combined reducer",
                   Kokkos::SinglePolicy<Ten, Kokkos::DefaultExecutionSpace>(),
                   f, sum1, sum2, sum3);
    EXPECT_EQ(sum1, 10);
    EXPECT_EQ(sum2, 10);
    EXPECT_EQ(sum3, 10);
  }
}

namespace Test {
TEST(defaultdevicetype, development_test) { test_func(); }
}  // namespace Test
