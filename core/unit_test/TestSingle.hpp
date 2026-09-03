// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <KokkosTest_Utils.hpp>

struct TimesTwoTag {};

struct Functor {
  Kokkos::View<double> v;
  KOKKOS_FUNCTION void operator()() const { v() *= 3; }
  KOKKOS_FUNCTION void operator()(const TimesTwoTag) const { v() *= 2; }
};

struct TenTag {};

struct ReductionFunctor {
  KOKKOS_FUNCTION void operator()(int& res) const { res = 5; }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res) const { res = 10; }

  KOKKOS_FUNCTION void operator()(int& res1, int& res2) const {
    res1 = 5;
    res2 = 5;
  }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res1, int& res2) const {
    res1 = 10;
    res2 = 10;
  }

  KOKKOS_FUNCTION void operator()(int& res1, int& res2, int& res3) const {
    res1 = 5;
    res2 = 5;
    res3 = 5;
  }
  KOKKOS_FUNCTION void operator()(const TenTag, int& res1, int& res2,
                                  int& res3) const {
    res1 = 10;
    res2 = 10;
    res3 = 10;
  }
};

// Test for the ParallelFor based API
void test() {
  TEST_EXECSPACE exec_space;
  double expected = 5;

  Kokkos::View<double, TEST_EXECSPACE> v("v");
  Kokkos::deep_copy(v, expected);

  Functor f{v};

  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>) {
    // These tests use the default execution space, thus they should not be run
    // when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace
    // Minimal
    Kokkos::single(f);
    expected *= 3;
    EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

    // Minimal lambda
    Kokkos::single(KOKKOS_LAMBDA() { v() += 2; });
    expected += 2;
    EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

    // +kernel_name
    Kokkos::single("single", f);
    expected *= 3;
    EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

    // +WorkTag
    Kokkos::single(Kokkos::SinglePolicy<TimesTwoTag>(), f);
    expected *= 2;
    EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

    // +WorkTag +kernel_name
    Kokkos::single("single+worktag", Kokkos::SinglePolicy<TimesTwoTag>(), f);
    expected *= 2;
    EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));
  }

  // +kernal_name +WorkTag +ExecSpace
  Kokkos::single("single+worktag+exec_space",
                 Kokkos::SinglePolicy<TimesTwoTag, TEST_EXECSPACE>(), f);
  expected *= 2;
  EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

  // +WorkTag +ExecSpace
  Kokkos::single(Kokkos::SinglePolicy<TimesTwoTag, TEST_EXECSPACE>(), f);
  expected *= 2;
  EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

  // +ExecSpace
  Kokkos::single(Kokkos::SinglePolicy<TEST_EXECSPACE>(), f);
  expected *= 3;
  EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

  // +kernel_name +ExecSpace
  Kokkos::single("single+policy", Kokkos::SinglePolicy<TEST_EXECSPACE>(), f);
  expected *= 3;
  EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));

  // +ExecSpace instance
  Kokkos::single(Kokkos::SinglePolicy(exec_space), f);
  exec_space.fence();
  expected *= 3;
  EXPECT_TRUE(KokkosTest::contains(exec_space, v, expected));
}

// Test for the ParallelReduce based API with a single return value
void test_one_ouput() {
  int val;
  ReductionFunctor f;

  // Full signature
  // Functor
  val = 0;
  Kokkos::single("single with output+worktag+exec_space",
                 Kokkos::SinglePolicy<TEST_EXECSPACE, TenTag>(), f, val);
  EXPECT_EQ(val, 10);

  // Lambda
  val = 0;
  Kokkos::single(
      "single with output+lambda+exec_space",
      Kokkos::SinglePolicy<TEST_EXECSPACE>(),
      KOKKOS_LAMBDA(int& ret) { ret = 5; }, val);
  EXPECT_EQ(val, 5);

  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>) {
    // These tests use the default execution space, thus they should not be run
    // when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace
    // Minimal
    val = 0;
    Kokkos::single(f, val);
    EXPECT_EQ(val, 5);

    // +kernel_name
    val = 0;
    Kokkos::single("single with output", f, val);
    EXPECT_EQ(val, 5);

    // +Worktag
    val = 0;
    Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val);
    EXPECT_EQ(val, 10);

    // +kernel_name +Worktag
    val = 0;
    Kokkos::single("single with output+worktag", Kokkos::SinglePolicy<TenTag>(),
                   f, val);
    EXPECT_EQ(val, 10);
  }

  // +Policy
  val = 0;
  Kokkos::single(Kokkos::SinglePolicy<TEST_EXECSPACE>(), f, val);
  EXPECT_EQ(val, 5);

  // +kernel_name +Policy
  val = 0;
  Kokkos::single("single with output+exec_space",
                 Kokkos::SinglePolicy<TEST_EXECSPACE>(), f, val);
  EXPECT_EQ(val, 5);

  // +Worktag +Policy
  val = 0;
  Kokkos::single(Kokkos::SinglePolicy<TEST_EXECSPACE, TenTag>(), f, val);
  EXPECT_EQ(val, 10);

  // +ExecSpace instance
  val = 0;
  TEST_EXECSPACE exec_space;
  Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val);
  exec_space.fence();
  EXPECT_EQ(val, 5);
}

// Test for the ParallelReduce based API with several return values
// (CombinedReducer based API)
void test_multiple_outputs() {
  // Two args
  {
    int val1, val2;

    auto l = KOKKOS_LAMBDA(int& s1, int& s2) {
      s1 = 1;
      s2 = 2;
    };

    // Lambda
    if constexpr (std::is_same_v<TEST_EXECSPACE,
                                 Kokkos::DefaultExecutionSpace>) {
      // These tests use the default execution space, thus they should not be
      // run when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace Minimal
      val1 = val2 = 0;
      Kokkos::single(l, val1, val2);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);

      // +Label
      val1 = val2 = 0;
      Kokkos::single("single with multiple outputs+lambda", l, val1, val2);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);

      // +Policy
      val1 = val2 = 0;
      Kokkos::single(Kokkos::SinglePolicy(), l, val1, val2);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);

      // Full
      val1 = val2 = 0;
      Kokkos::single("single with 2 outputs+lambda+policy",
                     Kokkos::SinglePolicy(), l, val1, val2);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);
    }

    // Full with ExecSpace
    val1 = val2 = 0;
    Kokkos::single("single with 2 outputs+lambda+exec_space",
                   Kokkos::SinglePolicy<TEST_EXECSPACE>(), l, val1, val2);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);

    // Functor
    ReductionFunctor f{};
    if constexpr (std::is_same_v<TEST_EXECSPACE,
                                 Kokkos::DefaultExecutionSpace>) {
      // These tests use the default execution space, thus they should not be
      // run when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace Minimal
      val1 = val2 = 0;
      Kokkos::single(f, val1, val2);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);

      // +Label
      val1 = val2 = 0;
      Kokkos::single("single with 2 outputs", f, val1, val2);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);

      // +Policy
      val1 = val2 = 0;
      Kokkos::single(Kokkos::SinglePolicy(), f, val1, val2);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);

      // +Policy with WorkTag
      val1 = val2 = 0;
      Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val1, val2);
      EXPECT_EQ(val1, 10);
      EXPECT_EQ(val2, 10);

      // Full
      val1 = val2 = 0;
      Kokkos::single("single with 2 outputs+policy", Kokkos::SinglePolicy(), f,
                     val1, val2);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);

      // Full with WorkTag
      val1 = val2 = 0;
      Kokkos::single("single with 2 outputs+worktag",
                     Kokkos::SinglePolicy<TenTag>(), f, val1, val2);
      EXPECT_EQ(val1, 10);
      EXPECT_EQ(val2, 10);
    }

    // Full with WorkTag and ExecSpace
    val1 = val2 = 0;
    Kokkos::single("single with 2 outputs+worktag+execspace",
                   Kokkos::SinglePolicy<TenTag, TEST_EXECSPACE>(), f, val1,
                   val2);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);

    // With ExecSpace instance
    val1 = val2 = 0;
    TEST_EXECSPACE exec_space;
    Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val1, val2);
    exec_space.fence();
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
  }

  // Three args
  {
    int val1, val2, val3;

    auto l = KOKKOS_LAMBDA(int& s1, int& s2, int& s3) {
      s1 = 1;
      s2 = 2;
      s3 = 3;
    };

    // Lambda
    if constexpr (std::is_same_v<TEST_EXECSPACE,
                                 Kokkos::DefaultExecutionSpace>) {
      // These tests use the default execution space, thus they should not be
      // run when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace Minimal
      val1 = val2 = val3 = 0;
      Kokkos::single(l, val1, val2, val3);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);
      EXPECT_EQ(val3, 3);

      // +Label
      val1 = val2 = val3 = 0;
      Kokkos::single("single with 3 outputs+lambda", l, val1, val2, val3);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);
      EXPECT_EQ(val3, 3);

      // +Policy
      val1 = val2 = val3 = 0;
      Kokkos::single(Kokkos::SinglePolicy(), l, val1, val2, val3);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);
      EXPECT_EQ(val3, 3);

      // Full
      val1 = val2 = val3 = 0;
      Kokkos::single("single with 3 outputs+lambda+policy",
                     Kokkos::SinglePolicy(), l, val1, val2, val3);
      EXPECT_EQ(val1, 1);
      EXPECT_EQ(val2, 2);
      EXPECT_EQ(val3, 3);
    }

    // Full with ExecSpace
    val1 = val2 = val3 = 0;
    Kokkos::single("single with 3 outputs+lambda+exec_space",
                   Kokkos::SinglePolicy<TEST_EXECSPACE>(), l, val1, val2, val3);
    EXPECT_EQ(val1, 1);
    EXPECT_EQ(val2, 2);
    EXPECT_EQ(val3, 3);

    // Functor
    ReductionFunctor f{};
    if constexpr (std::is_same_v<TEST_EXECSPACE,
                                 Kokkos::DefaultExecutionSpace>) {
      // These tests use the default execution space, thus they should not be
      // run when TEST_EXECSPACE != Kokkos::DefaultExecutionSpace Minimal
      val1 = val2 = val3 = 0;
      Kokkos::single(f, val1, val2, val3);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);
      EXPECT_EQ(val3, 5);

      // +Label
      val1 = val2 = val3 = 0;
      Kokkos::single("single with 3 outputs", f, val1, val2, val3);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);
      EXPECT_EQ(val3, 5);

      // +Policy
      val1 = val2 = val3 = 0;
      Kokkos::single(Kokkos::SinglePolicy(), f, val1, val2, val3);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);
      EXPECT_EQ(val3, 5);

      // +Policy with WorkTag
      val1 = val2 = val3 = 0;
      Kokkos::single(Kokkos::SinglePolicy<TenTag>(), f, val1, val2, val3);
      EXPECT_EQ(val1, 10);
      EXPECT_EQ(val2, 10);
      EXPECT_EQ(val3, 10);

      // Full
      val1 = val2 = val3 = 0;
      Kokkos::single("single with 3 outputs+policy", Kokkos::SinglePolicy(), f,
                     val1, val2, val3);
      EXPECT_EQ(val1, 5);
      EXPECT_EQ(val2, 5);
      EXPECT_EQ(val3, 5);

      // Full with WorkTag
      val1 = val2 = val3 = 0;
      Kokkos::single("single with 3 outputs+worktag",
                     Kokkos::SinglePolicy<TenTag>(), f, val1, val2, val3);
      EXPECT_EQ(val1, 10);
      EXPECT_EQ(val2, 10);
      EXPECT_EQ(val3, 10);
    }

    // Full with WorkTag and ExecSpace
    val1 = val2 = val3 = 0;
    Kokkos::single("single with 3 outputs+worktag+exec_space",
                   Kokkos::SinglePolicy<TenTag, TEST_EXECSPACE>(), f, val1,
                   val2, val3);
    EXPECT_EQ(val1, 10);
    EXPECT_EQ(val2, 10);
    EXPECT_EQ(val3, 10);

    // With ExecSpace instance
    val1 = val2 = val3 = 0;
    TEST_EXECSPACE exec_space;
    Kokkos::single(Kokkos::SinglePolicy(exec_space), f, val1, val2, val3);
    exec_space.fence();
    EXPECT_EQ(val1, 5);
    EXPECT_EQ(val2, 5);
    EXPECT_EQ(val3, 5);
  }
}

namespace Test {
TEST(TEST_CATEGORY, single) { test(); }

TEST(TEST_CATEGORY, single_with_output) { test_one_ouput(); }

TEST(TEST_CATEGORY, single_with_multiple_outputs) { test_multiple_outputs(); }
}  // namespace Test
