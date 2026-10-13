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
  Kokkos::View<double, TEST_EXECSPACE> v;
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

void test_single_policy() {
  using OtherExecSpace = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;

  static_assert(std::is_same_v<Kokkos::SinglePolicy<>::execution_policy,
                               Kokkos::SinglePolicy<>>);
  static_assert(
      std::is_same_v<
          Kokkos::SinglePolicy<>::execution_space,
          typename Kokkos::SinglePolicy<>::range_policy::execution_space>);
  static_assert(
      std::is_same_v<Kokkos::SinglePolicy<OtherExecSpace>::execution_space,
                     typename Kokkos::SinglePolicy<
                         OtherExecSpace>::range_policy::execution_space>);

  auto check_inner_policy = [](const auto& policy) {
    EXPECT_EQ(policy.impl_get_range_policy().begin(), 0);
    EXPECT_EQ(policy.impl_get_range_policy().end(), 1);
  };

  TEST_EXECSPACE exec_space;
  OtherExecSpace other_exec_space;
  Kokkos::SinglePolicy default_single_policy;
  Kokkos::SinglePolicy exec_space_single_policy(exec_space);
  Kokkos::SinglePolicy other_exec_space_single_policy(other_exec_space);

  check_inner_policy(default_single_policy);
  check_inner_policy(exec_space_single_policy);
  check_inner_policy(other_exec_space_single_policy);
}

// Test for the ParallelFor based API
void test_no_output() {
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
void test_one_output() {
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

// Test that single can be used to return types non conforming with the
// requirements for parallel_reduce
struct Restrictive {
  int val;

 private:
  Restrictive& operator+=(const Restrictive& rhs) {
    val += rhs.val;
    return *this;
  }
};

namespace Kokkos {
template <>
struct reduction_identity<Restrictive> {
 private:
  KOKKOS_FORCEINLINE_FUNCTION static Restrictive sum() { return Restrictive{}; }
};
}  // namespace Kokkos

void test_restrictive_return_type() {
  Restrictive ret;

  Kokkos::single(
      Kokkos::SinglePolicy<TEST_EXECSPACE>(),
      KOKKOS_LAMBDA(Restrictive & r) { r.val = 5; }, ret);

  EXPECT_EQ(ret.val, 5);
}

namespace Test {

TEST(TEST_CATEGORY, single_policy) { test_single_policy(); }

TEST(TEST_CATEGORY, single) { test_no_output(); }

TEST(TEST_CATEGORY, single_with_output) { test_one_output(); }

TEST(TEST_CATEGORY, single_restrictive_return_type) {
  test_restrictive_return_type();
}

}  // namespace Test
