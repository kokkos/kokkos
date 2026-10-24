// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "TestDeepCopy.hpp"

#include <cfenv>
#include <cstring>
#include <limits>

namespace {
struct DeepCopyFloatingPointEnvironment {
  std::fenv_t saved;
  DeepCopyFloatingPointEnvironment() { std::feholdexcept(&saved); }
  ~DeepCopyFloatingPointEnvironment() { std::fesetenv(&saved); }
};

template <class T>
void test_deep_copy_scalar_representation() {
  DeepCopyFloatingPointEnvironment environment;
  Kokkos::View<T*, TEST_EXECSPACE> view("scalar representation", 5);
  auto host = Kokkos::create_mirror_view(view);
  for (const T value :
       {T(0), -T(0), T(1), std::numeric_limits<T>::max(),
        std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::signaling_NaN()}) {
    for (const bool explicit_instance : {false, true}) {
      std::feclearexcept(FE_ALL_EXCEPT);
      if (explicit_instance) {
        TEST_EXECSPACE exec;
        Kokkos::deep_copy(exec, view, value);
        exec.fence();
      } else {
        Kokkos::deep_copy(view, value);
      }
      EXPECT_EQ(std::fetestexcept(FE_INVALID), 0);
      Kokkos::deep_copy(host, view);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(std::memcmp(&host(i), &value, sizeof(T)), 0);
      }
    }
  }
}
}  // namespace

TEST(TEST_CATEGORY, deep_copy_scalar_representation) {
  test_deep_copy_scalar_representation<float>();
  test_deep_copy_scalar_representation<double>();
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_0) {
  test_deep_copy_assignable_types<double, Foo>();
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_1) {
  test_deep_copy_assignable_types<double*, Foo*>(1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_2) {
  test_deep_copy_assignable_types<double**, Foo**>(1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_3) {
  test_deep_copy_assignable_types<double***, Foo***>(1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_4) {
  test_deep_copy_assignable_types<double****, Foo****>(1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_5) {
  test_deep_copy_assignable_types<double*****, Foo*****>(1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_6) {
  test_deep_copy_assignable_types<double******, Foo******>(1, 1, 1, 1, 1, 1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_7) {
  test_deep_copy_assignable_types<double*******, Foo*******>(1, 1, 1, 1, 1, 1,
                                                             1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_8) {
  test_deep_copy_assignable_types<double********, Foo********>(1, 1, 1, 1, 1, 1,
                                                               1, 1);
}

// Half types used to work just for rank 1-3
TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_1_half) {
  test_deep_copy_assignable_types<double*, Kokkos::Experimental::half_t*>(1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_2_half) {
  test_deep_copy_assignable_types<double**, Kokkos::Experimental::half_t**>(1,
                                                                            1);
}

TEST(TEST_CATEGORY, deep_copy_assignable_types_rank_3_half) {
  test_deep_copy_assignable_types<double***, Kokkos::Experimental::half_t***>(
      1, 1, 1);
}
