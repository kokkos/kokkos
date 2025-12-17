// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Kokkos_BitManipulation.hpp>

#include "KokkosTest_Utils.hpp"

#define TEST_UNARY_EXPECT(TESTER, TYPE)                                       \
  template <typename Space>                                                   \
  struct Test_##TESTER {                                                      \
    bool expect_pass_;                                                        \
    TYPE val_;                                                                \
                                                                              \
    Test_##TESTER(bool expectation, const TYPE& val)                          \
        : expect_pass_(expectation), val_(val) {}                             \
                                                                              \
    KOKKOS_FUNCTION void operator()(int, int& errors) const { TESTER(val_); } \
                                                                              \
    void run() {                                                              \
      int errors;                                                             \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, 1), *this,        \
                              errors);                                        \
      int expected_error = (expect_pass_ ? 0 : 1);                            \
      ASSERT_EQ(errors, expected_error)                                       \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)errors       \
          << ", expected " << (bool)expected_error                            \
          << "\n\tfor value: " << val_;                                       \
    }                                                                         \
  }

#define TEST_BINARY_EXPECT(TESTER, TYPE1, TYPE2)                          \
  template <typename Space>                                               \
  struct Test_##TESTER {                                                  \
    bool expect_pass_;                                                    \
    TYPE1 val1_;                                                          \
    TYPE2 val2_;                                                          \
                                                                          \
    Test_##TESTER(bool expectation, const TYPE1& val1, const TYPE2& val2) \
        : expect_pass_(expectation), val1_(val1), val2_(val2) {}          \
                                                                          \
    KOKKOS_FUNCTION void operator()(int, int& errors) const {             \
      TESTER(val1_, val2_);                                               \
    }                                                                     \
                                                                          \
    void run() {                                                          \
      int errors;                                                         \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, 1), *this,    \
                              errors);                                    \
      int expected_error = (expect_pass_ ? 0 : 1);                        \
      ASSERT_EQ(errors, expected_error)                                   \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)errors   \
          << ", expected " << (bool)expected_error                        \
          << "\n\tfor values: " << val1_ << " and " << val2_;             \
    }                                                                     \
  }

#define TEST_TERNARY_EXPECT(TESTER, TYPE1, TYPE2, TYPE3)                       \
  template <typename Space>                                                    \
  struct Test_##TESTER {                                                       \
    bool expect_pass_;                                                         \
    TYPE1 val1_;                                                               \
    TYPE2 val2_;                                                               \
    TYPE3 val3_;                                                               \
                                                                               \
    Test_##TESTER(bool expectation, const TYPE1& val1, const TYPE2& val2,      \
                  const TYPE3& val3)                                           \
        : expect_pass_(expectation), val1_(val1), val2_(val2), val3_(val3) {}  \
                                                                               \
    KOKKOS_FUNCTION void operator()(int, int& errors) const {                  \
      TESTER(val1_, val2_, val3_);                                             \
    }                                                                          \
                                                                               \
    void run() {                                                               \
      int errors;                                                              \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, 1), *this,         \
                              errors);                                         \
      int expected_error = (expect_pass_ ? 0 : 1);                             \
      ASSERT_EQ(errors, expected_error)                                        \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)errors        \
          << ", expected " << (bool)expected_error                             \
          << "\n\tfor values: " << val1_ << ", " << val2_ << " and " << val3_; \
    }                                                                          \
  }

TEST_UNARY_EXPECT(KOKKOS_EXPECT_TRUE, double);
TEST_UNARY_EXPECT(KOKKOS_EXPECT_FALSE, double);
#if !__FINITE_MATH_ONLY__
TEST_UNARY_EXPECT(KOKKOS_EXPECT_NAN, double);
TEST_UNARY_EXPECT(KOKKOS_EXPECT_INF, double);
#endif

TEST_BINARY_EXPECT(KOKKOS_EXPECT_EQ, double, double);
TEST_BINARY_EXPECT(KOKKOS_EXPECT_NE, double, double);

TEST_TERNARY_EXPECT(KOKKOS_EXPECT_NEAR_ULPS, double, double, int);

#define DO_UNARY_TEST(TESTER, expect_pass, val)           \
  do {                                                    \
    Test_##TESTER<TEST_EXECSPACE> test(expect_pass, val); \
    test.run();                                           \
  } while (0)

#define DO_BINARY_TEST(TESTER, expect_pass, val1, val2)          \
  do {                                                           \
    Test_##TESTER<TEST_EXECSPACE> test(expect_pass, val1, val2); \
    test.run();                                                  \
  } while (0)

#define DO_TERNARY_TEST(TESTER, expect_pass, val1, val2, val3)         \
  do {                                                                 \
    Test_##TESTER<TEST_EXECSPACE> test(expect_pass, val1, val2, val3); \
    test.run();                                                        \
  } while (0)

TEST(Test_Utils, test_expects) {
  DO_BINARY_TEST(KOKKOS_EXPECT_EQ, false, 5., 4.);
  DO_BINARY_TEST(KOKKOS_EXPECT_EQ, true, 5., 5.);

  DO_BINARY_TEST(KOKKOS_EXPECT_NE, false, 5., 5.);
  DO_BINARY_TEST(KOKKOS_EXPECT_NE, true, 5., 4.);

  DO_UNARY_TEST(KOKKOS_EXPECT_TRUE, false, false);
  DO_UNARY_TEST(KOKKOS_EXPECT_TRUE, true, true);

  DO_UNARY_TEST(KOKKOS_EXPECT_FALSE, true, false);
  DO_UNARY_TEST(KOKKOS_EXPECT_FALSE, false, true);

#if !__FINITE_MATH_ONLY__
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, 5.);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, 0.);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, Kokkos::finite_max_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, Kokkos::signaling_NaN_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, Kokkos::quiet_NaN_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, false, Kokkos::denorm_min_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, true, Kokkos::infinity_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_INF, true, -Kokkos::infinity_v<double>);

  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, 5.);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, 0.);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, Kokkos::finite_max_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, Kokkos::infinity_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, -Kokkos::infinity_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, false, Kokkos::denorm_min_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, true, Kokkos::signaling_NaN_v<double>);
  DO_UNARY_TEST(KOKKOS_EXPECT_NAN, true, Kokkos::quiet_NaN_v<double>);
#endif

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 2., 2., 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 2., 1.9, 1);

  {
    uint64_t u = 0x4000000000000002;
    double d   = Kokkos::bit_cast<double>(u);
    DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 2., d, 1);
    DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 2., d, 2);
  }

  {
    uint64_t u = 0x3FFFFFFFFFFFFFFE;
    double d   = Kokkos::bit_cast<double>(u);
    DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 2., d, 1);
    DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 2., d, 2);
  }
}
