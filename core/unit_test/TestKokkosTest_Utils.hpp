// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <Kokkos_BitManipulation.hpp>

#include "KokkosTest_Utils.hpp"

// KOKKOS_FUNCTION void toto() {
//   KOKKOS_EXPECT_NE(m_test_reporter, 1,1);
// }

#define TEST_UNARY_EXPECT(TESTER)                                        \
  template <typename Space, typename type>                               \
  struct Test_##TESTER {                                                 \
    bool expect_pass_;                                                   \
    type val_;                                                           \
                                                                         \
    KokkosTest::TestReporter<Space, 2048> m_test_reporter;               \
                                                                         \
    Test_##TESTER(bool expectation, const type& val)                     \
        : expect_pass_(expectation), val_(val) {}                        \
                                                                         \
    KOKKOS_FUNCTION void operator()(int) const {                         \
      TESTER(m_test_reporter, val_);                                     \
    }                                                                    \
                                                                         \
    void run(const char* file, int line) {                               \
      Kokkos::parallel_for(Kokkos::RangePolicy<Space>(0, 1), *this);     \
      std::stringstream os;                                              \
      int expected_error = (expect_pass_ ? 0 : 1);                       \
      int nerrors        = m_test_reporter.print_errors(os);             \
      ASSERT_EQ(nerrors, expected_error)                                 \
          << file << ":" << line << ":\n"                                \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)nerrors \
          << ", expected " << (bool)expected_error                       \
          << "\n\tfor value: " << val_;                                  \
    }                                                                    \
  }

#define TEST_BINARY_EXPECT(TESTER)                                        \
  template <typename Space, typename type1, typename type2>               \
  struct Test_##TESTER {                                                  \
    bool expect_pass_;                                                    \
    type1 val1_;                                                          \
    type2 val2_;                                                          \
                                                                          \
    KokkosTest::TestReporter<Space, 2048> m_test_reporter;                \
                                                                          \
    Test_##TESTER(bool expectation, const type1& val1, const type2& val2) \
        : expect_pass_(expectation), val1_(val1), val2_(val2) {}          \
                                                                          \
    KOKKOS_FUNCTION void operator()(int) const {                          \
      TESTER(m_test_reporter, val1_, val2_);                              \
    }                                                                     \
                                                                          \
    void run(const char* file, int line) {                                \
      Kokkos::parallel_for(Kokkos::RangePolicy<Space>(0, 1), *this);      \
      std::stringstream os;                                               \
      int expected_error = (expect_pass_ ? 0 : 1);                        \
      int nerrors        = m_test_reporter.print_errors(os);              \
      ASSERT_EQ(nerrors, expected_error)                                  \
          << file << ":" << line << ":\n"                                 \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)nerrors  \
          << ", expected " << (bool)expected_error                        \
          << "\n\tfor values: " << val1_ << " and " << val2_;             \
    }                                                                     \
  }

#define TEST_TERNARY_EXPECT(TESTER)                                            \
  template <typename Space, typename type1, typename type2, typename type3>    \
  struct Test_##TESTER {                                                       \
    bool expect_pass_;                                                         \
    type1 val1_;                                                               \
    type2 val2_;                                                               \
    type3 val3_;                                                               \
                                                                               \
    KokkosTest::TestReporter<Space, 2048> m_test_reporter;                     \
                                                                               \
    Test_##TESTER(bool expectation, const type1& val1, const type2& val2,      \
                  const type3& val3)                                           \
        : expect_pass_(expectation), val1_(val1), val2_(val2), val3_(val3) {}  \
                                                                               \
    KOKKOS_FUNCTION void operator()(int) const {                               \
      TESTER(m_test_reporter, val1_, val2_, val3_);                            \
    }                                                                          \
                                                                               \
    void run(const char* file, int line) {                                     \
      Kokkos::parallel_for(Kokkos::RangePolicy<Space>(0, 1), *this);           \
      std::stringstream os;                                                    \
      int expected_error = (expect_pass_ ? 0 : 1);                             \
      int nerrors        = m_test_reporter.print_errors(os);                   \
      ASSERT_EQ(nerrors, expected_error)                                       \
          << file << ":" << line << ":\n"                                      \
          << KOKKOS_IMPL_STRINGIFY(TESTER) " returned " << (bool)nerrors       \
          << ", expected " << (bool)expected_error                             \
          << "\n\tfor values: " << val1_ << ", " << val2_ << " and " << val3_; \
    }                                                                          \
  }

TEST_UNARY_EXPECT(KOKKOS_EXPECT_TRUE);
TEST_UNARY_EXPECT(KOKKOS_EXPECT_FALSE);
#if !__FINITE_MATH_ONLY__
TEST_UNARY_EXPECT(KOKKOS_EXPECT_NAN);
TEST_UNARY_EXPECT(KOKKOS_EXPECT_INF);
#endif

TEST_BINARY_EXPECT(KOKKOS_EXPECT_EQ);
TEST_BINARY_EXPECT(KOKKOS_EXPECT_NE);

TEST_TERNARY_EXPECT(KOKKOS_EXPECT_NEAR_ULPS);

#define DO_UNARY_TEST(TESTER, expect_pass, val)                          \
  do {                                                                   \
    Test_##TESTER<TEST_EXECSPACE, decltype(val)> test(expect_pass, val); \
    test.run(__FILE__, __LINE__);                                        \
  } while (false)

#define DO_BINARY_TEST(TESTER, expect_pass, val1, val2)                 \
  do {                                                                  \
    Test_##TESTER<TEST_EXECSPACE, decltype(val1), decltype(val2)> test( \
        expect_pass, val1, val2);                                       \
    test.run(__FILE__, __LINE__);                                       \
  } while (false)

#define DO_TERNARY_TEST(TESTER, expect_pass, val1, val2, val3)    \
  do {                                                            \
    Test_##TESTER<TEST_EXECSPACE, decltype(val1), decltype(val2), \
                  decltype(val3)>                                 \
        test(expect_pass, val1, val2, val3);                      \
    test.run(__FILE__, __LINE__);                                 \
  } while (false)

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

  // Expect test success
  // Near 0
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.f, -0.f, 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.f,
                  Kokkos::bit_cast<float>(0x1u), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.f,
                  Kokkos::bit_cast<float>(0x2u), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.f,
                  Kokkos::bit_cast<float>(0x4u), 4);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0., -0., 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.,
                  Kokkos::bit_cast<double>(0x1llu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.,
                  Kokkos::bit_cast<double>(0x2llu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.,
                  Kokkos::bit_cast<double>(0x4llu), 4);

  // Near 1
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f,
                  Kokkos::bit_cast<float>(0x3F7FFFFEu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f,
                  Kokkos::bit_cast<float>(0x3F7FFFFFu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f,
                  Kokkos::bit_cast<float>(0x3F800001u), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f,
                  Kokkos::bit_cast<float>(0x3F800002u), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f,
                  Kokkos::bit_cast<float>(0x3F800004u), 4);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.,
                  Kokkos::bit_cast<double>(0x3FEFFFFFFFFFFFFEllu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.,
                  Kokkos::bit_cast<double>(0x3FEFFFFFFFFFFFFFllu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000001llu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000002llu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000004llu), 4);

  // Near Inf
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFEu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFDu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFBu), 4);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFEllu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFDllu), 2);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFBllu), 4);

  // Equality
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0.f, 0.f, 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1.f, 1.f, 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFFu), 0);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 0., 0., 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true, 1., 1., 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, true,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu), 0);

  // Expect test failure
  // Near 0
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.f,
  //                 Kokkos::bit_cast<float>(0x80000005u), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.f,
                  Kokkos::bit_cast<float>(0x1u), 0);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.f,
  //                 Kokkos::bit_cast<float>(0x2u), 1);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.f,
  //                  Kokkos::bit_cast<float>(0x4u), 3);

  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.,
  //                 Kokkos::bit_cast<double>(0x8000000000000001llu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.,
                  Kokkos::bit_cast<double>(0x1llu), 0);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.,
  //                 Kokkos::bit_cast<double>(0x2llu), 1);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 0.,
  //                 Kokkos::bit_cast<double>(0x4llu), 3);

  // Near 1
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.f,
                  Kokkos::bit_cast<float>(0x3F7FFFFE), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.f,
                  Kokkos::bit_cast<float>(0x3F7FFFFF), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.f,
                  Kokkos::bit_cast<float>(0x3F800001), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.f,
                  Kokkos::bit_cast<float>(0x3F800002), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.f,
                  Kokkos::bit_cast<float>(0x3F800004), 3);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.,
                  Kokkos::bit_cast<double>(0x3FEFFFFFFFFFFFFEllu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.,
                  Kokkos::bit_cast<double>(0x3FEFFFFFFFFFFFFFllu), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000001llu), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000002llu), 1);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false, 1.,
                  Kokkos::bit_cast<double>(0x3FF0000000000004llu), 3);

  // Near Inf
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFEu), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
                  Kokkos::bit_cast<float>(0x7F7FFFFFu),
                  Kokkos::bit_cast<float>(0x7F7FFFFDu), 1);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
  //                 Kokkos::bit_cast<float>(0x7F7FFFFFu),
  //                 Kokkos::bit_cast<float>(0x7F7FFFFBu), 3);

  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFEllu), 0);
  DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
                  Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFDllu), 1);
  // DO_TERNARY_TEST(KOKKOS_EXPECT_NEAR_ULPS, false,
  //                 Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFFllu),
  //                 Kokkos::bit_cast<double>(0xFFEFFFFFFFFFFFFBllu), 3);

  // toto();
}
