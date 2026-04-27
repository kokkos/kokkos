// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSTEST_UTILS_HPP
#define KOKKOSTEST_UTILS_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <type_traits>

#include <../../containers/src/Kokkos_ErrorReporter.hpp>

namespace KokkosTest {

struct FloatingPointComparison {
 private:
  template <class T>
  KOKKOS_FUNCTION static double eps(T) {
    return DBL_EPSILON;
  }

#if !KOKKOS_HALF_T_IS_FLOAT
  KOKKOS_FUNCTION static Kokkos::Experimental::half_t eps(
      Kokkos::Experimental::half_t) {
// FIXME_NVHPC compile-time error
#ifdef KOKKOS_COMPILER_NVHPC
    return 0.0009765625F;
#else
    return Kokkos::Experimental::epsilon<Kokkos::Experimental::half_t>::value;
#endif
  }
#endif

#if !KOKKOS_BHALF_T_IS_FLOAT
  KOKKOS_FUNCTION
  Kokkos::Experimental::bhalf_t static eps(Kokkos::Experimental::bhalf_t) {
// FIXME_NVHPC compile-time error
#ifdef KOKKOS_COMPILER_NVHPC
    return 0.0078125;
#else
    return Kokkos::Experimental::epsilon<Kokkos::Experimental::bhalf_t>::value;
#endif
  }
#endif

  KOKKOS_FUNCTION static double eps(float) { return FLT_EPSILON; }

// POWER9 gives unexpected values with LDBL_EPSILON issues
// https://stackoverflow.com/questions/68960416/ppc64-long-doubles-machine-epsilon-calculation
#if defined(KOKKOS_ARCH_POWER9) || defined(KOKKOS_ARCH_POWER8)
  KOKKOS_FUNCTION static double eps(long double) { return DBL_EPSILON; }
#else
  KOKKOS_FUNCTION static double eps(long double) { return LDBL_EPSILON; }
#endif

  // Using absolute here instead of abs, since we actually test abs ...
  template <class T>
  KOKKOS_FUNCTION static T absolute(T val) {
    if constexpr (std::is_signed_v<T>) {
      return val < T(0) ? -val : val;
    }
    return val;
  }

 public:
  template <class FPT>
  KOKKOS_FUNCTION static bool compare_near_zero(FPT const& fpv, int ulp) {
    auto abs_tol = eps(fpv) * ulp;

    bool ar = absolute(fpv) <= abs_tol;
    if (!ar) {
      Kokkos::printf("absolute value exceeds tolerance [|%e| > %e]\n",
                     (double)fpv, (double)abs_tol);
    }

    return ar;
  }

  template <class Lhs, class Rhs>
  KOKKOS_FUNCTION static bool compare(Lhs const& lhs, Rhs const& rhs, int ulp) {
    if (lhs == 0) {
      return compare_near_zero(rhs, ulp);
    } else if (rhs == 0) {
      return compare_near_zero(lhs, ulp);
    } else {
      auto rel_tol     = (eps(lhs) < eps(rhs) ? eps(lhs) : eps(rhs)) * ulp;
      double abs_diff  = static_cast<double>(rhs > lhs ? rhs - lhs : lhs - rhs);
      double min_denom = static_cast<double>(
          absolute(rhs) < absolute(lhs) ? absolute(rhs) : absolute(lhs));
      double rel_diff = abs_diff / min_denom;
      bool ar         = rel_diff <= rel_tol;
      if (!ar) {
        Kokkos::printf("relative difference exceeds tolerance [%e > %e]\n",
                       (double)rel_diff, (double)rel_tol);
      }

      return ar;
    }
  }
};

struct IntegerComparison {
  template <class Lhs, class Rhs>
  KOKKOS_FUNCTION static bool compare(Lhs const& lhs, Rhs const& rhs) {
    static_assert(std::is_integral_v<Lhs>);
    static_assert(std::is_integral_v<Rhs>);
    return lhs == rhs;
  }
};

// Check if a rank-0 view contains a given value.
template <typename Exec, typename ViewType>
::testing::AssertionResult contains(
    const Exec& exec, const ViewType& view,
    const typename ViewType::value_type& expected) {
  static_assert(ViewType::rank() == 0);
  typename ViewType::non_const_value_type value;
  Kokkos::deep_copy(exec, value, view);
  exec.fence();
  if (value != expected)
    return ::testing::AssertionFailure()
           << expected << " is not in " << view.label() << ", got " << value;
  else
    return ::testing::AssertionSuccess();
}

template <size_t string_capacity>
struct ErrorLog {
  Kokkos::Impl::StaticString<string_capacity> error_message;
};

// Dumbed down version of gtest's EXPECT_ functions, usable on the device
// (needs to be used inside test defined with KOKKOS_DEVICE_TEST)

// clang-format off
#define KOKKOS_EXPECT_EQ(arg1, arg2)                                         \
  do {                                                                       \
    if (!((arg1) == (arg2))) {                                               \
      auto* log = m_error_reporter.emplace_report();                         \
      if (log == nullptr) { break; }                                         \
      log->error_message << __FILE__ ":" << __LINE__ <<                      \
      ": Failure\nExpected equality of these values:\n  "                    \
      KOKKOS_IMPL_STRINGIFY(arg1) << "\n    Which is: " << arg1 <<           \
      "\n  " KOKKOS_IMPL_STRINGIFY(arg2) "\n    Which is: " << arg2 << "\n"; \
    }                                                                        \
  } while (false)

#define KOKKOS_EXPECT_NE(arg1, arg2)                                      \
  do {                                                                    \
    if (!((arg1) != (arg2))) {                                            \
      auto* log = m_error_reporter.emplace_report();                      \
      if (log == nullptr) { break; }                                      \
      log->error_message << __FILE__ ":" << __LINE__ <<                   \
      ": Failure\nExpected: (" KOKKOS_IMPL_STRINGIFY(arg1) << ") != ("    \
      KOKKOS_IMPL_STRINGIFY(arg2) "), actual: " << arg1 << " vs " << arg2 \
      << "\n";                                                            \
    }                                                                     \
  } while (false)

#define KOKKOS_EXPECT_TRUE(arg)                                      \
  do {                                                               \
    if (!(arg)) {                                                    \
      auto* log = m_error_reporter.emplace_report();                 \
      if (log == nullptr) { break; }                                 \
      log->error_message << __FILE__ ":" << __LINE__ <<              \
      ": Failure\nValue of: " KOKKOS_IMPL_STRINGIFY(arg)             \
      "\n  Actual: false\n  Expected: true\n";                       \
    }                                                                \
  } while(false)

#define KOKKOS_EXPECT_FALSE(arg)                                     \
  do {                                                               \
    if (!!(arg)) {                                                   \
      auto* log = m_error_reporter.emplace_report();                 \
      if (log == nullptr) { break; }                                 \
      log->error_message << __FILE__ ":" << __LINE__ <<              \
      ": Failure\nValue of: " KOKKOS_IMPL_STRINGIFY(arg)             \
      "\n  Actual: true\n  Expected: false\n";                       \
    }                                                                \
  } while(false)

#define KOKKOS_EXPECT_NEAR_ULPS(float1, float2, ulps)                          \
  do {                                                                         \
    if (!KokkosTest::FloatingPointComparison::compare(float1, float2, ulps)) { \
      auto* log = m_error_reporter.emplace_report();                           \
      if (log == nullptr) { break; }                                           \
      log->error_message << __FILE__ ":" << __LINE__ <<                        \
      ": Failure\nExpected: " KOKKOS_IMPL_STRINGIFY(float1)                    \
      " within %i ulps of " KOKKOS_IMPL_STRINGIFY(float2) ", \n  Actual: "     \
      << float1 << " vs " << float2 << "\n";                                   \
    }                                                                          \
  } while(false)

#if __FINITE_MATH_ONLY__
// Nothing to test if NaN and infinite are disabled at compilation
#define KOKKOS_EXPECT_NAN(arg)
#define KOKKOS_EXPECT_INF(arg)
#else
#define KOKKOS_EXPECT_NAN(arg)                           \
  do {                                                   \
    if (!Kokkos::isnan(arg)) {                           \
      auto* log = m_error_reporter.emplace_report();     \
      if (log == nullptr) { break; }                     \
      log->error_message << __FILE__ ":" << __LINE__ <<  \
      ": Failure\nValue of: " KOKKOS_IMPL_STRINGIFY(arg) \
      "\n  Actual: " << arg << "\nExpected: NaN\n";      \
    }                                                    \
  } while (false)

#define KOKKOS_EXPECT_INF(arg)                           \
  do {                                                   \
    if (!Kokkos::isinf(arg)) {                           \
      auto* log = m_error_reporter.emplace_report();     \
      if (log == nullptr) { break; }                     \
      log->error_message << __FILE__ ":" << __LINE__ <<  \
      ": Failure\nValue of: " KOKKOS_IMPL_STRINGIFY(arg) \
      "\n  Actual: " << arg << "\nExpected: +/-Inf\n";   \
    }                                                    \
  } while (false)
#endif
// clang-format on

#define KOKKOS_DEFINE_ERROR_REPORTER(string_capacity, ...)            \
  using Log = KokkosTest::ErrorLog<string_capacity>;                  \
  Kokkos::Experimental::ErrorReporter<Log __VA_OPT__(, ) __VA_ARGS__> \
      m_error_reporter

#define KOKKOS_INIT_ERROR_REPORTER(max_errors) m_error_reporter(max_errors)

#define KOKKOS_RETRIEVE_ERRORS(ret)                                \
  do {                                                             \
    std::vector<int> reporters;                                    \
    std::vector<Log> reports;                                      \
    std::tie(reporters, reports) = m_error_reporter.get_reports(); \
    for (int i = 0; i < m_error_reporter.num_reports(); ++i) {     \
      std::cout << reports[i].error_message << std::endl;          \
    }                                                              \
    ret = m_error_reporter.num_report_attempts();                  \
  } while (false)

// TODO: allow parametrization of error reporter capacity
// TODO: allow parametrization of ErrorLog capacity
/**
 * Create a test that will run on the device, first template argument is the
 * execution space where the test need to run, can take up to one extra
 * template argument.
 */
#define KOKKOS_DEVICE_TEST(TestName, ...)                                \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>             \
  struct [[nodiscard]] TestName {                                        \
    KOKKOS_DEFINE_ERROR_REPORTER(2048, TestSpace);                       \
    TestName() : KOKKOS_INIT_ERROR_REPORTER(10) {}                       \
                                                                         \
    [[nodiscard]] int run() const {                                      \
      Kokkos::parallel_for(Kokkos::RangePolicy<TestSpace>(0, 1), *this); \
                                                                         \
      int ret;                                                           \
      KOKKOS_RETRIEVE_ERRORS(ret);                                       \
      return ret;                                                        \
    }                                                                    \
    KOKKOS_FUNCTION void operator()(int) const;                          \
  };                                                                     \
                                                                         \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>             \
  KOKKOS_FUNCTION void                                                   \
  TestName<TestSpace __VA_OPT__(, ) __VA_ARGS__>::operator()(int) const

}  // namespace KokkosTest

#endif
