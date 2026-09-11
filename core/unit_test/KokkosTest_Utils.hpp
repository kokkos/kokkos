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

template <size_t string_capacity>
struct TestLog {
  Kokkos::Impl::StaticString<string_capacity> error_message;
};

template <class Space, size_t string_capacity>
struct TestReporter {
  using Log      = KokkosTest::TestLog<string_capacity>;
  using Reporter = Kokkos::Experimental::ErrorReporter<Log, Space>;

  TestReporter() : m_reporter(10) {}

  TestReporter(int size) : m_reporter(size) {}

 private:
  Reporter m_reporter;

 public:
  KOKKOS_FUNCTION Log* emplace_report() const {
    return m_reporter.emplace_report();
  }

  int print_errors(std::ostream& os = std::cerr) {
    std::vector<int> reporters;
    std::vector<Log> reports;
    std::tie(reporters, reports) = m_reporter.get_reports();
    for (int i = 0; i < m_reporter.num_reports(); ++i) {
      os << reports[i].error_message << std::endl;
    }
    int ret = m_reporter.num_report_attempts();
    m_reporter.clear();

    return ret;
  }
};

namespace Impl {
template <size_t capacity>
struct OptionalString {
  using String = Kokkos::Impl::StaticString<capacity>;
  String* m_ss;

  KOKKOS_FUNCTION OptionalString(String* ss = nullptr) : m_ss(ss) {}

  template <typename Arg>
  KOKKOS_FUNCTION OptionalString<capacity>& operator<<(const Arg& arg) {
    if (m_ss != nullptr) {
      *m_ss << arg;
    }
    return *this;
  }
};

template <class Space, size_t string_capacity, class T1, class T2>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_eq(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg1, const T1 arg1_val,
    const char* arg2, const T2 arg2_val) {
  if (!((arg1_val) == (arg2_val))) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nExpected equality of these values:\n  "
                         << arg1 << "\n    Which is: " << arg1_val << "\n  "
                         << arg2 << "\n    Which is: " << arg2_val << "\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class T1, class T2>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_ne(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg1, const T1 arg1_val,
    const char* arg2, const T2 arg2_val) {
  if (!((arg1_val) != (arg2_val))) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line << ": Failure\nExpected: ("
                         << arg1 << ") != (" << arg2
                         << "), actual: " << arg1_val << " vs " << arg2_val
                         << "\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class T>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_true(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg, const T arg_val) {
  if (!(arg_val)) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nValue of: " << arg
                         << "\n  Actual: false\n  Expected: true\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class T>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_false(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg, const T arg_val) {
  if (!!(arg_val)) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nValue of: " << arg
                         << "\n  Actual: true\n  Expected: false\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class Float1, class Float2>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_near_ulps(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* float1, const Float1 float1_val,
    const char* float2, const Float2 float2_val, int ulps) {
  if (!KokkosTest::FloatingPointComparison::compare(float1_val, float2_val,
                                                    ulps)) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nExpected: " << float1 << " within "
                         << ulps << " ulps of " << float2
                         << ", \n  Actual: " << float1_val << " vs "
                         << float2_val << "\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class T>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_nan(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg, const T arg_val) {
  if (!Kokkos::isnan(arg_val)) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nValue of: " << arg
                         << "\n  Actual: " << arg_val << "\nExpected: NaN\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}

template <class Space, size_t string_capacity, class T>
KOKKOS_FUNCTION OptionalString<string_capacity> expect_inf(
    const KokkosTest::TestReporter<Space, string_capacity>& test_reporter,
    const char* file, int line, const char* arg, const T arg_val) {
  if (!Kokkos::isinf(arg_val)) {
    auto* log = test_reporter.emplace_report();
    if (log != nullptr) {
      log->error_message << file << ":" << line
                         << ": Failure\nValue of: " << arg
                         << "\n  Actual: " << arg_val << "\nExpected: Inf\n";

      return OptionalString<string_capacity>(&log->error_message);
    }
  }
  return OptionalString<string_capacity>();
}
}  // namespace Impl

#define KOKKOS_EXPECT_EQ(test_reporter, arg1, arg2)              \
  KokkosTest::Impl::expect_eq(test_reporter, __FILE__, __LINE__, \
                              KOKKOS_IMPL_STRINGIFY(arg1), arg1, \
                              KOKKOS_IMPL_STRINGIFY(arg2), arg2)

#define KOKKOS_EXPECT_NE(test_reporter, arg1, arg2)              \
  KokkosTest::Impl::expect_ne(test_reporter, __FILE__, __LINE__, \
                              KOKKOS_IMPL_STRINGIFY(arg1), arg1, \
                              KOKKOS_IMPL_STRINGIFY(arg2), arg2)

#define KOKKOS_EXPECT_TRUE(test_reporter, arg)                     \
  KokkosTest::Impl::expect_true(test_reporter, __FILE__, __LINE__, \
                                KOKKOS_IMPL_STRINGIFY(arg), arg)

#define KOKKOS_EXPECT_FALSE(test_reporter, arg)                     \
  KokkosTest::Impl::expect_false(test_reporter, __FILE__, __LINE__, \
                                 KOKKOS_IMPL_STRINGIFY(arg), arg)

#define KOKKOS_EXPECT_NEAR_ULPS(test_reporter, float1, float2, ulps)    \
  KokkosTest::Impl::expect_near_ulps(                                   \
      test_reporter, __FILE__, __LINE__, KOKKOS_IMPL_STRINGIFY(float1), \
      float1, KOKKOS_IMPL_STRINGIFY(float2), float2, ulps)

#if __FINITE_MATH_ONLY__
// Nothing to test if NaN and infinite are disabled at compilation
#define KOKKOS_EXPECT_NAN(test_reporter, arg)
#define KOKKOS_EXPECT_INF(test_reporter, arg)
#else
#define KOKKOS_EXPECT_NAN(test_reporter, arg)                     \
  KokkosTest::Impl::expect_nan(test_reporter, __FILE__, __LINE__, \
                               KOKKOS_IMPL_STRINGIFY(arg), arg)

#define KOKKOS_EXPECT_INF(test_reporter, arg)                     \
  KokkosTest::Impl::expect_inf(test_reporter, __FILE__, __LINE__, \
                               KOKKOS_IMPL_STRINGIFY(arg), arg)
#endif

/**
 * Create a test that will run on the device, first template argument is the
 * execution space where the test need to run, can take up to one extra
 * template argument.
 */
#define KOKKOS_DEVICE_TEST(TestName, ...)                                \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>             \
  struct [[nodiscard]] TestName {                                        \
    KokkosTest::TestReporter<TestSpace, 2048> m_test_reporter;           \
                                                                         \
    [[nodiscard]] int run() {                                            \
      Kokkos::parallel_for(Kokkos::RangePolicy<TestSpace>(0, 1), *this); \
      return m_test_reporter.print_errors();                             \
    }                                                                    \
    KOKKOS_FUNCTION void operator()(int) const;                          \
  };                                                                     \
                                                                         \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>             \
  KOKKOS_FUNCTION void                                                   \
  TestName<TestSpace __VA_OPT__(, ) __VA_ARGS__>::operator()(int) const

}  // namespace KokkosTest

#endif
