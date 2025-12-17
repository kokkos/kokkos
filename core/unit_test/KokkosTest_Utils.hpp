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

namespace Impl {
// Print the value of t deducing a sensible format from its type T.
template <typename T>
constexpr KOKKOS_FUNCTION void print_arg_value(T t) {
  using type = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<T, char*> || std::is_same_v<T, const char*>) {
    if (t != nullptr) {
      // FIXME This is not 100% safe, as there is no guarantee that a char* is
      // a pointer to a C-style string
      Kokkos::printf("\"%s\"", t);
    } else {
      Kokkos::printf("NULL");
    }
  } else if constexpr (std::is_pointer_v<type>) {
    if (t != nullptr) {
      Kokkos::printf("%p", t);
    } else {
      Kokkos::printf("NULL");
    }
  } else if constexpr (std::is_same_v<type, bool>) {
    if (t) {
      Kokkos::printf("true");
    } else {
      Kokkos::printf("false");
    }
  } else if constexpr (std::is_integral_v<type>) {
    if constexpr (std::is_unsigned_v<type>) {
      Kokkos::printf("%u", t);
    } else {
      Kokkos::printf("%i", t);
    }
  } else if constexpr (std::is_floating_point_v<type>) {
    Kokkos::printf("%f", t);
  } else if constexpr (std::is_enum_v<type>) {
    Kokkos::printf("%i", t);
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
  } else if constexpr (std::is_same_v<T, Kokkos::Experimental::half_t> ||
                       std::is_same_v<T, Kokkos::Experimental::bhalf_t>) {
#if (defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_COMPILER_MSVC))
    // Bit cast is not available for half types with msvc
    Kokkos::printf("%f", static_cast<float>(t));
#else
    Kokkos::printf("%f (0x%X)", static_cast<float>(t),
                   Kokkos::bit_cast<uint16_t, T>(t));
#endif
#endif
  } else {
    Kokkos::abort("Unknown type, can't display value.");
  }
}
}  // namespace Impl

// Dumbed down version of gtest's EXPECT_ functions, usable on the device
// (needs to be used inside test defined with KOKKOS_DEVICE_TEST)
// FIXME the printf can get interleaved when executing in parallel, the whole
// error message should be constructed at once in a string and displayed in a
// printf, but there is no robust enough implementation of string on GPU to do
// that portably
#define KOKKOS_EXPECT_EQ(arg1, arg2)                                         \
  do {                                                                       \
    if (!((arg1) == (arg2))) {                                               \
      errors += 1;                                                           \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);                \
      Kokkos::printf("Expected equality of these values:\n");                \
      Kokkos::printf("  " KOKKOS_IMPL_STRINGIFY(arg1) "\n    Which is: ");   \
      KokkosTest::Impl::print_arg_value(arg1);                               \
      Kokkos::printf("\n  " KOKKOS_IMPL_STRINGIFY(arg2) "\n    Which is: "); \
      KokkosTest::Impl::print_arg_value(arg2);                               \
      Kokkos::printf("\n");                                                  \
    }                                                                        \
  } while (false)

#define KOKKOS_EXPECT_NE(arg1, arg2)                                 \
  do {                                                               \
    if (!((arg1) != (arg2))) {                                       \
      errors += 1;                                                   \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);        \
      Kokkos::printf("Expected: (" KOKKOS_IMPL_STRINGIFY(            \
          arg1) ") != (" KOKKOS_IMPL_STRINGIFY(arg2) "), actual: "); \
      KokkosTest::Impl::print_arg_value(arg1);                       \
      Kokkos::printf(" vs ");                                        \
      KokkosTest::Impl::print_arg_value(arg2);                       \
      Kokkos::printf("\n");                                          \
    }                                                                \
  } while (false)

#define KOKKOS_EXPECT_TRUE(arg)                               \
  do {                                                        \
    if (!(arg)) {                                             \
      errors += 1;                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__); \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(      \
          arg) "\n  Actual: false\nExpected: true\n");        \
    }                                                         \
  } while (false)

#define KOKKOS_EXPECT_FALSE(arg)                              \
  do {                                                        \
    if (!!(arg)) {                                            \
      errors += 1;                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__); \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(      \
          arg) "\n  Actual: true\nExpected: false\n");        \
    }                                                         \
  } while (false)

// clang-format off
#define KOKKOS_EXPECT_NEAR_ULPS(float1, float2, ulps)           \
  do {                                                          \
    if (!KokkosTest::FloatingPointComparison                    \
            ::compare(float1, float2, ulps)) {                  \
      errors += 1;                                              \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);   \
      Kokkos::printf("Expected: " KOKKOS_IMPL_STRINGIFY(float1) \
           " within %i ulps of " KOKKOS_IMPL_STRINGIFY(float2)  \
           ",\n  Actual: ", ulps);                              \
      KokkosTest::Impl::print_arg_value(float1);                \
      Kokkos::printf(" vs ");                                   \
      KokkosTest::Impl::print_arg_value(float2);                \
      Kokkos::printf("\n");                                     \
    }                                                           \
  } while (false)
// clang-format on

#if __FINITE_MATH_ONLY__
// Nothing to test if NaN and infinite are disabled at compilation
#define KOKKOS_EXPECT_NAN(arg)
#define KOKKOS_EXPECT_INF(arg)
#else
#define KOKKOS_EXPECT_NAN(arg)                                                \
  do {                                                                        \
    if (!Kokkos::isnan(arg)) {                                                \
      errors += 1;                                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);                 \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(arg) "\n  Actual: "); \
      KokkosTest::Impl::print_arg_value(arg);                                 \
      Kokkos::printf("\nExpected: NaN\n");                                    \
    }                                                                         \
  } while (false)

#define KOKKOS_EXPECT_INF(arg)                                                \
  do {                                                                        \
    if (!Kokkos::isinf(arg)) {                                                \
      errors += 1;                                                            \
      Kokkos::printf("%s:%i: Failure\n", __FILE__, __LINE__);                 \
      Kokkos::printf("Value of: " KOKKOS_IMPL_STRINGIFY(arg) "\n  Actual: "); \
      KokkosTest::Impl::print_arg_value(arg);                                 \
      Kokkos::printf("\nExpected: +/-Inf\n");                                 \
    }                                                                         \
  } while (false)
#endif

/**
 * Create a test that will run on the device, first template argument is the
 * execution space where the test need to run, can take up to one extra
 * template argument.
 */
#define KOKKOS_DEVICE_TEST(TestName, ...)                                      \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>                   \
  struct [[nodiscard]] TestName {                                              \
    [[nodiscard]] int run() const {                                            \
      int errors = 0;                                                          \
      Kokkos::parallel_reduce(Kokkos::RangePolicy<TestSpace>(0, 1), *this,     \
                              errors);                                         \
      return errors;                                                           \
    }                                                                          \
    KOKKOS_FUNCTION void operator()(int, int& errors) const;                   \
  };                                                                           \
                                                                               \
  template <class TestSpace __VA_OPT__(, class) __VA_ARGS__>                   \
  KOKKOS_FUNCTION void                                                         \
  TestName<TestSpace __VA_OPT__(, ) __VA_ARGS__>::operator()(int, int& errors) \
      const

}  // namespace KokkosTest

#endif
