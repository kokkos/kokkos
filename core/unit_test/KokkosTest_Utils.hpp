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

    return absolute(fpv) <= abs_tol;
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
      return rel_diff <= rel_tol;
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

KOKKOS_INLINE_FUNCTION void* memcpy(void* dest, const void* src,
                                    std::size_t count) {
  char* dest_char      = reinterpret_cast<char*>(dest);
  const char* src_char = reinterpret_cast<const char*>(src);
  for (std::size_t i = 0; i < count; ++i) {
    *dest_char++ = *src_char++;
  }

  return dest;
}

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

enum class TypeIdentifier : char {
  c_string = 0x1,
  boolean,
  c,
  i8,
  i16,
  i32,
  i64,
  u8,
  u16,
  u32,
  u64,
  f16,
  b16,
  f32,
  f64,
  f128,
  longlong
};

template <typename T>
struct ToTypeIdentifier {
  static_assert(false, "Unsupported type");
  using T::ErrorTypeToDisplayNameInErrorMessage;

  static constexpr char value = 0;
};

#define KOKKOS_TO_TYPE_IDENTIFIER(type, type_identifier)    \
  template <>                                               \
  struct ToTypeIdentifier<type> {                           \
    static constexpr char value =                           \
        static_cast<char>(TypeIdentifier::type_identifier); \
  };

KOKKOS_TO_TYPE_IDENTIFIER(const char*, c_string)
KOKKOS_TO_TYPE_IDENTIFIER(bool, boolean)
KOKKOS_TO_TYPE_IDENTIFIER(char, c)
KOKKOS_TO_TYPE_IDENTIFIER(std::int8_t, i8)
KOKKOS_TO_TYPE_IDENTIFIER(std::int16_t, i16)
KOKKOS_TO_TYPE_IDENTIFIER(std::int32_t, i32)
KOKKOS_TO_TYPE_IDENTIFIER(std::int64_t, i64)
KOKKOS_TO_TYPE_IDENTIFIER(std::uint8_t, u8)
KOKKOS_TO_TYPE_IDENTIFIER(std::uint16_t, u16)
KOKKOS_TO_TYPE_IDENTIFIER(std::uint32_t, u32)
KOKKOS_TO_TYPE_IDENTIFIER(std::uint64_t, u64)
KOKKOS_TO_TYPE_IDENTIFIER(long long int, longlong)
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
KOKKOS_TO_TYPE_IDENTIFIER(Kokkos::Experimental::half_t, f16)
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
KOKKOS_TO_TYPE_IDENTIFIER(Kokkos::Experimental::bhalf_t, b16)
#endif
KOKKOS_TO_TYPE_IDENTIFIER(float, f32)
KOKKOS_TO_TYPE_IDENTIFIER(double, f64)
KOKKOS_TO_TYPE_IDENTIFIER(long double, f128)
#undef KOKKOS_TO_TYPE_IDENTIFIER

template <typename T>
inline void print_from_type_identifier(std::ostream& os, const char*& data) {
  T val;
  KokkosTest::memcpy(&val, data, sizeof(T));
  os << val;
  data += sizeof(T);
}

#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
template <>
inline void print_from_type_identifier<Kokkos::Experimental::half_t>(
    std::ostream& os, const char*& data) {
  Kokkos::Experimental::half_t val;
  KokkosTest::memcpy(&val, data, sizeof(Kokkos::Experimental::half_t));
  os << static_cast<float>(val);
  data += sizeof(bool);
}
#endif

#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
template <>
inline void print_from_type_identifier<Kokkos::Experimental::bhalf_t>(
    std::ostream& os, const char*& data) {
  Kokkos::Experimental::bhalf_t val;
  KokkosTest::memcpy(&val, data, sizeof(Kokkos::Experimental::bhalf_t));
  os << static_cast<float>(val);
  data += sizeof(bool);
}
#endif

template <>
inline void print_from_type_identifier<bool>(std::ostream& os,
                                             const char*& data) {
  bool val;
  KokkosTest::memcpy(&val, data, sizeof(bool));
  if (val) {
    os << "True";
  } else {
    os << "False";
  }
  data += sizeof(bool);
}

template <typename T>
inline void print_non_printable_chars(std::ostream& os, const char*& data) {
  T val;
  KokkosTest::memcpy(&val, data, sizeof(T));
  std::ostream::fmtflags flags = os.flags();

  if (val >= 0x20 && val <= 0x7E) {
    // Printable ASCII character
    os << '\'' << val << "' (" << std::dec << static_cast<int>(val) << ", 0x"
       << std::hex << std::uppercase << static_cast<int>(val) << ')';
  } else {
    // Non printable ASCII character
    os << "'0x" << std::hex << std::uppercase << static_cast<int>(val) << '\''
       << " (" << std::dec << static_cast<int>(val) << ")";
  }

  os.flags(flags);
  data += sizeof(T);
}

inline void print_variable(TypeIdentifier t, std::ostream& os,
                           const char*& data) {
  switch (t) {
    case TypeIdentifier::c_string:
      while (*data != '\0') os << *data++;
      ++data;
      break;
    case TypeIdentifier::boolean:
      print_from_type_identifier<bool>(os, data);
      break;
    case TypeIdentifier::c: print_non_printable_chars<char>(os, data); break;
    case TypeIdentifier::i8: print_non_printable_chars<int8_t>(os, data); break;
    case TypeIdentifier::i16:
      print_from_type_identifier<uint16_t>(os, data);
      break;
    case TypeIdentifier::i32:
      print_from_type_identifier<int32_t>(os, data);
      break;
    case TypeIdentifier::i64:
      print_from_type_identifier<int64_t>(os, data);
      break;
    case TypeIdentifier::u8:
      print_non_printable_chars<uint8_t>(os, data);
      break;
    case TypeIdentifier::u16:
      print_from_type_identifier<uint16_t>(os, data);
      break;
    case TypeIdentifier::u32:
      print_from_type_identifier<uint32_t>(os, data);
      break;
    case TypeIdentifier::u64:
      print_from_type_identifier<uint64_t>(os, data);
      break;
    case TypeIdentifier::f32:
      print_from_type_identifier<float>(os, data);
      break;
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
    case TypeIdentifier::f16:
      print_from_type_identifier<Kokkos::Experimental::half_t>(os, data);
      break;
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
    case TypeIdentifier::b16:
      print_from_type_identifier<Kokkos::Experimental::bhalf_t>(os, data);
      break;
#endif
    case TypeIdentifier::f64:
      print_from_type_identifier<double>(os, data);
      break;
    case TypeIdentifier::f128:
      print_from_type_identifier<long double>(os, data);
      break;
    case TypeIdentifier::longlong:
      print_from_type_identifier<long long int>(os, data);
      break;
    default:
      // Error while preparing string
      KOKKOS_ASSERT(false);
      break;
  }
}

template <std::size_t capacity>
class UnformatedString {
 private:
  char m_data[capacity];
  std::size_t m_size;
  bool m_full;

 public:
  KOKKOS_FUNCTION constexpr UnformatedString() : m_size(0), m_full(false) {}

  KOKKOS_FUNCTION constexpr UnformatedString(const char* str)
      : m_size(0), m_full(false) {
    *this << str;
  }

  template <typename T>
  KOKKOS_FUNCTION constexpr UnformatedString<capacity>& operator<<(const T i) {
    if (m_full) {
      return *this;
    }

    // +1 because we need room for the type tag
    std::size_t needed_size = sizeof(T) + sizeof(char);
    // Comparison written this way to avoid the risk of overflowing size_t
    if ((needed_size > capacity) || (capacity - needed_size < m_size)) {
      m_full = true;
      return *this;
    }

    m_data[m_size++] = ToTypeIdentifier<T>::value;
    KokkosTest::memcpy(&m_data[m_size], &i, sizeof(T));
    m_size += sizeof(T);

    return *this;
  }

  KOKKOS_FUNCTION constexpr UnformatedString<capacity>& operator<<(
      const char* string) {
    if (m_full) {
      return *this;
    }

    // Not enough space for \0 + type tag
    if (m_size + 2 >= capacity) {
      m_full = true;
      return *this;
    }

    m_data[m_size++] = ToTypeIdentifier<const char*>::value;

    for (; *string != '\0' && m_size < capacity; ++string) {
      m_data[m_size++] = *string;
    }

    if (m_size >= capacity) {
      // In case we exit because max capacity was reached, we need to append the
      // \0.
      m_data[capacity - 1] = '\0';
      m_full               = true;
      return *this;
    }

    m_data[m_size++] = '\0';

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const UnformatedString& unformatedString) {
    const char* data = unformatedString.m_data;
    while (static_cast<std::size_t>(data - unformatedString.m_data) <
           unformatedString.m_size) {
      TypeIdentifier var_type = static_cast<TypeIdentifier>(*(data++));
      print_variable(var_type, os, data);
    }

    if (unformatedString.m_full) {
      os << "...";
    }

    return os;
  }
};

template <size_t string_capacity>
struct TestLog {
  UnformatedString<string_capacity> error_message;
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
  using String = UnformatedString<capacity>;
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
  if (!(arg1_val == arg2_val)) {
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
  if (!(arg1_val != arg2_val)) {
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
  if (!static_cast<bool>(arg_val)) {
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
  if (static_cast<bool>(arg_val)) {
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
