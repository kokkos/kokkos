// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

struct TestStaticString {
  TestStaticString() { EXPECT_EQ(run(), 0); }

  int run() const {
    int nerrors;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), *this,
                            nerrors);
    return nerrors;
  }

  template <std::size_t capacity>
  KOKKOS_FUNCTION int check(const Kokkos::Impl::StaticString<capacity>& ss,
                            const char* ref) const {
    if (Kokkos::Impl::strncmp(ss.c_str(), ref, ss.size() + 1) != 0) {
      Kokkos::printf("Error: %s != %s\n", ss.c_str(), ref);
      return 1;
    }
    return 0;
  }

  KOKKOS_FUNCTION void operator()(int, int& errors) const {
    {
      // Empty constructor should return an empty valid cstring
      Kokkos::Impl::StaticString<10> ss;
      const char* ref = "";
      errors += check(ss, ref);
    }
    {
      // Constructor taking a cstring
      Kokkos::Impl::StaticString<10> ss("test");
      const char* ref = "test";
      errors += check(ss, ref);
    }
    {
      // Append cstring
      Kokkos::Impl::StaticString<10> ss("test");
      ss << "test";
      const char* ref = "testtest";
      errors += check(ss, ref);
    }
    {
      // Exceed string capacity in constructor
      Kokkos::Impl::StaticString<10> ss("testtesttest");
      const char* ref = "testte...";
      errors += check(ss, ref);
    }
    {
      // Append int
      Kokkos::Impl::StaticString<10> ss("test");
      ss << 5;
      const char* ref = "test5";
      errors += check(ss, ref);
    }
    {
      // Completely fill the string without exceeding the capacity
      Kokkos::Impl::StaticString<10> ss("testtest");
      ss << 5;
      const char* ref = "testtest5";
      errors += check(ss, ref);
    }
    {
      // Fill the string and try to append an int
      Kokkos::Impl::StaticString<10> ss("testtestt");
      ss << 5;
      const char* ref = "testte...";
      errors += check(ss, ref);
    }
    {
      // Fill the string and try to append a cstring
      Kokkos::Impl::StaticString<10> ss("testtestt");
      ss << "test";
      const char* ref = "testte...";
      errors += check(ss, ref);
    }
    {
      // Try to write an int to big for the capacity, before appending a string
      Kokkos::Impl::StaticString<10> ss;
      ss << 1'000'000'000 << "test";
      const char* ref = "...";
      errors += check(ss, ref);
    }
    {
      // Multiple append
      Kokkos::Impl::StaticString<250> ss;
      ss << "test" << 5 << "test" << 5 << "test";
      const char* ref = "test5test5test";
      errors += check(ss, ref);
    }
    {
      // Append Booleans
      Kokkos::Impl::StaticString<10> ss;
      ss << true << false;
      const char* ref = "TrueFalse";
      errors += check(ss, ref);
    }
    {
      // Append Booleans without enough capacity
      Kokkos::Impl::StaticString<10> ss;
      ss << "test" << true << false;
      const char* ref = "testTr...";
      errors += check(ss, ref);
    }
    {
      // Append Float
      Kokkos::Impl::StaticString<15> ss;
      ss << 1.5f;
      const char* ref = "1";
      errors += check(ss, ref);
    }
    {
      // Append Double
      Kokkos::Impl::StaticString<15> ss;
      ss << 1.5;
      const char* ref = "1";
      errors += check(ss, ref);
    }
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
    {
      // Append half_t
      Kokkos::Impl::StaticString<20> ss;
      ss << Kokkos::Experimental::half_t(1.5f);
      const char* ref = "1";
      errors += check(ss, ref);
    }
#endif
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
    {
      // Append half_t
      Kokkos::Impl::StaticString<20> ss;
      ss << Kokkos::Experimental::bhalf_t(1.5f);
      const char* ref = "1";
      errors += check(ss, ref);
    }
#endif
  }
};

TEST(TEST_CATEGORY, TestStaticString) { TestStaticString(); }

}  // namespace
