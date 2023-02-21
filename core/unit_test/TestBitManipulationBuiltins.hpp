//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// clang-format off
template <class>
struct type_helper;
#define DEFINE_TYPE_NAME(T) \
template <> struct type_helper<T> { static char const * name() { return #T; } };
DEFINE_TYPE_NAME(unsigned char)
DEFINE_TYPE_NAME(unsigned short)
DEFINE_TYPE_NAME(unsigned int)
DEFINE_TYPE_NAME(unsigned long)
DEFINE_TYPE_NAME(unsigned long long)
#undef DEFINE_TYPE_NAME
// clang-format on

#define DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(FUNC)   \
  struct BitManipFunction_##FUNC {                    \
    template <class T>                                \
    static KOKKOS_FUNCTION auto eval_constexpr(T x) { \
      return Kokkos::FUNC(x);                         \
    }                                                 \
    template <class T>                                \
    static KOKKOS_FUNCTION auto eval_builtin(T x) {   \
      return Kokkos::Experimental::FUNC##_builtin(x); \
    }                                                 \
    static char const* name() { return #FUNC; }       \
  }

DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countl_zero);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countl_one);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countr_zero);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(countr_one);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(popcount);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(has_single_bit);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_ceil);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_floor);
DEFINE_BIT_MANIPULATION_FUNCTION_EVAL(bit_width);

template <class Space, class Func, class Arg, std::size_t N>
struct TestBitManipFunction {
  Arg val_[N];
  TestBitManipFunction(const Arg (&val)[N]) {
    std::copy(val, val + N, val_);
    run();
  }
  void run() const {
    int errors = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<Space>(0, N), *this, errors);
    ASSERT_EQ(errors, 0) << "Failed check no error for " << Func::name() << "("
                         << type_helper<Arg>::name() << ")";
  }
  KOKKOS_FUNCTION void operator()(int i, int& e) const {
    if (Func::eval_builtin(val_[i]) != Func::eval_constexpr(val_[i])) {
      ++e;
      KOKKOS_IMPL_DO_NOT_USE_PRINTF(
          "value at %x which is %d was expected to be %d\n", (unsigned)val_[i],
          (int)Func::eval_builtin(val_[i]), (int)Func::eval_constexpr(val_[i]));
    }
  }
};

template <class Space, class... Func, class Arg, std::size_t N>
void do_test_bit_manip_function(const Arg (&x)[N]) {
  (void)std::initializer_list<int>{
      (TestBitManipFunction<Space, Func, Arg, N>(x), 0)...};
}

#define TEST_BIT_MANIP_FUNCTION(FUNC) \
  do_test_bit_manip_function<TEST_EXECSPACE, BitManipFunction_##FUNC>

template <class UInt>
void test_bit_manip_countl_zero() {
  using Kokkos::Experimental::countl_zero_builtin;
  static_assert(noexcept(countl_zero_builtin(UInt())));
  static_assert(std::is_same_v<decltype(countl_zero_builtin(UInt())), int>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(countl_zero)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(127),
      UInt(128),
      UInt(max),
  });
}

TEST(TEST_CATEGORY, bit_manip_countl_zero) {
  test_bit_manip_countl_zero<unsigned char>();
  test_bit_manip_countl_zero<unsigned short>();
  test_bit_manip_countl_zero<unsigned int>();
  test_bit_manip_countl_zero<unsigned long>();
  test_bit_manip_countl_zero<unsigned long long>();
}

template <class UInt>
void test_bit_manip_countl_one() {
  using Kokkos::Experimental::countl_one_builtin;
  static_assert(noexcept(countl_one_builtin(UInt())));
  static_assert(std::is_same_v<decltype(countl_one_builtin(UInt())), int>);
  constexpr auto dig = Kokkos::Experimental::digits_v<UInt>;
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(countl_one)
  ({
      // clang-format off
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(100),
      UInt(127),
      UInt(128),
      UInt(max),
      UInt(max - 1),
      UInt(max - 2),
      UInt(max - 3),
      UInt(max - 4),
      UInt(max - 5),
      UInt(max - 6),
      UInt(max - 7),
      UInt(max - 8),
      UInt(max - 9),
      UInt(max - 126),
      UInt(max - 127),
      UInt(max - 128),
      UInt(UInt(1) << (dig - 1)),
      UInt(UInt(3) << (dig - 2)),
      UInt(UInt(7) << (dig - 3)),
      UInt(UInt(255) << (dig - 8)),
      // clang-format on
  });
}

TEST(TEST_CATEGORY, bit_manip_countl_one) {
  test_bit_manip_countl_one<unsigned char>();
  test_bit_manip_countl_one<unsigned short>();
  test_bit_manip_countl_one<unsigned int>();
  test_bit_manip_countl_one<unsigned long>();
  test_bit_manip_countl_one<unsigned long long>();
}

template <class UInt>
void test_bit_manip_countr_zero() {
  using Kokkos::Experimental::countr_zero_builtin;
  static_assert(noexcept(countr_zero_builtin(UInt())));
  static_assert(std::is_same_v<decltype(countr_zero_builtin(UInt())), int>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(countr_zero)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(126),
      UInt(127),
      UInt(128),
      UInt(129),
      UInt(130),
      UInt(max),
  });
}

TEST(TEST_CATEGORY, bit_manip_countr_zero) {
  test_bit_manip_countr_zero<unsigned char>();
  test_bit_manip_countr_zero<unsigned short>();
  test_bit_manip_countr_zero<unsigned int>();
  test_bit_manip_countr_zero<unsigned long>();
  test_bit_manip_countr_zero<unsigned long long>();
}

template <class UInt>
void test_bit_manip_countr_one() {
  using Kokkos::Experimental::countr_one_builtin;
  static_assert(noexcept(countr_one_builtin(UInt())));
  static_assert(std::is_same_v<decltype(countr_one_builtin(UInt())), int>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(countr_one)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(126),
      UInt(127),
      UInt(128),
      UInt(max - 1),
      UInt(max),
  });
}

TEST(TEST_CATEGORY, bit_manip_countr_one) {
  test_bit_manip_countr_one<unsigned char>();
  test_bit_manip_countr_one<unsigned short>();
  test_bit_manip_countr_one<unsigned int>();
  test_bit_manip_countr_one<unsigned long>();
  test_bit_manip_countr_one<unsigned long long>();
}

template <class UInt>
void test_bit_manip_popcount() {
  using Kokkos::Experimental::popcount_builtin;
  static_assert(noexcept(popcount_builtin(UInt())));
  static_assert(std::is_same_v<decltype(popcount_builtin(UInt())), int>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(popcount)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(127),
      UInt(max),
      UInt(max - 1),
  });
}

TEST(TEST_CATEGORY, bit_manip_popcount) {
  test_bit_manip_popcount<unsigned char>();
  test_bit_manip_popcount<unsigned short>();
  test_bit_manip_popcount<unsigned int>();
  test_bit_manip_popcount<unsigned long>();
  test_bit_manip_popcount<unsigned long long>();
}

template <class UInt>
void test_bit_manip_has_single_bit() {
  using Kokkos::Experimental::has_single_bit_builtin;
  static_assert(noexcept(has_single_bit_builtin(UInt())));
  static_assert(std::is_same_v<decltype(has_single_bit_builtin(UInt())), bool>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  constexpr UInt one = 1;
  TEST_BIT_MANIP_FUNCTION(has_single_bit)
  ({
      // clang-format off
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(max),
      UInt(one << 0),
      UInt(one << 1),
      UInt(one << 2),
      UInt(one << 3),
      UInt(one << 4),
      UInt(one << 5),
      UInt(one << 6),
      UInt(one << 7),
      // clang-format on
  });
}

TEST(TEST_CATEGORY, bit_manip_has_single_bit) {
  test_bit_manip_has_single_bit<unsigned char>();
  test_bit_manip_has_single_bit<unsigned short>();
  test_bit_manip_has_single_bit<unsigned int>();
  test_bit_manip_has_single_bit<unsigned long>();
  test_bit_manip_has_single_bit<unsigned long long>();
}

template <class UInt>
void test_bit_manip_bit_floor() {
  using Kokkos::Experimental::bit_floor_builtin;
  static_assert(noexcept(bit_floor_builtin(UInt())));
  static_assert(std::is_same_v<decltype(bit_floor_builtin(UInt())), UInt>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(bit_floor)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(125),
      UInt(126),
      UInt(127),
      UInt(128),
      UInt(129),
      UInt(max),
  });
}

TEST(TEST_CATEGORY, bit_manip_bit_floor) {
  test_bit_manip_bit_floor<unsigned char>();
  test_bit_manip_bit_floor<unsigned short>();
  test_bit_manip_bit_floor<unsigned int>();
  test_bit_manip_bit_floor<unsigned long>();
  test_bit_manip_bit_floor<unsigned long long>();
}

template <class UInt>
void test_bit_manip_bit_ceil() {
  using Kokkos::Experimental::bit_ceil_builtin;
  static_assert(noexcept(bit_ceil_builtin(UInt())));
  static_assert(std::is_same_v<decltype(bit_ceil_builtin(UInt())), UInt>);
  TEST_BIT_MANIP_FUNCTION(bit_ceil)
  ({
      // clang-format off
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(60),
      UInt(61),
      UInt(62),
      UInt(63),
      UInt(64),
      UInt(65),
      UInt(66),
      UInt(67),
      UInt(68),
      UInt(69),
      // clang-format on
  });
}

TEST(TEST_CATEGORY, bit_manip_bit_ceil) {
  test_bit_manip_bit_ceil<unsigned char>();
  test_bit_manip_bit_ceil<unsigned short>();
  test_bit_manip_bit_ceil<unsigned int>();
  test_bit_manip_bit_ceil<unsigned long>();
  test_bit_manip_bit_ceil<unsigned long long>();
}

template <class UInt>
void test_bit_manip_bit_width() {
  using Kokkos::Experimental::bit_width_builtin;
  static_assert(noexcept(bit_width_builtin(UInt())));
  static_assert(std::is_same_v<decltype(bit_width_builtin(UInt())), UInt>);
  constexpr auto max = Kokkos::Experimental::finite_max_v<UInt>;
  TEST_BIT_MANIP_FUNCTION(bit_width)
  ({
      UInt(0),
      UInt(1),
      UInt(2),
      UInt(3),
      UInt(4),
      UInt(5),
      UInt(6),
      UInt(7),
      UInt(8),
      UInt(9),
      UInt(max - 1),
      UInt(max),
  });
}

TEST(TEST_CATEGORY, bit_manip_bit_width) {
  test_bit_manip_bit_width<unsigned char>();
  test_bit_manip_bit_width<unsigned short>();
  test_bit_manip_bit_width<unsigned int>();
  test_bit_manip_bit_width<unsigned long>();
  test_bit_manip_bit_width<unsigned long long>();
}
