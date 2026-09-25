// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include <type_traits>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

TEST(defaultdevicetype, pair_construction_and_std_interop) {
  // value construction and member access
  constexpr Kokkos::pair p1{1, 2};
  static_assert(p1.first == 1 && p1.second == 2);
  EXPECT_EQ(p1.first, 1);
  EXPECT_EQ(p1.second, 2);

  // make_pair
  constexpr auto mp = Kokkos::make_pair(5u, 6u);
  static_assert(
      std::is_same_v<decltype(mp), const Kokkos::pair<unsigned, unsigned>>);
  static_assert(mp.first == 5u && mp.second == 6u);

  // construction from std::pair and to_std_pair()
  const std::pair<int, int> sp{3, 4};
  Kokkos::pair kp_from_std(sp);
  EXPECT_EQ(kp_from_std.first, 3);
  EXPECT_EQ(kp_from_std.second, 4);

  // to_std_pair is host-only; verify conversion
  auto sp2 = kp_from_std.to_std_pair();
  EXPECT_EQ(sp2.first, 3);
  EXPECT_EQ(sp2.second, 4);

  // deduction guide from std::pair
  std::pair ded_sp    = {7, 8};
  Kokkos::pair ded_kp = ded_sp;  // CTAD should deduce pair<int,int>
  static_assert(std::is_same_v<decltype(ded_kp), Kokkos::pair<int, int>>);
  EXPECT_EQ(ded_kp.first, 7);
  EXPECT_EQ(ded_kp.second, 8);
}

TEST(defaultdevicetype, pair_copy_and_assignment_templates) {
  Kokkos::pair p1{10, 11};

  // templated copy construction to different, but convertible types
  Kokkos::pair<long, long> pcopy(p1);
  EXPECT_EQ(pcopy.first, 10);
  EXPECT_EQ(pcopy.second, 11);

  // templated assignment operator
  Kokkos::pair<long, long> passign{0, 0};
  passign = p1;
  EXPECT_EQ(passign.first, 10);
  EXPECT_EQ(passign.second, 11);

  // construction from another Kokkos::pair with same types
  Kokkos::pair<int, int> p2 = p1;
  EXPECT_EQ(p2.first, 10);
  EXPECT_EQ(p2.second, 11);
}

TEST(defaultdevicetype, pair_relational_operators_and_constexpr) {
  constexpr Kokkos::pair a{1, 2};
  constexpr Kokkos::pair b{2, 1};
  constexpr Kokkos::pair a_same{1, 2};

  static_assert(a == a_same);
  static_assert(!(a != a_same));
  static_assert(a < b);
  static_assert(a <= a_same);
  static_assert(b > a);
  static_assert(b >= a);

  EXPECT_TRUE(a == a_same);
  EXPECT_FALSE(a != a_same);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a <= a_same);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);
}

TEST(defaultdevicetype, tie_and_reference_specialization_assignment) {
  int x = 0, y = 0;

  // tie() returns a pair of references
  auto ref_pair = Kokkos::tie(x, y);
  static_assert(std::is_same_v<decltype(ref_pair), Kokkos::pair<int&, int&>>);

  // assign from a value pair -> should assign through references
  ref_pair = Kokkos::make_pair(9, 10);
  EXPECT_EQ(x, 9);
  EXPECT_EQ(y, 10);

  // assignment operator template on reference specialization
  int a = 1, b = 2;
  Kokkos::pair<int&, int&> p_ref = Kokkos::tie(a, b);
  Kokkos::pair p_val{42, 43};
  p_ref = p_val;  // templated operator= should copy into a,b
  EXPECT_EQ(a, 42);
  EXPECT_EQ(b, 43);
}

template <typename ViewType>
struct DeviceTest {
  ViewType results;
  DeviceTest(ViewType r) : results(r) {}

  KOKKOS_FUNCTION void operator()(const int) const {
    // value construction
    Kokkos::pair p{1, 2};
    if (p.first == 1 && p.second == 2) results(0) = 1;

    // make_pair
    auto mp = Kokkos::make_pair(5, 6);
    if (mp.first == 5 && mp.second == 6) results(1) = 1;

    // relational
    Kokkos::pair a{1, 2}, b{2, 1};
    if (a < b) results(2) = 1;

    // tie and reference assignment
    int x = 0, y = 0;
    auto refp = Kokkos::tie(x, y);
    refp      = Kokkos::make_pair(9, 10);
    if (x == 9 && y == 10) results(3) = 1;

    // templated copy construction
    Kokkos::pair<long, long> pcopy(a);
    if (pcopy.first == 1 && pcopy.second == 2) results(4) = 1;

    // reference specialization assignment from value pair
    int a2 = 0, b2 = 0;
    Kokkos::pair<int&, int&> pref = Kokkos::tie(a2, b2);
    pref                          = Kokkos::pair(42, 43);
    if (a2 == 42 && b2 == 43) results(5) = 1;
  }
};

TEST(defaultdevicetype, pair_on_device) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<int*, exec_space>;

  view_t results("pair_device_results", 6);
  Kokkos::deep_copy(results, 0);

  Kokkos::parallel_for(Kokkos::RangePolicy<exec_space>(0, 1),
                       DeviceTest(results));
  Kokkos::fence();

  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  for (int i = 0; i < 6; ++i) EXPECT_EQ(h(i), 1);
}

constexpr bool test_pair_converting_constructor_from_std_pair() {
  constexpr std::pair<int, float> sp{1, 2.f};
  constexpr Kokkos::pair<int, float> p(sp);
  if (p.first != 1 || p.second != 2.f) return false;

  // constructor is also a template, allowing conversion between element types
  constexpr std::pair<int, int> sp2{3, 4};
  constexpr Kokkos::pair<long, long> p2(sp2);
  return p2.first == 3 && p2.second == 4;
}

static_assert(test_pair_converting_constructor_from_std_pair());

KOKKOS_INLINE_FUNCTION auto generate_pair() {
  const auto pair = Kokkos::make_pair(11, 12.0);

  auto [f, s] = pair;
  static_assert(std::same_as<decltype(f), int> &&
                std::same_as<decltype(s), double>);
  return pair;
}

template <typename ViewType>
struct TupleInterfaceDeviceTest {
  ViewType results;

  KOKKOS_FUNCTION void operator()(int) const {
    // Const lvalue
    {
      // On GCC 13 and below there is an issue with the way structure bindings
      // work from templates so this is pulled outside of the template
      const auto pair = generate_pair();

      static_assert(std::tuple_size_v<decltype(pair)> == 2);
      static_assert(
          std::same_as<std::tuple_element_t<0, decltype(pair)>, const int>);
      static_assert(
          std::same_as<std::tuple_element_t<1, decltype(pair)>, const double>);

      // Test structured binding
      auto [f, s] = pair;
      results(0)  = (f == 11);
      results(1)  = (s == 12.0);

      // test get
      // FIXME_NVCC: NVCC does not correctly ADL-find this overload so we have
      // to qualify the calls structured binding decomposition is not affected
      // even though it's defined to use unqualified calls to get
      auto f2 = Kokkos::get<0>(pair);
      auto s2 = Kokkos::get<1>(pair);
      static_assert(std::same_as<decltype(f2), int> &&
                    std::same_as<decltype(s2), double>);
      results(2) = (f2 == 11);
      results(3) = (s2 == 12.0);

      auto f3 = Kokkos::get<int>(pair);
      auto s3 = Kokkos::get<double>(pair);
      static_assert(std::same_as<decltype(f3), int> &&
                    std::same_as<decltype(s3), double>);
      results(4) = (f3 == 11);
      results(5) = (s3 == 12.0);
    }

    // Non-const lvalue
    {
      auto pair = generate_pair();

      static_assert(std::tuple_size_v<decltype(pair)> == 2);
      static_assert(std::same_as<std::tuple_element_t<0, decltype(pair)>, int>);
      static_assert(
          std::same_as<std::tuple_element_t<1, decltype(pair)>, double>);

      // Test structured binding
      auto [f, s] = pair;
      results(6)  = (f == 11);
      results(7)  = (s == 12.0);

      // test get
      // FIXME_NVCC: NVCC does not correctly ADL-find this overload so we have
      // to qualify the calls structured binding decomposition is not affected
      // even though it's defined to use unqualified calls to get
      auto f2 = Kokkos::get<0>(pair);
      auto s2 = Kokkos::get<1>(pair);
      static_assert(std::same_as<decltype(f2), int> &&
                    std::same_as<decltype(s2), double>);
      results(8) = (f2 == 11);
      results(9) = (s2 == 12.0);

      auto f3 = Kokkos::get<int>(pair);
      auto s3 = Kokkos::get<double>(pair);
      static_assert(std::same_as<decltype(f3), int> &&
                    std::same_as<decltype(s3), double>);
      results(10) = (f3 == 11);
      results(11) = (s3 == 12.0);
    }

    // Const rvalue
    {
      const auto pair = generate_pair();

      // Test structured binding
      auto [f, s] = std::move(pair);
      results(12) = (f == 11);
      results(13) = (s == 12.0);

      // test get

      // this is a somewhat convoluted way of getting a const pair &&
      const auto pair2 = generate_pair();
      auto f2          = Kokkos::get<0>(std::move(pair2));
      const auto pair3 = generate_pair();
      auto s2          = Kokkos::get<1>(std::move(pair3));
      static_assert(std::same_as<decltype(f2), int> &&
                    std::same_as<decltype(s2), double>);
      results(14) = (f2 == 11);
      results(15) = (s2 == 12.0);

      const auto pair4 = generate_pair();
      auto f3          = Kokkos::get<int>(pair4);
      const auto pair5 = generate_pair();
      auto s3          = Kokkos::get<double>(pair5);
      static_assert(std::same_as<decltype(f3), int> &&
                    std::same_as<decltype(s3), double>);
      results(16) = (f3 == 11);
      results(17) = (s3 == 12.0);
    }

    // Non-const rvalue
    {
      // Test structured binding
      auto [f, s] = generate_pair();
      results(18) = (f == 11);
      results(19) = (s == 12.0);

      // test get
      auto f2 = Kokkos::get<0>(generate_pair());
      auto s2 = Kokkos::get<1>(generate_pair());
      static_assert(std::same_as<decltype(f2), int> &&
                    std::same_as<decltype(s2), double>);
      results(20) = (f2 == 11);
      results(21) = (s2 == 12.0);

      auto f3 = Kokkos::get<int>(generate_pair());
      auto s3 = Kokkos::get<double>(generate_pair());
      static_assert(std::same_as<decltype(f3), int> &&
                    std::same_as<decltype(s3), double>);
      results(22) = (f3 == 11);
      results(23) = (s3 == 12.0);
    }
  }
};

TEST(defaultdevicetype, structured_bindings_and_tuple_interface) {
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_t     = Kokkos::View<int*, exec_space>;
  view_t results("pair_device_results", 24);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<exec_space>(0, 1),
      TupleInterfaceDeviceTest<view_t>{
          results});  // could CTAD here but some compilers have bugs around it

  Kokkos::fence();

  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
  for (int i = 0; i < 24; ++i) EXPECT_EQ(h(i), 1);
}
}  // namespace
