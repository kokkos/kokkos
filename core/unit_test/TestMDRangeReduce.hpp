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

template <typename T>
void MDRangeReduceTester([[maybe_unused]] int bound, int k) {
  const auto policy_MD = Kokkos::MDRangePolicy<Kokkos::Rank<2>, TEST_EXECSPACE>(
      {0, 0}, {bound, 2});

  // No explicit fence() calls needed because result is in HostSpace
  {
    T lor_MD = 0;
    Kokkos::parallel_reduce(
        policy_MD,
        KOKKOS_LAMBDA(const int i, const int, T& res) { res = res || i == k; },
        Kokkos::LOr<T>(lor_MD));
    EXPECT_EQ(lor_MD, 1);
  }
  {
    // Stick just a few true values in the Logical-OR reduction space,
    // to try to make sure every value is being captured
    T land_MD = 0;
    Kokkos::parallel_reduce(
        policy_MD, KOKKOS_LAMBDA(const int, const int, T& res) { res = 1; },
        Kokkos::LAnd<T>(land_MD));
    EXPECT_EQ(land_MD, 1);
  }
}

TEST(TEST_CATEGORY, mdrange_parallel_reduce_primitive_types) {
  auto doit = [](int bound, int k) {
    MDRangeReduceTester<bool>(bound, k);
    MDRangeReduceTester<signed char>(bound, k);
    MDRangeReduceTester<int8_t>(bound, k);
    MDRangeReduceTester<int16_t>(bound, k);
    MDRangeReduceTester<int32_t>(bound, k);
    MDRangeReduceTester<int64_t>(bound, k);
  };

// FIXME_NEXTSILICON: full sequence very slow in handoff 1.3.0-120
#ifdef KOKKOS_ENABLE_NEXTSILICON
  if constexpr (std::is_same_v<TEST_EXECSPACE,
                               Kokkos::Experimental::NextSilicon>) {
    for (int bound : {0, 7000}) {
      for (int k : {0, 3500, 6999}) {
        if (k < bound) doit(bound, k);
      }
    }
    SUCCEED() << "ran shorter NextSilicon test";
    return;
  }
#endif
  for (int bound : {0, 1, 7, 32, 65, 7000}) {
    for (int k = 0; k < bound; ++k) {
      doit(bound, k);
    }
  }
}

template <typename T, std::size_t N>
void test_array_reduction_identity() {
  using array_type = Kokkos::Array<T, N>;

  constexpr array_type array_min =
      Kokkos::reduction_identity<array_type>::min();
  constexpr array_type array_max =
      Kokkos::reduction_identity<array_type>::max();

  for (std::size_t i = 0; i < N; ++i) {
    ASSERT_EQ(array_min[i], Kokkos::reduction_identity<T>::min());
    ASSERT_EQ(array_max[i], Kokkos::reduction_identity<T>::max());
  }
}

template <typename T>
void test_array_reduction_identity_sizes() {
  test_array_reduction_identity<T, 1>();
  test_array_reduction_identity<T, 2>();
  test_array_reduction_identity<T, 3>();
}

TEST(TEST_CATEGORY, array_reduction_identity) {
  test_array_reduction_identity_sizes<int32_t>();
  test_array_reduction_identity_sizes<int64_t>();
  test_array_reduction_identity_sizes<uint32_t>();
  test_array_reduction_identity_sizes<uint64_t>();
}

}  // namespace
