#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif
#include <cstddef>

// Test View equality operators
// View equivalence defined as matching value_type, traits,
// layout_type, memory_space, rank, span, data pointers and extents

namespace {

template <class Left, class Right>
bool check_equivalent(Left l, Right r) {
  return std::is_same_v<Left, Right> && l == r && !(l != r);
}

void test_view_equality_operator() {
  using T = double;
  // TBD: Add more test permutations
  using a = Kokkos::View<T, Kokkos::Serial>;
  using b = Kokkos::View<T* [1], Kokkos::Serial>;
  using c =
      Kokkos::View<T, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  ASSERT_EQ(check_equivalent(a(), a()), true);
  ASSERT_EQ(check_equivalent(a(), b()), false);
  ASSERT_EQ(check_equivalent(a(), c()), false);
  // TBD: Add more test permutations
}

TEST(TEST_CATEGORY, View_comparison_operator) { test_view_equality_operator(); }

}  // namespace