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
// View equivalence defined as matching value_type,,
// layout_type, memory_space, rank, span, data pointers and extents

namespace {

template <class T>
struct Another;

template <Kokkos::MemorySpace T>
struct Another<T> {
  using type =
      std::conditional_t<std::is_same_v<T, Kokkos::DefaultExecutionSpace>,
                         Kokkos::DefaultHostExecutionSpace,
                         Kokkos::DefaultExecutionSpace>;
};

template <Kokkos::ArrayLayoutConcept T>
struct Another<T> {
  using type = std::conditional_t<std::is_same_v<T, Kokkos::LayoutLeft>,
                                  Kokkos::LayoutRight, Kokkos::LayoutLeft>;
};

template <Kokkos::MemoryTraitsConcept T>
struct Another<T> {
  using type = std::conditional_t<std::is_same_v<T, Kokkos::MemoryRandomAccess>,
                                  typename Kokkos::MemoryUnmanaged,
                                  Kokkos::MemoryRandomAccess>;
};

template <class Left, class Right>
bool check_equal(Left l, Right r) {
  return l == r && !(l != r);
}

void test_view_equality_operator() {
  using T = double;
  using V = Kokkos::View<T>;

  using V_data_type_0 = Kokkos::View<V::data_type*>;
  using V_data_type_1 = Kokkos::View<V::value_type*>;
  using V_data_type_2 = Kokkos::View<V::value_type[1]>;
  using V_data_type_3 = Kokkos::View<const V::value_type>;
  using V_data_type_4 = Kokkos::View<const V::data_type>;

  // Check for static properties
  ASSERT_EQ(check_equal(V(), V_data_type_0()), false);
  ASSERT_EQ(check_equal(V(), V_data_type_1()), false);
  ASSERT_EQ(check_equal(V(), V_data_type_2()), false);
  // Note: We do not enforce const'ness of data_type or value_type
  ASSERT_EQ(check_equal(V(), V_data_type_3()), true);
  ASSERT_EQ(check_equal(V(), V_data_type_4()), true);

  using V_unmanged = Kokkos::View<V::data_type, Kokkos::MemoryUnmanaged>;
  using V_memory_traits =
      Kokkos::View<T, V::memory_space, Another<V::memory_traits>::type>;
  using V_layout_type       = Kokkos::View<T*, Another<V::array_layout>::type>;
  using V_memory_space_type = Kokkos::View<T, Another<V::memory_space>::type>;

  // Note: We do not enforce Traits::memory_traits equality
  ASSERT_EQ(check_equal(V(), V_memory_traits()), true);
  ASSERT_EQ(check_equal(V(), V_layout_type()), false);
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>)
    ASSERT_EQ(check_equal(V(), V_memory_space_type()), true);
  else
    ASSERT_EQ(check_equal(V(), V_memory_space_type()), false);

  // Check for pointer equality
  ASSERT_EQ(check_equal(V(), V()), true);  // nullptr is equal
  ASSERT_EQ(check_equal(V("V", 3), V("V", 3)), false);
  auto v_1D = V("v_1D", 3);
  ASSERT_EQ(check_equal(v_1D, V_unmanged(v_1D.data())), true);

  // Check for matching static extents (same span, same ptr)
  auto v_2D_0 = Kokkos::View<T[1][3]>("v_2D_0");
  auto v_2D_1 = Kokkos::View<T[3][1], Kokkos::MemoryUnmanaged>(v_2D_0.data());
  ASSERT_EQ(check_equal(v_2D_0, v_2D_1), false);

  // Check for matching dynamic extents (same span, same ptr)
  using v_2D_t = Kokkos::View<T**>;
  auto v_2D_2  = v_2D_t("v_2D_2", 3, 3);
  auto v_2D_3  = Kokkos::View<T[3][3], Kokkos::MemoryUnmanaged>(v_2D_2.data());
  ASSERT_EQ(check_equal(v_2D_2, v_2D_3), true);
}

TEST(TEST_CATEGORY, View_comparison_operator) { test_view_equality_operator(); }

}  // namespace