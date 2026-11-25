// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <cstddef>

template <class ExtentsType, class... Args>
void test_constructor_with_dynamic_extents(const ExtentsType& extents,
                                           const Args&... args) {
  using extents_type  = ExtentsType;
  using layout_type   = Kokkos::layout_left;
  using accessor_type = Kokkos::Experimental::Accessor<
      float, typename TEST_EXECSPACE::memory_space, Kokkos::MemoryTraits<>>;
  using view_type =
      Kokkos::View<float, extents_type, layout_type, accessor_type>;

  view_type view("test_view", args...);

  for (int r = 0; r < static_cast<int>(view_type::rank()); r++) {
    EXPECT_EQ(view.extent(r), extents.extent(r));
  }
}

template <class T>
void test_constructor() {
  test_constructor_with_dynamic_extents(Kokkos::extents<T, 2, 3, 4>());
  test_constructor_with_dynamic_extents(
      Kokkos::extents<T, Kokkos::dynamic_extent, 3, 4>(2), 2);
  test_constructor_with_dynamic_extents(
      Kokkos::extents<T, Kokkos::dynamic_extent, 3, 4>(2), 2, 3, 4);
  test_constructor_with_dynamic_extents(
      Kokkos::extents<T, Kokkos::dynamic_extent, Kokkos::dynamic_extent, 4>(2,
                                                                            3),
      2, 3);
  test_constructor_with_dynamic_extents(
      Kokkos::extents<T, Kokkos::dynamic_extent, Kokkos::dynamic_extent, 4>(2,
                                                                            3),
      2, 3, 4);
  test_constructor_with_dynamic_extents(
      Kokkos::extents<T, Kokkos::dynamic_extent, Kokkos::dynamic_extent,
                      Kokkos::dynamic_extent>(2, 3, 4),
      2, 3, 4);
}

TEST(TEST_CATEGORY, view_mdspan_args_constructor) {
  test_constructor<int>();
  test_constructor<std::size_t>();
}
