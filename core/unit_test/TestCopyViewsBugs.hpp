// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef TEST_COPY_VIEWS_BUGS_HPP_
#define TEST_COPY_VIEWS_BUGS_HPP_
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace TestCopyViewsBugs {

template <class DeviceType>
void test_resize_exec_space_with_layout() {
  using exec_space = typename DeviceType::execution_space;
  using view_type  = Kokkos::View<int**, Kokkos::LayoutLeft, DeviceType>;
  view_type v("test_view", 4, 5);
  Kokkos::deep_copy(v, 99);
  typename view_type::array_layout new_layout(6, 7);

  Kokkos::resize(exec_space{}, v, new_layout);

  EXPECT_EQ(v.extent(0), 6u);
  EXPECT_EQ(v.extent(1), 7u);

  // Verify old data preserved in overlapping region
  auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);

  bool data_preserved = true;

  for (int i0 = 0; i0 < 4; ++i0) {
    for (int i1 = 0; i1 < 5; ++i1) {
      if (h_v(i0, i1) != 99) {
        data_preserved = false;

        break;
      }
    }
  }

  EXPECT_TRUE(data_preserved);
}

template <class DeviceType>
void test_resize_with_view_alloc_and_layout() {
  using exec_space = typename DeviceType::execution_space;
  using view_type  = Kokkos::View<int**, Kokkos::LayoutLeft, DeviceType>;
  view_type v("test_view", 4, 5);

  Kokkos::deep_copy(v, 99);
  typename view_type::array_layout new_layout(6, 7);

  // This calls ok.
  Kokkos::resize(Kokkos::view_alloc(exec_space{}), v, new_layout);

  EXPECT_EQ(v.extent(0), 6u);
  EXPECT_EQ(v.extent(1), 7u);

  auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
  bool data_preserved = true;

  for (int i0 = 0; i0 < 4; ++i0) {
    for (int i1 = 0; i1 < 5; ++i1) {
      if (h_v(i0, i1) != 99) {
        data_preserved = false;

        break;
      }
    }
  }

  EXPECT_TRUE(data_preserved);
}

template <class DeviceType>
void testCopyViewsBugs() {
  test_resize_exec_space_with_layout<DeviceType>();
  test_resize_with_view_alloc_and_layout<DeviceType>();
}
}  // namespace TestCopyViewsBugs

#endif  // TEST_COPY_VIEWS_BUGS_HPP_
