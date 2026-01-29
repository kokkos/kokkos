// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

template <class D, class... Properties>
struct Compare {
  using old_view_t = Kokkos::View<D, Properties...>;
  using mdspan_t   = typename old_view_t::mdspan_type;
  using new_view_t = Kokkos::View<
      typename mdspan_t::element_type, typename mdspan_t::extents_type,
      typename mdspan_t::layout_type, typename mdspan_t::accessor_type>;
  static_assert(std::is_same_v<typename new_view_t::mdspan_type,
                               typename old_view_t::mdspan_type>);
  template <class... Args>
  Compare(Args... args) {
    new_view_t a(args...);
  }
};

TEST(defaultdevicetype, development_test) {
  int* data = new int[12];
  {
    Compare<int**> c0("A", 1, 2);
    Compare<const int*, Kokkos::LayoutLeft> c1(data, 12);
    Compare<const int* [3][4], Kokkos::HostSpace> c2(data, 1, 3, 4);
    Compare<int*** [1], Kokkos::LayoutLeft, Kokkos::HostSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        c3(data, 1, 2, 3);
    Compare<const int******, Kokkos::MemoryTraits<Kokkos::Unmanaged>> c4(
        data, 1, 2, 2, 3, 1, 1);
  }
  delete[] data;
}

}  // namespace Test
