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

template <class MemoryTraits>
struct CheckAccessor {
  using acc_t = typename Kokkos::Impl::ImplAccessor<int, Kokkos::HostSpace,
                                                    MemoryTraits>::type;
  static_assert(
      std::is_same_v<MemoryTraits, decltype(acc_t::impl_memory_traits())>);
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
  {
    CheckAccessor<Kokkos::MemoryTraits<>> a0;
    (void)a0;
    CheckAccessor<Kokkos::MemoryTraits<Kokkos::Unmanaged>> a1;
    (void)a1;
    CheckAccessor<Kokkos::MemoryTraits<Kokkos::Atomic>> a2;
    (void)a2;
    CheckAccessor<Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>> a3;
    (void)a3;
  }
  {
    using new_view_t =
        Kokkos::View<float, Kokkos::dextents<unsigned, 7>, Kokkos::layout_right,
                     Kokkos::Experimental::Accessor<float>>;
    using old_view_t = Kokkos::View<float*******, Kokkos::LayoutRight>;
    static_assert(std::is_same_v<typename new_view_t::index_type, unsigned>);
    static_assert(std::is_same_v<typename old_view_t::index_type, size_t>);
    static_assert(sizeof(new_view_t) == 4 * 7 + 4 + 16);
    static_assert(sizeof(old_view_t) == 8 * 7 + 8 + 16);
  }
  delete[] data;
}

}  // namespace Test
