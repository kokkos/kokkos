// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {
template <class Slice>
KOKKOS_FUNCTION auto replace_full_extent_with_ALL(Slice slice) {
  return slice;
}

KOKKOS_INLINE_FUNCTION
auto replace_full_extent_with_ALL(Kokkos::full_extent_t) { return Kokkos::ALL; }

static_assert(
    std::is_same_v<decltype(replace_full_extent_with_ALL(Kokkos::full_extent)),
                   Kokkos::ALL_t>);

// Can't use a pack for Slices, because NVCC doesn't allow capturing pack
// in a host/device lambda
template <class ViewType, class Slice0, class Slice1, class Slice2>
void test_ALL_full_extent_equivalency(ViewType view, Slice0 slice0,
                                      Slice1 slice1, Slice2 slice2) {
  // TODO: use Kokkos::single when available
  Kokkos::RangePolicy<TEST_EXECSPACE> policy(0, 1);

  int errors = 0;
  Kokkos::parallel_reduce(
      "subview_ALL_full_extent", policy,
      KOKKOS_LAMBDA(int, int& err) {
        auto sub_full_extent = Kokkos::subview(view, slice0, slice1, slice2);
        auto sub_ALL =
            Kokkos::subview(view, replace_full_extent_with_ALL(slice0),
                            replace_full_extent_with_ALL(slice1),
                            replace_full_extent_with_ALL(slice2));

        static_assert(
            std::is_same_v<decltype(sub_full_extent), decltype(sub_ALL)>);
        if (sub_full_extent != sub_ALL) err++;
      },
      errors);
  ASSERT_EQ(errors, 0);
}
}  // namespace

TEST(TEST_CATEGORY, subview_ALL_full_extent_equivalence) {
  // Not testing the same slice args for each view type is intentional
  {
    Kokkos::View<float***, TEST_EXECSPACE> rank3dynamic3("A", 10, 17, 21);
    test_ALL_full_extent_equivalency(rank3dynamic3, Kokkos::full_extent,
                                     Kokkos::full_extent, 3);
    test_ALL_full_extent_equivalency(rank3dynamic3, 3, Kokkos::full_extent,
                                     Kokkos::pair{3, 7});
  }
  {
    Kokkos::View<float* [17][21], TEST_EXECSPACE> rank3dynamic1("B", 10, 17,
                                                                21);
    test_ALL_full_extent_equivalency(rank3dynamic1, Kokkos::pair{3, 7},
                                     Kokkos::full_extent, 3);
    test_ALL_full_extent_equivalency(rank3dynamic1, 3, Kokkos::full_extent,
                                     Kokkos::full_extent);
  }
  {
    Kokkos::View<float, Kokkos::extents<int, Kokkos::dynamic_extent, 17, 21>,
                 Kokkos::layout_left,
                 Kokkos::Experimental::Accessor<
                     float, typename TEST_EXECSPACE::memory_space,
                     Kokkos::MemoryTraits<>>>
        rank3dynamic1("C", 10, 17, 21);
    test_ALL_full_extent_equivalency(rank3dynamic1, Kokkos::pair{3, 7},
                                     Kokkos::full_extent, 3);
    test_ALL_full_extent_equivalency(rank3dynamic1, 3, Kokkos::full_extent,
                                     Kokkos::full_extent);
  }
}
