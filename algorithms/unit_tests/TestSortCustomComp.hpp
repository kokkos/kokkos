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

#ifndef KOKKOS_ALGORITHMS_SORTING_CUSTOM_COMP_HPP
#define KOKKOS_ALGORITHMS_SORTING_CUSTOM_COMP_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <TestStdAlgorithmsCommon.hpp>

namespace Test {
namespace SortWithComp {

template <class T>
struct MyComp {
  KOKKOS_INLINE_FUNCTION
  bool operator()(T a, T b) const { return a < b; }
};

template <class LayoutTagType, class ValueType>
auto create_random_view_and_host_clone(
    LayoutTagType LayoutTag, std::size_t n,
    Kokkos::pair<ValueType, ValueType> bounds, const std::string& label,
    std::size_t seedIn = 12371) {
  using namespace ::Test::stdalgos;

  // construct in memory space associated with default exespace
  auto dataView = create_view<ValueType>(LayoutTag, n, label);

  // dataView might not deep copyable (e.g. strided layout) so to
  // randomize it, we make a new view that is for sure deep copyable,
  // modify it on the host, deep copy to device and then launch
  // a kernel to copy to dataView
  auto dataView_dc =
      create_deep_copyable_compatible_view_with_same_extent(dataView);
  auto dataView_dc_h = create_mirror_view(Kokkos::HostSpace(), dataView_dc);

  // randomly fill the view
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(
      seedIn);
  Kokkos::fill_random(dataView_dc_h, pool, bounds.first, bounds.second);

  // copy to dataView_dc and then to dataView
  Kokkos::deep_copy(dataView_dc, dataView_dc_h);
  // use CTAD
  CopyFunctor F1(dataView_dc, dataView);
  Kokkos::parallel_for("copy", dataView.extent(0), F1);

  return std::make_pair(dataView, dataView_dc_h);
}

template <class Tag, class ValueType>
void run_all_scenarios(int api) {
  for (const auto& scenario : stdalgos::default_scenarios) {
    auto [dataView, dataViewBeforeOp_h] = create_random_view_and_host_clone(
        Tag{}, scenario.second, Kokkos::pair<ValueType, ValueType>{-1045, 565},
        "dataView");

    namespace KE = Kokkos::Experimental;
    if (api == 0) {
      Kokkos::sort(dataView);
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h));
    } else if (api == 1) {
      Kokkos::sort(Kokkos::DefaultExecutionSpace{}, dataView);
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h));
    } else if (api == 2) {
      using comp_t = MyComp<ValueType>;
      Kokkos::sort(dataView, comp_t{});
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h),
                comp_t{});
    } else if (api == 3) {
      using comp_t = MyComp<ValueType>;
      Kokkos::sort(Kokkos::DefaultExecutionSpace{}, dataView, comp_t{});
      std::sort(KE::begin(dataViewBeforeOp_h), KE::end(dataViewBeforeOp_h),
                comp_t{});
    }

    auto dataView_h = stdalgos::create_host_space_copy(dataView);
    stdalgos::compare_views(dataViewBeforeOp_h, dataView_h);
  }
}

TEST(TEST_CATEGORY, SortWithCustomComparator) {
  for (int api = 0; api < 4; api++) {
    run_all_scenarios<stdalgos::DynamicTag, int>(api);
    run_all_scenarios<stdalgos::DynamicTag, double>(api);
    run_all_scenarios<stdalgos::StridedTwoTag, int>(api);
    run_all_scenarios<stdalgos::StridedThreeTag, int>(api);
    run_all_scenarios<stdalgos::StridedTwoTag, double>(api);
    run_all_scenarios<stdalgos::StridedThreeTag, double>(api);
  }
}

}  // namespace SortWithComp
}  // namespace Test
#endif
