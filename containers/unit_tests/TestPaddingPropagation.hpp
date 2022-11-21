/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_DualView.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Kokkos_Core.hpp>

namespace Test {

using DType   = int;
using DType_0 = DType;
using DType_1 = DType *;
using DType_2 = DType **;
using DType_3 = DType ***;
using DType_4 = DType ****;
using DType_5 = DType *****;
using DType_6 = DType ******;
using DType_7 = DType *******;
using DType_8 = DType ********;

using pod = DType;

template <typename View1, typename View2>
bool test_view_strides(View1 &view1, View2 &view2, int rank) {
  return (view1.stride(0) == view2.stride(0) || rank < 1) &&
         (view1.stride(1) == view2.stride(1) || rank < 2) &&
         (view1.stride(2) == view2.stride(2) || rank < 3) &&
         (view1.stride(3) == view2.stride(3) || rank < 4) &&
         (view1.stride(4) == view2.stride(4) || rank < 5) &&
         (view1.stride(5) == view2.stride(5) || rank < 6) &&
         (view1.stride(6) == view2.stride(6) || rank < 7) &&
         (view1.stride(7) == view2.stride(7) || rank < 8);
}

template <typename View, typename DRV>
bool test_view_offset_strides(View &view, DRV &drv, int rank) {
  return (view.stride(0) == drv.stride_0() || rank < 1) &&
         (view.stride(1) == drv.stride_1() || rank < 2) &&
         (view.stride(2) == drv.stride_2() || rank < 3) &&
         (view.stride(3) == drv.stride_3() || rank < 4) &&
         (view.stride(4) == drv.stride_4() || rank < 5) &&
         (view.stride(5) == drv.stride_5() || rank < 6) &&
         (view.stride(6) == drv.stride_6() || rank < 7) &&
         (view.stride(7) == drv.stride_7() || rank < 8);
}

template <typename view_t, typename dt, typename layout, typename... Vals>
bool padding_assignment_and_copy(Vals... params) {
  constexpr int rank = sizeof...(params);
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  if constexpr (std::is_same_v<view_t, Kokkos::DualView<dt, layout>>) {
    view_t dv(alloc_prop, params...);
    view_t dv_2(dv.d_view, dv.h_view);
    bool result = test_view_strides(dv.d_view, dv.h_view, rank);
    result &= test_view_strides(dv_2.d_view, dv_2.h_view, rank);
    return result;
  } else {
    static_assert(std::is_same_v<view_t, Kokkos::DynRankView<pod, layout>>);
    Kokkos::View<dt, layout> v(alloc_prop, params...);
    view_t drv(v);
    // We do not propagate padding to DynRankView and we only
    // pad up to rank 2, so strides should only match in that case
    if ((std::is_same_v<dt, DType_0> || std::is_same_v<dt, DType_1>) ||
        (std::is_same_v<layout, Kokkos::LayoutRight> &&
         std::is_same_v<dt, DType_2>))
      return test_view_offset_strides(v, drv, rank);
    else
      return !test_view_offset_strides(v, drv, rank);
  }
}

template <typename... Args>
using DV = Kokkos::DualView<Args...>;

template <typename... Args>
using DRV = Kokkos::DynRankView<Args...>;

template <typename layout>
bool padding_assignment_and_copy_helper() {
  auto result = true;
  // clang-format off
  result &= padding_assignment_and_copy<DV<DType_0, layout>, DType_0, layout>();
  result &= padding_assignment_and_copy<DV<DType_1, layout>, DType_1, layout>(50);
  result &= padding_assignment_and_copy<DV<DType_2, layout>, DType_2, layout>(50, 1);
  result &= padding_assignment_and_copy<DV<DType_3, layout>, DType_3, layout>(50, 1, 50);
  result &= padding_assignment_and_copy<DV<DType_4, layout>, DType_4, layout>(50, 1, 1, 50);
  result &= padding_assignment_and_copy<DV<DType_5, layout>, DType_5, layout>(50, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DV<DType_6, layout>, DType_6, layout>(50, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DV<DType_7, layout>, DType_7, layout>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DV<DType_8, layout>, DType_8, layout>(50, 1, 1, 1, 1, 1, 1, 50);

  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_0, layout>();
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_1, layout>(50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_2, layout>(50, 1);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_3, layout>(50, 1, 50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_4, layout>(50, 1, 1, 50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_5, layout>(50, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_6, layout>(50, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_7, layout>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<DRV<pod, layout>, DType_8, layout>(50, 1, 1, 1, 1, 1, 1, 50);
  // clang-format on
  return result;
}

TEST(TEST_CATEGORY, container_padding_propagation) {
  /*Test correct propagation in view assignements*/
  ASSERT_TRUE(padding_assignment_and_copy_helper<Kokkos::LayoutLeft>());
  ASSERT_TRUE(padding_assignment_and_copy_helper<Kokkos::LayoutRight>());
}

}  // namespace Test
