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

#define DV Kokkos::DualView
#define DRV Kokkos::DynRankView

#define RANK_0 0
#define RANK_1 1
#define RANK_2 2
#define RANK_3 3
#define RANK_4 4
#define RANK_5 5
#define RANK_6 6
#define RANK_7 7

#define PARAM_0
#define PARAM_1 50
#define PARAM_2 50, 1
#define PARAM_3 50, 1, 50
#define PARAM_4 50, 1, 1, 50
#define PARAM_5 50, 1, 1, 1, 50
#define PARAM_6 50, 1, 1, 1, 1, 50
#define PARAM_7 50, 1, 1, 1, 1, 1, 50

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

using pod = DType;

#define ASSERT_VIEW_STRIDES(view1, view2, rank)           \
  (view1.stride(0) == view2.stride(0) || rank < 1) &&     \
      (view1.stride(1) == view2.stride(1) || rank < 2) && \
      (view1.stride(2) == view2.stride(2) || rank < 3) && \
      (view1.stride(3) == view2.stride(3) || rank < 4) && \
      (view1.stride(4) == view2.stride(4) || rank < 5) && \
      (view1.stride(5) == view2.stride(5) || rank < 6) && \
      (view1.stride(6) == view2.stride(6) || rank < 7) && \
      (view1.stride(7) == view2.stride(7) || rank < 8)

#define ASSERT_VIEW_OFFSET_STRIDES(view, drv, rank)     \
  (view.stride(0) == drv.stride_0() || rank < 1) &&     \
      (view.stride(1) == drv.stride_1() || rank < 2) && \
      (view.stride(2) == drv.stride_2() || rank < 3) && \
      (view.stride(3) == drv.stride_3() || rank < 4) && \
      (view.stride(4) == drv.stride_4() || rank < 5) && \
      (view.stride(5) == drv.stride_5() || rank < 6) && \
      (view.stride(6) == drv.stride_6() || rank < 7) && \
      (view.stride(7) == drv.stride_7() || rank < 8)

template <typename view_t, typename dt, typename l, typename... Vals>
typename std::enable_if_t<std::is_same<view_t, DV<dt, l>>::value, bool>
padding_assignment_and_copy(int rank, Vals... params) {
  bool result = true;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t dv(alloc_prop, params...);
  view_t dv_2(dv.d_view, dv.h_view);
  result &= ASSERT_VIEW_STRIDES(dv.d_view, dv.h_view, rank);
  result &= ASSERT_VIEW_STRIDES(dv_2.d_view, dv_2.h_view, rank);
  return result;
}

template <typename view_t, typename dt, typename l, typename... Vals>
typename std::enable_if_t<std::is_same<view_t, DRV<pod, l>>::value &&
                              ((std::is_same<dt, DType_0>::value ||
                                std::is_same<dt, DType_1>::value) ||
                               (std::is_same<l, Kokkos::LayoutRight>::value &&
                                std::is_same<dt, DType_2>::value)),
                          bool>
padding_assignment_and_copy(int rank, Vals... params) {
  bool result = true;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  Kokkos::View<dt, l> v(alloc_prop, params...);
  view_t drv(v);
  // We do not propagate padding to DynRankView and we do not
  // pad for this data type so strides should match
  result &= ASSERT_VIEW_OFFSET_STRIDES(v, drv, rank);
  return result;
}

template <typename view_t, typename dt, typename l, typename... Vals>
typename std::enable_if_t<std::is_same<view_t, DRV<pod, l>>::value &&
                              (!(std::is_same<dt, DType_0>::value ||
                                 std::is_same<dt, DType_1>::value) &&
                               !(std::is_same<l, Kokkos::LayoutRight>::value &&
                                 std::is_same<dt, DType_2>::value)),
                          bool>
padding_assignment_and_copy(int rank, Vals... params) {
  bool result = true;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  Kokkos::View<dt, l> v(alloc_prop, params...);
  view_t drv(v);
  // We do not propagate padding to DynRankView
  // so strides should not match
  result &= !(ASSERT_VIEW_OFFSET_STRIDES(v, drv, rank));
  return result;
}

TEST(TEST_CATEGORY, container_padding_propagation) {
  /*Test correct propagation in view assignements*/
  auto result = true;
  using ll    = Kokkos::LayoutLeft;
  using lr    = Kokkos::LayoutRight;
  result &= padding_assignment_and_copy<DV<DType_0, ll>, DType_0, ll>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignment_and_copy<DV<DType_1, ll>, DType_1, ll>(RANK_1,
                                                                      PARAM_1);
  result &= padding_assignment_and_copy<DV<DType_2, ll>, DType_2, ll>(RANK_2,
                                                                      PARAM_2);
  result &= padding_assignment_and_copy<DV<DType_3, ll>, DType_3, ll>(RANK_3,
                                                                      PARAM_3);
  result &= padding_assignment_and_copy<DV<DType_4, ll>, DType_4, ll>(RANK_4,
                                                                      PARAM_4);
  result &= padding_assignment_and_copy<DV<DType_5, ll>, DType_5, ll>(RANK_5,
                                                                      PARAM_5);
  result &= padding_assignment_and_copy<DV<DType_6, ll>, DType_6, ll>(RANK_6,
                                                                      PARAM_6);
  result &= padding_assignment_and_copy<DV<DType_7, ll>, DType_7, ll>(RANK_7,
                                                                      PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_assignment_and_copy<DV<DType_0, lr>, DType_0, lr>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignment_and_copy<DV<DType_1, lr>, DType_1, lr>(RANK_1,
                                                                      PARAM_1);
  result &= padding_assignment_and_copy<DV<DType_2, lr>, DType_2, lr>(RANK_2,
                                                                      PARAM_2);
  result &= padding_assignment_and_copy<DV<DType_3, lr>, DType_3, lr>(RANK_3,
                                                                      PARAM_3);
  result &= padding_assignment_and_copy<DV<DType_4, lr>, DType_4, lr>(RANK_4,
                                                                      PARAM_4);
  result &= padding_assignment_and_copy<DV<DType_5, lr>, DType_5, lr>(RANK_5,
                                                                      PARAM_5);
  result &= padding_assignment_and_copy<DV<DType_6, lr>, DType_6, lr>(RANK_6,
                                                                      PARAM_6);
  result &= padding_assignment_and_copy<DV<DType_7, lr>, DType_7, lr>(RANK_7,
                                                                      PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_assignment_and_copy<DRV<pod, ll>, DType_0, ll>(
      RANK_0 /*, PARAM_0*/);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_1, ll>(RANK_1, PARAM_1);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_2, ll>(RANK_2, PARAM_2);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_3, ll>(RANK_3, PARAM_3);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_4, ll>(RANK_4, PARAM_4);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_5, ll>(RANK_5, PARAM_5);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_6, ll>(RANK_6, PARAM_6);
  result &=
      padding_assignment_and_copy<DRV<pod, ll>, DType_7, ll>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_assignment_and_copy<DRV<pod, lr>, DType_0, lr>(
      RANK_0 /*, PARAM_0*/);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_1, lr>(RANK_1, PARAM_1);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_2, lr>(RANK_2, PARAM_2);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_3, lr>(RANK_3, PARAM_3);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_4, lr>(RANK_4, PARAM_4);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_5, lr>(RANK_5, PARAM_5);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_6, lr>(RANK_6, PARAM_6);
  result &=
      padding_assignment_and_copy<DRV<pod, lr>, DType_7, lr>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  // Padding is not allowed for LayoutStride
  // using ls = Kokkos::LayoutStride;
}

#undef RANK_0
#undef RANK_1
#undef RANK_2
#undef RANK_3
#undef RANK_4
#undef RANK_5
#undef RANK_6
#undef RANK_7

#undef PARAM_0
#undef PARAM_1
#undef PARAM_2
#undef PARAM_3
#undef PARAM_4
#undef PARAM_5
#undef PARAM_6
#undef PARAM_7

#undef ASSERT_VIEW_STRIDES
#undef ASSERT_VIEW_OFFSET_STRIDES

#undef DV
#undef DRV

}  // namespace Test
