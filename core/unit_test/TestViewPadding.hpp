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

#include <sstream>
#include <iostream>
#include <time.h>
#include <string>

#include <Kokkos_Core.hpp>

namespace Test {

#define RANK_0 0
#define RANK_1 1
#define RANK_2 2
#define RANK_3 3
#define RANK_4 4
#define RANK_5 5
#define RANK_6 6
#define RANK_7 7

#define ZERO \
  0  // Use Zero to make ViewOffset's
     // Dimension::rank_dynamic > 0
#define ARG_0
#define ARG_1 ZERO
#define ARG_2 50, ZERO
#define ARG_3 50, ZERO, 50
#define ARG_4 50, 1, ZERO, 50
#define ARG_5 50, 1, ZERO, 1, 50
#define ARG_6 50, 1, ZERO, 1, 1, 50
#define ARG_7 50, 1, ZERO, 1, 1, 1, 50

// Use non-zero sizes for Views
#define PARAM_0
#define PARAM_1 50
#define PARAM_2 50, 1
#define PARAM_3 50, 1, 50
#define PARAM_4 50, 1, 1, 50
#define PARAM_5 50, 1, 1, 1, 50
#define PARAM_6 50, 1, 1, 1, 1, 50
#define PARAM_7 50, 1, 1, 1, 1, 1, 50

#define SUBRANGE_DIM 20, 30

#define VIEW_T Kokkos::View

using DType   = int;
using DType_0 = DType;
using DType_1 = DType *;
using DType_2 = DType **;
using DType_3 = DType ***;
using DType_4 = DType ****;
using DType_5 = DType *****;
using DType_6 = DType ******;
using DType_7 = DType *******;

using pair_t = Kokkos::pair<int, int>;

#define ASSERT_VIEW_OFFSET_STRIDES(view, offset, rank)     \
  (view.stride(0) == offset.stride_0() || rank < 1) &&     \
      (view.stride(1) == offset.stride_1() || rank < 2) && \
      (view.stride(2) == offset.stride_2() || rank < 3) && \
      (view.stride(3) == offset.stride_3() || rank < 4) && \
      (view.stride(4) == offset.stride_4() || rank < 5) && \
      (view.stride(5) == offset.stride_5() || rank < 6) && \
      (view.stride(6) == offset.stride_6() || rank < 7) && \
      (view.stride(7) == offset.stride_7() || rank < 8)

#define ASSERT_VIEW_STRIDES(view1, view2, rank)           \
  (view1.stride(0) == view2.stride(0) || rank < 1) &&     \
      (view1.stride(1) == view2.stride(1) || rank < 2) && \
      (view1.stride(2) == view2.stride(2) || rank < 3) && \
      (view1.stride(3) == view2.stride(3) || rank < 4) && \
      (view1.stride(4) == view2.stride(4) || rank < 5) && \
      (view1.stride(5) == view2.stride(5) || rank < 6) && \
      (view1.stride(6) == view2.stride(6) || rank < 7) && \
      (view1.stride(7) == view2.stride(7) || rank < 8)

template <typename viewDim, typename layout, typename... Vals>
typename std::
    enable_if_t</*
std::is_same<layout, Kokkos::LayoutRight>::value ||*/
                std::is_same<viewDim,
                             Kokkos::Impl::ViewDimension<ARG_0>>::value ||
                    std::is_same<viewDim,
                                 Kokkos::Impl::ViewDimension<ARG_1>>::value ||
                    (std::is_same<layout, Kokkos::LayoutRight>::value &&
                     std::is_same<viewDim,
                                  Kokkos::Impl::ViewDimension<ARG_2>>::value),
                bool>
    padding_has_effect(Vals... params) {
  using allow_padding = std::true_type;

  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = std::bool_constant<
      allow_padding() &&
      !std::is_same<layout, typename Kokkos::LayoutStride>::value>;

  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  using no_padding =
      std::integral_constant<unsigned int,
                             !use_padding::value ? sizeof(DType) : 0>;

  auto offset_wPadding = offset_t(padding(), layout(params...));
  auto offset          = offset_t(no_padding(), layout(params...));

  /* We never pad for rank 0 and rank 1 and for LR && dyn_rank > 1 */
  return offset.span_is_contiguous() == offset_wPadding.span_is_contiguous();
}

template <typename viewDim, typename layout, typename... Vals>
typename std::enable_if_t<
    !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value &&
        !(std::is_same<layout, Kokkos::LayoutRight>::value &&
          std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_2>>::value),
    bool>
padding_has_effect(Vals... params) {
  using allow_padding = std::true_type;

  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = std::bool_constant<
      allow_padding() &&
      !std::is_same<layout, typename Kokkos::LayoutStride>::value>;

  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  using no_padding =
      std::integral_constant<unsigned int,
                             !use_padding::value ? sizeof(DType) : 0>;

  auto offset_wPadding = offset_t(padding(), layout(params...));
  auto offset          = offset_t(no_padding(), layout(params...));

  /* if we pad, the span becomes non-contiguous*/
  return offset.span_is_contiguous() != offset_wPadding.span_is_contiguous();
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<std::is_same<allow_padding, std::true_type>::value,
                          bool>
padding_construction(int rank, Vals... params) {
  using view_t      = Kokkos::View<dim_t, layout>;
  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = std::bool_constant<
      allow_padding() &&
      !std::is_same<layout, typename Kokkos::LayoutStride>::value>;
  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  offset_t offset = offset_t(padding(), layout(params...));
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);

  view_t v(alloc_prop, params...);
  return ASSERT_VIEW_OFFSET_STRIDES(v, offset, rank) ? true : false;
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<std::is_same<allow_padding, std::false_type>::value,
                          bool>
padding_construction(int rank, Vals... params) {
  using view_t      = Kokkos::View<dim_t, layout>;
  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = allow_padding;
  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  offset_t offset = offset_t(padding(), layout(params...));
  auto alloc_prop = Kokkos::view_alloc("vDim" + std::to_string(rank));
  view_t v(alloc_prop, params...);
  return ASSERT_VIEW_OFFSET_STRIDES(v, offset, rank);
}

template <typename view_t, typename... Vals>
bool padding_assignment_and_copy(int rank, Vals... params) {
  bool result = true;

  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  deep_copy(v, 1);
  auto v_mirror = Kokkos::create_mirror_view(v);
  result &= ASSERT_VIEW_STRIDES(v_mirror, v, rank);

  /*deep_copy should support strided views if stride from padding*/
  Kokkos::deep_copy(v_mirror, v);
  return result;
}
template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value, void>
        *ptr = nullptr) {
  return Kokkos::subview(view, 0);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_2>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_3>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL, Kokkos::ALL);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_4>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_5>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_6>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL);
}

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<dim_t, layout> &view,
    [[maybe_unused]] typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_7>>::value, void>
        *ptr = nullptr) {
  auto pair = pair_t(SUBRANGE_DIM);
  return Kokkos::subview(view, pair, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<
    std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value ||
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value ||
        (std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_2>>::value &&
         std::is_same<layout, Kokkos::LayoutRight>::value),
    bool>
padding_subview_and_copy(int rank, Vals... params) {
  bool result  = true;
  using view_t = Kokkos::View<dim_t, layout>;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<dim_t, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);
  result &= ASSERT_VIEW_STRIDES(v, v_sub, rank);
  result &= ASSERT_VIEW_STRIDES(v_sub, v_sub_mirror, rank);
  return result;
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<
    !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value &&
        std::is_same<layout, Kokkos::LayoutLeft>::value,
    bool>
padding_subview_and_copy(int rank, Vals... params) {
  bool result  = true;
  using view_t = Kokkos::View<dim_t, layout>;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<dim_t, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);
  result &= ASSERT_VIEW_STRIDES(v, v_sub, rank);
  result &= !(ASSERT_VIEW_STRIDES(v_sub, v_sub_mirror, rank));
  return result;
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<
    !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_2>>::value &&
        std::is_same<layout, Kokkos::LayoutRight>::value &&
        std::is_same<allow_padding, std::true_type>::value,
    bool>
padding_subview_and_copy(int rank, Vals... params) {
  bool result  = true;
  using view_t = Kokkos::View<dim_t, layout>;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<dim_t, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);
  result &= ASSERT_VIEW_STRIDES(v, v_sub, rank);
  result &= !(ASSERT_VIEW_STRIDES(v_sub, v_sub_mirror, rank));
  return result;
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
typename std::enable_if_t<
    !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_0>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_1>>::value &&
        !std::is_same<viewDim, Kokkos::Impl::ViewDimension<ARG_2>>::value &&
        std::is_same<layout, Kokkos::LayoutRight>::value &&
        std::is_same<allow_padding, std::false_type>::value,
    bool>
padding_subview_and_copy(int rank, Vals... params) {
  bool result     = true;
  using view_t    = Kokkos::View<dim_t, layout>;
  auto alloc_prop = Kokkos::view_alloc("vDim" + std::to_string(rank));
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<dim_t, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);
  result &= ASSERT_VIEW_STRIDES(v, v_sub, rank);
  result &= ASSERT_VIEW_STRIDES(v_sub, v_sub_mirror, rank);
  return result;
}

TEST(TEST_CATEGORY, padding_has_effect) {
  auto result = true;
  using ll    = Kokkos::LayoutLeft;
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_0>, ll>();
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_1>, ll>(PARAM_1);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_2>, ll>(PARAM_2);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_3>, ll>(PARAM_3);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_4>, ll>(PARAM_4);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_5>, ll>(PARAM_5);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_6>, ll>(PARAM_6);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_7>, ll>(PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_0>, lr>();
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_1>, lr>(PARAM_1);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_2>, lr>(PARAM_2);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_3>, lr>(PARAM_3);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_4>, lr>(PARAM_4);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_5>, lr>(PARAM_5);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_6>, lr>(PARAM_6);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<ARG_7>, lr>(PARAM_7);
  ASSERT_TRUE(result);
}

TEST(TEST_CATEGORY, view_with_padding_construction) {
  /*Test correct propagation of Padding through View construction*/
  auto result  = true;
  using pad    = std::true_type;
  using no_pad = std::false_type;
  using ll     = Kokkos::LayoutLeft;
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<ARG_0>,
                                 ll, pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<ARG_1>,
                                 ll, pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<ARG_2>,
                                 ll, pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<ARG_3>,
                                 ll, pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<ARG_4>,
                                 ll, pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<ARG_5>,
                                 ll, pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<ARG_6>,
                                 ll, pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<ARG_7>,
                                 ll, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<ARG_0>,
                                 ll, no_pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<ARG_1>,
                                 ll, no_pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<ARG_2>,
                                 ll, no_pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<ARG_3>,
                                 ll, no_pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<ARG_4>,
                                 ll, no_pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<ARG_5>,
                                 ll, no_pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<ARG_6>,
                                 ll, no_pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<ARG_7>,
                                 ll, no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<ARG_0>,
                                 lr, pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<ARG_1>,
                                 lr, pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<ARG_2>,
                                 lr, pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<ARG_3>,
                                 lr, pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<ARG_4>,
                                 lr, pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<ARG_5>,
                                 lr, pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<ARG_6>,
                                 lr, pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<ARG_7>,
                                 lr, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<ARG_0>,
                                 lr, no_pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<ARG_1>,
                                 lr, no_pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<ARG_2>,
                                 lr, no_pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<ARG_3>,
                                 lr, no_pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<ARG_4>,
                                 lr, no_pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<ARG_5>,
                                 lr, no_pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<ARG_6>,
                                 lr, no_pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<ARG_7>,
                                 lr, no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  // Padding is not allowed for LayoutStride
  // using ls = Kokkos::LayoutStride;
}

TEST(TEST_CATEGORY, view_with_padding_assignements) {
  /*Test correct propagation in view assignements*/
  auto result  = true;
  using pad    = std::true_type;
  using no_pad = std::false_type;
  using ll     = Kokkos::LayoutLeft;

  result &=
      padding_assignment_and_copy<VIEW_T<DType_0, ll>>(RANK_0 /*, PARAM_0*/);
  result &= padding_assignment_and_copy<VIEW_T<DType_1, ll>>(RANK_1, PARAM_1);
  result &= padding_assignment_and_copy<VIEW_T<DType_2, ll>>(RANK_2, PARAM_2);
  result &= padding_assignment_and_copy<VIEW_T<DType_3, ll>>(RANK_3, PARAM_3);
  result &= padding_assignment_and_copy<VIEW_T<DType_4, ll>>(RANK_4, PARAM_4);
  result &= padding_assignment_and_copy<VIEW_T<DType_5, ll>>(RANK_5, PARAM_5);
  result &= padding_assignment_and_copy<VIEW_T<DType_6, ll>>(RANK_6, PARAM_6);
  result &= padding_assignment_and_copy<VIEW_T<DType_7, ll>>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  using lr = Kokkos::LayoutRight;
  result &=
      padding_assignment_and_copy<VIEW_T<DType_0, lr>>(RANK_0 /*, PARAM_0*/);
  result &= padding_assignment_and_copy<VIEW_T<DType_1, lr>>(RANK_1, PARAM_1);
  result &= padding_assignment_and_copy<VIEW_T<DType_2, lr>>(RANK_2, PARAM_2);
  result &= padding_assignment_and_copy<VIEW_T<DType_3, lr>>(RANK_3, PARAM_3);
  result &= padding_assignment_and_copy<VIEW_T<DType_4, lr>>(RANK_4, PARAM_4);
  result &= padding_assignment_and_copy<VIEW_T<DType_5, lr>>(RANK_5, PARAM_5);
  result &= padding_assignment_and_copy<VIEW_T<DType_6, lr>>(RANK_6, PARAM_6);
  result &= padding_assignment_and_copy<VIEW_T<DType_7, lr>>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  // Padding is not allowed for LayoutStride
  // using ls = Kokkos::LayoutStride;
}

TEST(TEST_CATEGORY, view_with_padding_copying) {
  /*Test correct copy semantic when copying padded views*/
  auto result  = true;
  using pad    = std::true_type;
  using no_pad = std::false_type;
  using ll     = Kokkos::LayoutLeft;

  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<ARG_1>, ll,
                               pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<ARG_2>, ll,
                               pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<ARG_3>, ll,
                               pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<ARG_4>, ll,
                               pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<ARG_5>, ll,
                               pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<ARG_6>, ll,
                               pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<ARG_7>, ll,
                               pad>(RANK_7, PARAM_7);
  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<ARG_1>, ll,
                               no_pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<ARG_2>, ll,
                               no_pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<ARG_3>, ll,
                               no_pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<ARG_4>, ll,
                               no_pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<ARG_5>, ll,
                               no_pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<ARG_6>, ll,
                               no_pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<ARG_7>, ll,
                               no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;
  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<ARG_1>, lr,
                               pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<ARG_2>, lr,
                               pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<ARG_3>, lr,
                               pad>(RANK_3, PARAM_3);

  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<ARG_4>, lr,
                               pad>(RANK_4, PARAM_4);

  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<ARG_5>, lr,
                               pad>(RANK_5, PARAM_5);

  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<ARG_6>, lr,
                               pad>(RANK_6, PARAM_6);
  ASSERT_TRUE(result);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<ARG_7>, lr,
                               pad>(RANK_7, PARAM_7);
  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<ARG_1>, lr,
                               no_pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<ARG_2>, lr,
                               no_pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<ARG_3>, lr,
                               no_pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<ARG_4>, lr,
                               no_pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<ARG_5>, lr,
                               no_pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<ARG_6>, lr,
                               no_pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<ARG_7>, lr,
                               no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  // Padding is not allowed for LayoutStride
  // using ls = Kokkos::LayoutStride;
}

#undef SUBRANGE_DIM
#undef ZERO

#undef RANK_0
#undef RANK_1
#undef RANK_2
#undef RANK_3
#undef RANK_4
#undef RANK_5
#undef RANK_6
#undef RANK_7

#undef ARG_0
#undef ARG_1
#undef ARG_2
#undef ARG_3
#undef ARG_4
#undef ARG_5
#undef ARG_6
#undef ARG_7

#undef PARAM_0
#undef PARAM_1
#undef PARAM_2
#undef PARAM_3
#undef PARAM_4
#undef PARAM_5
#undef PARAM_6
#undef PARAM_7

#undef VIEW_T

}  // namespace Test
