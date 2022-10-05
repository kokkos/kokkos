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
#define PARAM_0
#define PARAM_1 30
#define PARAM_2 30, ZERO
#define PARAM_3 30, ZERO, 30
#define PARAM_4 30, 2, ZERO, 30
#define PARAM_5 30, 2, ZERO, 2, 30
#define PARAM_6 30, 2, ZERO, 1, 2, 30
#define PARAM_7 30, 2, ZERO, 1, 1, 2, 30

#define SUBRANGE_DIM0
#define SUBRANGE_DIMX 10, 15

using DType   = int;
using DType_0 = DType;
using DType_1 = DType *;
using DType_2 = DType **;
using DType_3 = DType ***;
using DType_4 = DType ****;
using DType_5 = DType *****;
using DType_6 = DType ******;
using DType_7 = DType *******;

#define ASSERT_OFFSET_STRIDES(view, offset)    \
  (view.stride(0) == offset.stride_0()) &&     \
      (view.stride(1) == offset.stride_1()) && \
      (view.stride(2) == offset.stride_2()) && \
      (view.stride(3) == offset.stride_3()) && \
      (view.stride(4) == offset.stride_4()) && \
      (view.stride(5) == offset.stride_5()) && \
      (view.stride(6) == offset.stride_6()) && \
      (view.stride(7) == offset.stride_7())

#define ASSERT_VIEW_STRIDES(view1, view2)     \
  (view1.stride(0) == view2.stride(0)) &&     \
      (view1.stride(1) == view2.stride(1)) && \
      (view1.stride(2) == view2.stride(2)) && \
      (view1.stride(3) == view2.stride(3)) && \
      (view1.stride(4) == view2.stride(4)) && \
      (view1.stride(5) == view2.stride(5)) && \
      (view1.stride(6) == view2.stride(6)) && \
      (view1.stride(7) == view2.stride(7))

template <typename T, typename viewDim, typename layout, typename allow_padding,
          typename... Vals>
typename std::enable_if_t<std::is_same<allow_padding, std::true_type>::value,
                          bool>
padding_construction(int dim, Vals... params) {
  using view_t      = Kokkos::View<T, layout>;
  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = std::bool_constant<
      allow_padding() &&
      !std::is_same<layout, typename Kokkos::LayoutStride>::value>;
  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  offset_t offset = offset_t(padding(), layout(params...));
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(dim), Kokkos::AllowPadding);

  view_t v(alloc_prop, params...);
  return ASSERT_OFFSET_STRIDES(v, offset) ? true : false;
}

template <typename T, typename viewDim, typename layout, typename allow_padding,
          typename... Vals>
typename std::enable_if_t<std::is_same<allow_padding, std::false_type>::value,
                          bool>
padding_construction(int dim, Vals... params) {
  using view_t      = Kokkos::View<T, layout>;
  using offset_t    = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding = allow_padding;
  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  offset_t offset = offset_t(padding(), layout(params...));
  auto alloc_prop = Kokkos::view_alloc("vDim" + std::to_string(dim));
  view_t v(alloc_prop, params...);
  return ASSERT_OFFSET_STRIDES(v, offset) ? true : false;
}

template <typename T, typename viewDim, typename layout, typename allow_padding,
          typename... Vals>
bool padding_assignement_and_copy(int dim, Vals... params) {
  bool result  = true;
  using view_t = Kokkos::View<T, layout>;

  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(dim), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  deep_copy(v, 1);
  auto v_mirror = Kokkos::create_mirror_view(v);
  result &= ASSERT_VIEW_STRIDES(v_mirror, v) ? true : false;

  /*deep_copy should support strided views if stride from padding*/
  Kokkos::deep_copy(v_mirror, v);
  return result;
}
template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_0>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, 0);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_1>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX));
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_2>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_3>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL, Kokkos::ALL);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_4>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_5>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_6>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL);
}

template <typename T, typename layout, typename viewDim>
auto get_subview(
    Kokkos::View<T, layout> &view,
    typename std::enable_if_t<
        std::is_same<viewDim, Kokkos::Impl::ViewDimension<PARAM_7>>::value,
        void> *ptr = nullptr) {
  return Kokkos::subview(view, Kokkos::pair<int, int>(SUBRANGE_DIMX),
                         Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                         Kokkos::ALL, Kokkos::ALL);
}

template <typename T, typename viewDim, typename layout, typename allow_padding,
          typename... Vals>
bool padding_subview_and_copy(int dim, Vals... params) {
  bool result  = true;
  using view_t = Kokkos::View<T, layout>;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(dim), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<T, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);

  // Subview should inherit AllowPadding and thus generate dim0 stride for
  // for padding but should not inherit striding from subview
  result &= ASSERT_VIEW_STRIDES(v_sub, v_sub_mirror);
  return result;
}

TEST(TEST_CATEGORY, view_with_padding_construction) {
  /*Test correct propagation of Padding through View construction*/
  auto result  = true;
  using pad    = std::true_type;
  using no_pad = std::false_type;
  using ll     = Kokkos::LayoutLeft;
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<PARAM_0>,
                                 ll, pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                                 ll, pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                                 ll, pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                                 ll, pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                                 ll, pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                                 ll, pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                                 ll, pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                                 ll, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<PARAM_0>,
                                 ll, no_pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                                 ll, no_pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                                 ll, no_pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                                 ll, no_pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                                 ll, no_pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                                 ll, no_pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                                 ll, no_pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                                 ll, no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<PARAM_0>,
                                 lr, pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                                 lr, pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                                 lr, pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                                 lr, pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                                 lr, pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                                 lr, pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                                 lr, pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                                 lr, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<PARAM_0>,
                                 lr, no_pad>(RANK_0 /*, PARAM_0*/);
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                                 lr, no_pad>(RANK_1, PARAM_1);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                                 lr, no_pad>(RANK_2, PARAM_2);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                                 lr, no_pad>(RANK_3, PARAM_3);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                                 lr, no_pad>(RANK_4, PARAM_4);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                                 lr, no_pad>(RANK_5, PARAM_5);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                                 lr, no_pad>(RANK_6, PARAM_6);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
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
  result &= padding_assignement_and_copy<
      DType_0, Kokkos::Impl::ViewDimension<PARAM_0>, ll, pad>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignement_and_copy<
      DType_1, Kokkos::Impl::ViewDimension<PARAM_1>, ll, pad>(RANK_1, PARAM_1);
  result &= padding_assignement_and_copy<
      DType_2, Kokkos::Impl::ViewDimension<PARAM_2>, ll, pad>(RANK_2, PARAM_2);
  result &= padding_assignement_and_copy<
      DType_3, Kokkos::Impl::ViewDimension<PARAM_3>, ll, pad>(RANK_3, PARAM_3);
  result &= padding_assignement_and_copy<
      DType_4, Kokkos::Impl::ViewDimension<PARAM_4>, ll, pad>(RANK_4, PARAM_4);
  result &= padding_assignement_and_copy<
      DType_5, Kokkos::Impl::ViewDimension<PARAM_5>, ll, pad>(RANK_5, PARAM_5);
  result &= padding_assignement_and_copy<
      DType_6, Kokkos::Impl::ViewDimension<PARAM_6>, ll, pad>(RANK_6, PARAM_6);
  result &= padding_assignement_and_copy<
      DType_7, Kokkos::Impl::ViewDimension<PARAM_7>, ll, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_assignement_and_copy<
      DType_0, Kokkos::Impl::ViewDimension<PARAM_0>, ll, no_pad>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignement_and_copy<
      DType_1, Kokkos::Impl::ViewDimension<PARAM_1>, ll, no_pad>(RANK_1,
                                                                 PARAM_1);
  result &= padding_assignement_and_copy<
      DType_2, Kokkos::Impl::ViewDimension<PARAM_2>, ll, no_pad>(RANK_2,
                                                                 PARAM_2);
  result &= padding_assignement_and_copy<
      DType_3, Kokkos::Impl::ViewDimension<PARAM_3>, ll, no_pad>(RANK_3,
                                                                 PARAM_3);
  result &= padding_assignement_and_copy<
      DType_4, Kokkos::Impl::ViewDimension<PARAM_4>, ll, no_pad>(RANK_4,
                                                                 PARAM_4);
  result &= padding_assignement_and_copy<
      DType_5, Kokkos::Impl::ViewDimension<PARAM_5>, ll, no_pad>(RANK_5,
                                                                 PARAM_5);
  result &= padding_assignement_and_copy<
      DType_6, Kokkos::Impl::ViewDimension<PARAM_6>, ll, no_pad>(RANK_6,
                                                                 PARAM_6);
  result &= padding_assignement_and_copy<
      DType_7, Kokkos::Impl::ViewDimension<PARAM_7>, ll, no_pad>(RANK_7,
                                                                 PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;
  result &= padding_assignement_and_copy<
      DType_0, Kokkos::Impl::ViewDimension<PARAM_0>, lr, pad>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignement_and_copy<
      DType_1, Kokkos::Impl::ViewDimension<PARAM_1>, lr, pad>(RANK_1, PARAM_1);
  result &= padding_assignement_and_copy<
      DType_2, Kokkos::Impl::ViewDimension<PARAM_2>, lr, pad>(RANK_2, PARAM_2);
  result &= padding_assignement_and_copy<
      DType_3, Kokkos::Impl::ViewDimension<PARAM_3>, lr, pad>(RANK_3, PARAM_3);
  result &= padding_assignement_and_copy<
      DType_4, Kokkos::Impl::ViewDimension<PARAM_4>, lr, pad>(RANK_4, PARAM_4);
  result &= padding_assignement_and_copy<
      DType_5, Kokkos::Impl::ViewDimension<PARAM_5>, lr, pad>(RANK_5, PARAM_5);
  result &= padding_assignement_and_copy<
      DType_6, Kokkos::Impl::ViewDimension<PARAM_6>, lr, pad>(RANK_6, PARAM_6);
  result &= padding_assignement_and_copy<
      DType_7, Kokkos::Impl::ViewDimension<PARAM_7>, lr, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  result &= padding_assignement_and_copy<
      DType_0, Kokkos::Impl::ViewDimension<PARAM_0>, lr, no_pad>(
      RANK_0 /*, PARAM_0*/);
  result &= padding_assignement_and_copy<
      DType_1, Kokkos::Impl::ViewDimension<PARAM_1>, lr, no_pad>(RANK_1,
                                                                 PARAM_1);
  result &= padding_assignement_and_copy<
      DType_2, Kokkos::Impl::ViewDimension<PARAM_2>, lr, no_pad>(RANK_2,
                                                                 PARAM_2);
  result &= padding_assignement_and_copy<
      DType_3, Kokkos::Impl::ViewDimension<PARAM_3>, lr, no_pad>(RANK_3,
                                                                 PARAM_3);
  result &= padding_assignement_and_copy<
      DType_4, Kokkos::Impl::ViewDimension<PARAM_4>, lr, no_pad>(RANK_4,
                                                                 PARAM_4);
  result &= padding_assignement_and_copy<
      DType_5, Kokkos::Impl::ViewDimension<PARAM_5>, lr, no_pad>(RANK_5,
                                                                 PARAM_5);
  result &= padding_assignement_and_copy<
      DType_6, Kokkos::Impl::ViewDimension<PARAM_6>, lr, no_pad>(RANK_6,
                                                                 PARAM_6);
  result &= padding_assignement_and_copy<
      DType_7, Kokkos::Impl::ViewDimension<PARAM_7>, lr, no_pad>(RANK_7,
                                                                 PARAM_7);
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
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                               ll, pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                               ll, pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                               ll, pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                               ll, pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                               ll, pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                               ll, pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                               ll, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                               ll, no_pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                               ll, no_pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                               ll, no_pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                               ll, no_pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                               ll, no_pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                               ll, no_pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                               ll, no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);
  using lr = Kokkos::LayoutRight;

  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                               lr, pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                               lr, pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                               lr, pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                               lr, pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                               lr, pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                               lr, pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                               lr, pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  result &=
      padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<PARAM_1>,
                               lr, no_pad>(RANK_1, PARAM_1);
  result &=
      padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<PARAM_2>,
                               lr, no_pad>(RANK_2, PARAM_2);
  result &=
      padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<PARAM_3>,
                               lr, no_pad>(RANK_3, PARAM_3);
  result &=
      padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<PARAM_4>,
                               lr, no_pad>(RANK_4, PARAM_4);
  result &=
      padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<PARAM_5>,
                               lr, no_pad>(RANK_5, PARAM_5);
  result &=
      padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<PARAM_6>,
                               lr, no_pad>(RANK_6, PARAM_6);
  result &=
      padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<PARAM_7>,
                               lr, no_pad>(RANK_7, PARAM_7);
  ASSERT_TRUE(result);

  // Padding is not allowed for LayoutStride
  // using ls = Kokkos::LayoutStride;
}

#undef PARAM_0
#undef PARAM_1
#undef PARAM_2
#undef PARAM_3
#undef PARAM_4
#undef PARAM_5
#undef PARAM_6
#undef PARAM_7

#undef PARAM_0_RANK
#undef PARAM_1_RANK
#undef PARAM_2_RANK
#undef PARAM_3_RANK
#undef PARAM_4_RANK
#undef PARAM_5_RANK
#undef PARAM_6_RANK
#undef PARAM_7_RANK

#undef DType

}  // namespace Test
