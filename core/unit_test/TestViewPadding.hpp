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

using pair_t = Kokkos::pair<int, int>;

template <typename View, typename Offset>
bool test_view_offset_strides(View &view, Offset &offset, int rank) {
  return (view.stride(0) == offset.stride_0() || rank < 1) &&
         (view.stride(1) == offset.stride_1() || rank < 2) &&
         (view.stride(2) == offset.stride_2() || rank < 3) &&
         (view.stride(3) == offset.stride_3() || rank < 4) &&
         (view.stride(4) == offset.stride_4() || rank < 5) &&
         (view.stride(5) == offset.stride_5() || rank < 6) &&
         (view.stride(6) == offset.stride_6() || rank < 7) &&
         (view.stride(7) == offset.stride_7() || rank < 8);
}

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

template <typename viewDim, typename layout, typename... Vals>
bool padding_has_effect(Vals... params) {
  using allow_padding = std::true_type;
  using offset_t      = Kokkos::Impl::ViewOffset<viewDim, layout, void>;
  using use_padding   = std::bool_constant<
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

  if constexpr (std::is_same<viewDim, Kokkos::Impl::ViewDimension<>>::value ||
                std::is_same<viewDim, Kokkos::Impl::ViewDimension<0>>::value) {
    /* We never pad for rank 0 and rank 1 */
    return offset.span_is_contiguous() == offset_wPadding.span_is_contiguous();
  } else {
    /* if we pad, the span becomes non-contiguous*/
    return offset.span_is_contiguous() != offset_wPadding.span_is_contiguous();
  }
}

template <typename layout>
bool padding_has_effect_helper() {
  auto result = true;
  // clang-format off
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<>,                         layout>();
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<0>,                        layout>(50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 0>,                    layout>(50, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 0, 50>,                layout>(50, 1, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 1, 0, 50>,             layout>(50, 1, 1, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 1, 0, 1, 50>,          layout>(50, 1, 1, 1, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 50>,       layout>(50, 1, 1, 1, 1, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 50>,    layout>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_has_effect<Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 1, 50>, layout>(50, 1, 1, 1, 1, 1, 1, 50);
  // clang-format on
  return result;
}

template <typename dim_t, typename viewDim, typename layout,
          typename allow_padding, typename... Vals>
bool padding_construction(Vals... params) {
  constexpr int rank = sizeof...(params);
  using view_t       = Kokkos::View<dim_t, layout>;
  using offset_t     = Kokkos::Impl::ViewOffset<viewDim, layout, void>;

  using use_padding = std::bool_constant<
      allow_padding() &&
      !std::is_same<layout, typename Kokkos::LayoutStride>::value>;
  using padding =
      std::integral_constant<unsigned int,
                             use_padding::value ? sizeof(DType) : 0>;
  offset_t offset = offset_t(padding(), layout(params...));

  if constexpr (std::is_same_v<allow_padding, std::true_type>) {
    auto alloc_prop =
        Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
    view_t v(alloc_prop, params...);
    return test_view_offset_strides(v, offset, rank);
  } else {
    auto alloc_prop = Kokkos::view_alloc("vDim" + std::to_string(rank));
    view_t v(alloc_prop, params...);
    return test_view_offset_strides(v, offset, rank);
  }
}

template <typename layout, typename allow_padding>
bool padding_construction_helper() {
  auto result = true;
  // clang-format off
  result &= padding_construction<DType_0, Kokkos::Impl::ViewDimension<>,                         layout, allow_padding>();
  result &= padding_construction<DType_1, Kokkos::Impl::ViewDimension<0>,                        layout, allow_padding>(50);
  result &= padding_construction<DType_2, Kokkos::Impl::ViewDimension<50, 0>,                    layout, allow_padding>(50, 50);
  result &= padding_construction<DType_3, Kokkos::Impl::ViewDimension<50, 0, 50>,                layout, allow_padding>(50, 1, 50);
  result &= padding_construction<DType_4, Kokkos::Impl::ViewDimension<50, 1, 0, 50>,             layout, allow_padding>(50, 1, 1, 50);
  result &= padding_construction<DType_5, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 50>,          layout, allow_padding>(50, 1, 1, 1, 50);
  result &= padding_construction<DType_6, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 50>,       layout, allow_padding>(50, 1, 1, 1, 1, 50);
  result &= padding_construction<DType_7, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 50>,    layout, allow_padding>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_construction<DType_8, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 1, 50>, layout, allow_padding>(50, 1, 1, 1, 1, 1, 1, 50);
  // clang-format on
  return result;
}

template <typename view_t, typename... Vals>
bool padding_assignment_and_copy(Vals... params) {
  constexpr int rank = sizeof...(params);
  bool result        = true;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  deep_copy(v, 1);
  auto v_mirror = Kokkos::create_mirror_view(v);
  result &= test_view_strides(v_mirror, v, rank);

  /*deep_copy should support strided views if stride from padding*/
  Kokkos::deep_copy(v_mirror, v);
  return result;
}

template <typename layout>
bool padding_assignment_and_copy_helper() {
  auto result = true;
  // clang-format off
  result &= padding_assignment_and_copy<Kokkos::View<DType_0, layout>>();
  result &= padding_assignment_and_copy<Kokkos::View<DType_1, layout>>(50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_2, layout>>(50, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_3, layout>>(50, 1, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_4, layout>>(50, 1, 1, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_5, layout>>(50, 1, 1,1, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_6, layout>>(50, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_7, layout>>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_assignment_and_copy<Kokkos::View<DType_8, layout>>(50, 1, 1, 1, 1, 1, 1, 50);
  // clang-format off
  return result;
}

template <typename dim_t, typename layout, typename... all_types>
auto get_subview_helper(Kokkos::View<dim_t, layout> &view,
                        std::tuple<all_types...>) {
  auto pair = pair_t(20, 30);
  return Kokkos::subview(view, pair, all_types{}...);
}

template <int Rank>
struct SubViewArgumentTuple {
  using type =
      decltype(std::tuple_cat(typename SubViewArgumentTuple<Rank - 1>::type{},
                              std::tuple<Kokkos::Impl::ALL_t>{}));
};

template <>
struct SubViewArgumentTuple<1> {
  using type = std::tuple<>;
};

template <typename dim_t, typename layout, typename viewDim>
auto get_subview(Kokkos::View<dim_t, layout> &view) {
  if constexpr (std::is_same_v<viewDim, Kokkos::Impl::ViewDimension<>>)
    return Kokkos::subview(view, 0);
  else
    return get_subview_helper(
        view, typename SubViewArgumentTuple<viewDim::rank>::type{});
}

template <typename dim_t, typename viewDim, typename layout, typename... Vals>
bool padding_subview_and_copy(Vals... params) {
  constexpr int rank = sizeof...(params);
  bool result        = true;
  using view_t       = Kokkos::View<dim_t, layout>;
  auto alloc_prop =
      Kokkos::view_alloc("vDim" + std::to_string(rank), Kokkos::AllowPadding);
  view_t v(alloc_prop, params...);
  auto v_sub        = get_subview<dim_t, layout, viewDim>(v);
  auto v_sub_mirror = Kokkos::create_mirror_view(v_sub);
  result &= test_view_strides(v, v_sub, rank);

  /* We never pad for rank 0 and rank 1 and for LR && dyn_rank > 1 */
  if constexpr (std::is_same_v<viewDim, Kokkos::Impl::ViewDimension<>> ||
                std::is_same_v<viewDim, Kokkos::Impl::ViewDimension<0>>)
    result &= test_view_strides(v_sub, v_sub_mirror, rank);
  else {
    if (v_sub_mirror.span_is_contiguous())
      result &= !(test_view_strides(v_sub, v_sub_mirror, rank));
    else
      result &= test_view_strides(v_sub, v_sub_mirror, rank);
  }
  return result;
}

template <typename layout>
bool padding_subview_and_copy_helper() {
  auto result = true;
  // clang-format off
  result &= padding_subview_and_copy<DType_1, Kokkos::Impl::ViewDimension<0>,                        layout>(50);
  result &= padding_subview_and_copy<DType_2, Kokkos::Impl::ViewDimension<50, 0>,                    layout>(50, 50);
  result &= padding_subview_and_copy<DType_3, Kokkos::Impl::ViewDimension<50, 0, 50>,                layout>(50, 1, 50);
  result &= padding_subview_and_copy<DType_4, Kokkos::Impl::ViewDimension<50, 1, 0, 50>,             layout>(50, 1, 1, 50);
  result &= padding_subview_and_copy<DType_5, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 50>,          layout>(50, 1, 1, 1, 50);
  result &= padding_subview_and_copy<DType_6, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 50>,       layout>(50, 1, 1, 1, 1, 50);
  result &= padding_subview_and_copy<DType_7, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 50>,    layout>(50, 1, 1, 1, 1, 1, 50);
  result &= padding_subview_and_copy<DType_8, Kokkos::Impl::ViewDimension<50, 1, 0, 1, 1, 1, 1, 50>, layout>(50, 1, 1, 1, 1, 1, 1, 50);
  // clang-format on
  return result;
}

TEST(TEST_CATEGORY, padding_has_effect) {
  ASSERT_TRUE(padding_has_effect_helper<Kokkos::LayoutLeft>());
  ASSERT_TRUE(padding_has_effect_helper<Kokkos::LayoutRight>());
}

TEST(TEST_CATEGORY, view_with_padding_construction) {
  /*Test correct propagation of Padding through View construction*/
  using pad    = std::true_type;
  using no_pad = std::false_type;
  ASSERT_TRUE((padding_construction_helper<Kokkos::LayoutLeft, pad>()));
  ASSERT_TRUE((padding_construction_helper<Kokkos::LayoutLeft, no_pad>()));
  ASSERT_TRUE((padding_construction_helper<Kokkos::LayoutRight, pad>()));
  ASSERT_TRUE((padding_construction_helper<Kokkos::LayoutRight, no_pad>()));
}

TEST(TEST_CATEGORY, view_with_padding_assignements) {
  /*Test correct propagation in view assignements*/
  ASSERT_TRUE(padding_assignment_and_copy_helper<Kokkos::LayoutLeft>());
  ASSERT_TRUE(padding_assignment_and_copy_helper<Kokkos::LayoutRight>());
}

TEST(TEST_CATEGORY, view_with_padding_copying) {
  /*Test correct copy semantic when copying padded views*/
  ASSERT_TRUE(padding_subview_and_copy_helper<Kokkos::LayoutLeft>());
  ASSERT_TRUE(padding_subview_and_copy_helper<Kokkos::LayoutRight>());
}

}  // namespace Test
