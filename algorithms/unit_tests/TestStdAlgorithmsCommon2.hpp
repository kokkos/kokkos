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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON2_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON2_HPP

#include <gtest/gtest.h>
#include <TestStdAlgorithmsHelperFunctors.hpp>
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <numeric>
#include <random>

namespace Test {
namespace stdalgos {

struct DynamicTag {};
struct StridedTwoTag {};
struct StridedThreeTag {};

const std::map<std::string, std::size_t> default_scenarios = {
    {"empty", 0},          {"one-element", 1}, {"two-elements-a", 2},
    {"two-elements-b", 2}, {"small-a", 9},     {"small-b", 13},
    {"medium-a", 1003},    {"medium-b", 1003}, {"large-a", 101513},
    {"large-b", 101513}};

// see cpp file for these functions
std::string view_tag_to_string(DynamicTag);
std::string view_tag_to_string(StridedTwoTag);
std::string view_tag_to_string(StridedThreeTag);

template <class ValueType>
auto create_view(DynamicTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*>;
  view_t view{label + "_" + view_tag_to_string(DynamicTag{}), ext};
  return view;
}

template <class ValueType>
auto create_view(StridedTwoTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext, 2};
  view_t view{label + "_" + view_tag_to_string(DynamicTag{}), layout};
  return view;
}

template <class ValueType>
auto create_view(StridedThreeTag, std::size_t ext, const std::string label) {
  using view_t = Kokkos::View<ValueType*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{ext, 3};
  view_t view{label + "_" + view_tag_to_string(DynamicTag{}), layout};
  return view;
}

template <class ViewType>
auto create_deep_copyable_compatible_view_with_same_extent(ViewType view) {
  const std::size_t ext      = view.extent(0);
  using view_value_type      = typename ViewType::value_type;
  using view_exespace        = typename ViewType::execution_space;
  using view_deep_copyable_t = Kokkos::View<view_value_type*, view_exespace>;
  view_deep_copyable_t view_dc("view_dc", ext);
  return view_dc;
}

template <class ViewType>
auto create_deep_copyable_compatible_clone(ViewType view) {
  auto view_dc    = create_deep_copyable_compatible_view_with_same_extent(view);
  using view_dc_t = decltype(view_dc);
  CopyFunctor<ViewType, view_dc_t> F1(view, view_dc);
  Kokkos::parallel_for("copy", view.extent(0), F1);
  return view_dc;
}

template <class ViewType>
auto create_host_space_copy(ViewType view) {
  auto view_dc = create_deep_copyable_compatible_clone(view);
  return create_mirror_view_and_copy(Kokkos::HostSpace(), view_dc);
}

// fill the views with sequentially increasing values
template <class ViewType, class ViewHostType>
void fill_views_inc(ViewType view, ViewHostType host_view) {
  namespace KE = Kokkos::Experimental;

  Kokkos::parallel_for(view.extent(0), AssignIndexFunctor<ViewType>(view));
  std::iota(KE::begin(host_view), KE::end(host_view), 0);
  // compare_views(expected, view);
}

}  // namespace stdalgos
}  // namespace Test

#endif
