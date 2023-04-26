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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TEST_BINSORT_SETB_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TEST_BINSORT_SETB_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <TestStdAlgorithmsHelperFunctors.hpp>
#include <random>

namespace Test {
namespace BinSortWithLayoutStride {

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctorRank2 {
  ViewTypeFrom m_view_from;
  ViewTypeTo m_view_to;

  CopyFunctorRank2() = delete;

  CopyFunctorRank2(const ViewTypeFrom view_from, const ViewTypeTo view_to)
      : m_view_from(view_from), m_view_to(view_to) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int k) const {
    const auto i    = k / m_view_from.extent(1);
    const auto j    = k % m_view_from.extent(1);
    m_view_to(i, j) = m_view_from(i, j);
  }
};

template <class ViewType>
auto create_deep_copyable_compatible_view_with_same_extent(ViewType view) {
  using view_value_type  = typename ViewType::value_type;
  using view_exespace    = typename ViewType::execution_space;
  const std::size_t ext0 = view.extent(0);
  if constexpr (ViewType::rank == 1) {
    using view_deep_copyable_t = Kokkos::View<view_value_type*, view_exespace>;
    return view_deep_copyable_t{"view_dc", ext0};
  } else {
    static_assert(ViewType::rank == 2, "Only rank 1 or 2 supported.");
    using view_deep_copyable_t = Kokkos::View<view_value_type**, view_exespace>;
    const std::size_t ext1     = view.extent(1);
    return view_deep_copyable_t{"view_dc", ext0, ext1};
  }
}

template <class ViewType>
auto create_deep_copyable_compatible_clone(ViewType view) {
  auto view_dc    = create_deep_copyable_compatible_view_with_same_extent(view);
  using view_dc_t = decltype(view_dc);
  if constexpr (ViewType::rank == 1) {
    Test::stdalgos::CopyFunctor<ViewType, view_dc_t> F1(view, view_dc);
    Kokkos::parallel_for("copy", view.extent(0), F1);
  } else {
    static_assert(ViewType::rank == 2, "Only rank 1 or 2 supported.");
    CopyFunctorRank2<ViewType, view_dc_t> F1(view, view_dc);
    Kokkos::parallel_for("copy", view.extent(0) * view.extent(1), F1);
  }
  return view_dc;
}

template <class ViewType>
auto create_host_space_copy(ViewType view) {
  auto view_dc = create_deep_copyable_compatible_clone(view);
  return create_mirror_view_and_copy(Kokkos::HostSpace(), view_dc);
}

template <class KeyType, class ExecutionSpace>
auto create_rank1_dev_and_host_views_of_keys(const ExecutionSpace& exec,
                                             int N) {
  namespace KE = Kokkos::Experimental;
  Kokkos::DefaultHostExecutionSpace defaultHostExeSpace;

  using KeyViewType = Kokkos::View<KeyType*, ExecutionSpace>;
  KeyViewType keys("keys", N);
  auto keys_h = Kokkos::create_mirror_view(keys);
  std::iota(KE::begin(keys_h), KE::end(keys_h), KeyType(0));
  KE::reverse(defaultHostExeSpace, keys_h);
  // keys now is = [N-1,N-2,...,2,1,0], shuffle it for avoid trivial case
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(KE::begin(keys_h), KE::end(keys_h), g);
  Kokkos::deep_copy(exec, keys, keys_h);

  return std::make_pair(keys, keys_h);
}

template <class ExecutionSpace, class KeyType, class ValueType>
void test_on_view_with_layout_stride_1d(int N) {
  ExecutionSpace exec;
  Kokkos::DefaultHostExecutionSpace defaultHostExeSpace;
  namespace KE = Kokkos::Experimental;

  // 1. generate 1D view of keys
  auto [keys, keys_h] =
      create_rank1_dev_and_host_views_of_keys<KeyType>(exec, N);
  using KeyViewType = decltype(keys);

  // 2. create sorter
  using BinOp = Kokkos::BinOp1D<KeyViewType>;
  auto it     = KE::minmax_element(defaultHostExeSpace, keys_h);
  BinOp binner(N, *it.first, *it.second);
  Kokkos::BinSort<KeyViewType, BinOp> Sorter(keys, binner, false);
  Sorter.create_permute_vector(exec);

  // 3. sort 1D view with strided layout
  Kokkos::LayoutStride layout{std::size_t(N), 3};
  using v_t = Kokkos::View<ValueType*, Kokkos::LayoutStride, ExecutionSpace>;
  v_t v("v", layout);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(73931);
  Kokkos::fill_random(v, pool, ValueType(545));
  auto v_before_sort_h = create_host_space_copy(v);
  Sorter.sort(exec, v);
  auto v_after_sort_h = create_host_space_copy(v);

  for (size_t i = 0; i < size_t(N); ++i) {
    EXPECT_TRUE(v_before_sort_h(i) == v_after_sort_h(keys_h(i)));
  }
}

template <class ExecutionSpace, class KeyType, class ValueType>
void test_on_2d_view_with_stride(int numRows, int numCols, int indB, int indE) {
  ExecutionSpace exec;
  Kokkos::DefaultHostExecutionSpace defaultHostExeSpace;
  namespace KE = Kokkos::Experimental;

  // 1. generate 1D view of keys
  auto [keys, keys_h] =
      create_rank1_dev_and_host_views_of_keys<KeyType>(exec, numRows);
  using KeyViewType = decltype(keys);

  // need to store this map from key to row because it is used later for
  // checking
  std::unordered_map<KeyType, int> keyToRowBeforeSort;
  for (int i = 0; i < numRows; ++i) {
    keyToRowBeforeSort[keys_h(i)] = i;
  }

  // 2. create binOp
  using BinOp = Kokkos::BinOp1D<KeyViewType>;
  auto itB    = KE::cbegin(keys_h) + indB;
  auto itE    = itB + indE - indB;
  // std::cout << *itB << " " << *itE << std::endl;
  auto it = KE::minmax_element(defaultHostExeSpace, itB, itE);
  BinOp binner(indE - indB + 5, *it.first, *it.second);
  //  std::cout << *it.first << " " << *it.second << std::endl;

  // 3. create sorter
  Kokkos::BinSort<KeyViewType, BinOp> sorter(keys, indB, indE, binner, false);
  sorter.create_permute_vector(exec);
  sorter.sort(exec, keys_h, indB, indE);
  // std::cout << "Keys after sorting\n";
  // for (size_t i=0; i<keys_h.extent(0); ++i){ std::cout << keys_h(i) << " "; }
  // std::cout << "\n";

  // 3. sort 2D view with strided layout
  Kokkos::LayoutStride layout{size_t(numRows), 2, size_t(numCols),
                              size_t(numRows * 2)};
  using v_t = Kokkos::View<ValueType**, Kokkos::LayoutStride, ExecutionSpace>;
  v_t v("v", layout);
  Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace> pool(73931);
  Kokkos::fill_random(v, pool, ValueType(545));
  auto v_before_sort_h = create_host_space_copy(v);
  sorter.sort(exec, v, indB, indE);
  auto v_after_sort_h = create_host_space_copy(v);

  // std::cout << "\n values before sortin\n";
  // for (size_t i=0; i<v.extent(0); ++i){
  //   std::cout << i << " " << keys_h(i) << " ";
  //   for (size_t j=0; j<v.extent(1); ++j){
  //     std::cout << v_before_sort_h(i,j) << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\nAfter sortin\n";
  // for (size_t i=0; i<v.extent(0); ++i){
  //   std::cout << i << " ";
  //   for (size_t j=0; j<v.extent(1); ++j){
  //     std::cout << v_after_sort_h(i,j) << " ";
  //   }
  //   std::cout << "\n";
  // }
  // std::cout << "\n";

  for (size_t i = 0; i < v.extent(0); ++i) {
    // if i is within the target bounds indB,indE, the sorting was done
    // so we need to do proper checking since rows have changed
    if (i >= size_t(indB) && i < size_t(indE)) {
      const KeyType key = keys_h(i);
      for (size_t j = 0; j < v.extent(1); ++j) {
        EXPECT_TRUE(v_before_sort_h(keyToRowBeforeSort.at(key), j) ==
                    v_after_sort_h(i, j));
      }
    }
    // if we are NOT within the target bounds, then the i-th row remains
    // unchanged
    else {
      for (size_t j = 0; j < v.extent(1); ++j) {
        EXPECT_TRUE(v_before_sort_h(i, j) == v_after_sort_h(i, j));
      }
    }
  }
}
}  // namespace BinSortWithLayoutStride

TEST(TEST_CATEGORY, BinSortUnsignedKeyStridedValuesView) {
  using key_type = unsigned;
  for (int Nr : {10, 55, 189, 1157}) {
    for (int Nc : {1, 3, 5, 111}) {
      // various cases for bounds
      BinSortWithLayoutStride::test_on_2d_view_with_stride<
          TEST_EXECSPACE, key_type, int /*ValueType*/>(Nr, Nc, 0, Nr);
      BinSortWithLayoutStride::test_on_2d_view_with_stride<
          TEST_EXECSPACE, key_type, int /*ValueType*/>(Nr, Nc, 3, Nr);
      BinSortWithLayoutStride::test_on_2d_view_with_stride<
          TEST_EXECSPACE, key_type, int /*ValueType*/>(Nr, Nc, 0, Nr - 4);
      BinSortWithLayoutStride::test_on_2d_view_with_stride<
          TEST_EXECSPACE, key_type, int /*ValueType*/>(Nr, Nc, 4, Nr - 3);
    }
  }
}

}  // namespace Test

#endif
