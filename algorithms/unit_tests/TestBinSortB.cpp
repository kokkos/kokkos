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

template <class ExecutionSpace, class ValueType, int ValuesViewRank>
auto create_strided_view(int numRows, int numCols) {
  if constexpr (ValuesViewRank == 1) {
    (void)numCols;
    Kokkos::LayoutStride layout{size_t(numRows), 2};
    using v_t = Kokkos::View<ValueType*, Kokkos::LayoutStride, ExecutionSpace>;
    v_t v("v", layout);
    return v;
  } else {
    Kokkos::LayoutStride layout{size_t(numRows), 2, size_t(numCols),
                                size_t(numRows * 2)};
    using v_t = Kokkos::View<ValueType**, Kokkos::LayoutStride, ExecutionSpace>;
    v_t v("v", layout);
    return v;
  }
}

template <class ExecutionSpace, class KeyType, class ValueType,
          int ValuesViewRank>
void test_on_view_with_stride(int numRows,
			      int indB, int indE,
			      int numCols = 1)
{

  ExecutionSpace exec;
  Kokkos::DefaultHostExecutionSpace defaultHostExeSpace;
  namespace KE = Kokkos::Experimental;

  // 1. generate 1D view of keys
  auto [keys, keys_h] =
      create_rank1_dev_and_host_views_of_keys<KeyType>(exec, numRows);
  using KeyViewType = decltype(keys);
  // std::cout << "Keys before sorting\n";
  // for (size_t i=0; i<keys_h.extent(0); ++i){ std::cout << keys_h(i) << " "; }
  // std::cout << "\n";

  // need this map key->row to use later for checking
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
  // seems like the behavior is odd when we use # buckets = # keys
  // so use +5 for using more buckets than keys.
  // This is something to investigate.
  BinOp binner(indE - indB + 5, *it.first, *it.second);
  //  std::cout << *it.first << " " << *it.second << std::endl;

  // 3. create sorter
  Kokkos::BinSort<KeyViewType, BinOp> sorter(keys, indB, indE, binner, false);
  sorter.create_permute_vector(exec);
  sorter.sort(exec, keys, indB, indE);
  Kokkos::deep_copy(exec, keys_h, keys);
  // std::cout << "Keys after sorting\n";
  // for (size_t i=0; i<keys_h.extent(0); ++i){ std::cout << keys_h(i) << " "; }
  // std::cout << "\n";

  auto v = create_strided_view<ExecutionSpace, ValueType, ValuesViewRank>(
      numRows, numCols);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> pool(73931);
  Kokkos::fill_random(v, pool, ValueType(545));
  auto v_before_sort_h = create_host_space_copy(v);
  sorter.sort(exec, v, indB, indE);
  auto v_after_sort_h = create_host_space_copy(v);

  for (size_t i = 0; i < v.extent(0); ++i) {
    // if i within [indB,indE), the sorting was done
    // so we need to do proper checking since rows have changed
    if (i >= size_t(indB) && i < size_t(indE)) {
      const KeyType key = keys_h(i);
      if constexpr (ValuesViewRank == 1) {
        EXPECT_TRUE(v_before_sort_h(keyToRowBeforeSort.at(key)) ==
                    v_after_sort_h(i));
      } else {
        for (size_t j = 0; j < v.extent(1); ++j) {
          EXPECT_TRUE(v_before_sort_h(keyToRowBeforeSort.at(key), j) ==
                      v_after_sort_h(i, j));
        }
      }
    }
    // outside the target bounds, then the i-th row remains unchanged
    else {
      if constexpr (ValuesViewRank == 1) {
        EXPECT_TRUE(v_before_sort_h(i) == v_after_sort_h(i));
      } else {
        for (size_t j = 0; j < v.extent(1); ++j) {
          EXPECT_TRUE(v_before_sort_h(i, j) == v_after_sort_h(i, j));
        }
      }
    }
  }
}

template<class ExecutionSpace, class KeyType, class ValueType>
void run_for_rank1()
{
  constexpr int rank = 1;

  // trivial case
  test_on_view_with_stride<
    ExecutionSpace, KeyType, ValueType, rank>(1, 0, 1);

  // nontrivial cases
  for (int N : {311, 710017}){
    // various cases for bounds
    test_on_view_with_stride<
      ExecutionSpace, KeyType, ValueType, rank>(N, 0, N);
    test_on_view_with_stride<
      ExecutionSpace, KeyType, ValueType, rank>(N, 3, N);
    test_on_view_with_stride<
      ExecutionSpace, KeyType, ValueType, rank>(N, 0, N - 4);
    test_on_view_with_stride<
      ExecutionSpace, KeyType, ValueType, rank>(N, 4, N - 3);
  }
}

template<class ExecutionSpace, class KeyType, class ValueType>
void run_for_rank2()
{
  constexpr int rank = 2;

  // trivial case
  test_on_view_with_stride<
    ExecutionSpace, KeyType, ValueType, rank>(1, 0, 1, 1);

  // nontrivial cases
  for (int Nr : {11, 1157, 710017}){
    for (int Nc : {3, 51}) {
      // various cases for bounds
      test_on_view_with_stride<
	ExecutionSpace, KeyType, ValueType, rank>(Nr, 0, Nr, Nc);
      test_on_view_with_stride<
	ExecutionSpace, KeyType, ValueType, rank>(Nr, 3, Nr, Nc);
      test_on_view_with_stride<
	ExecutionSpace, KeyType, ValueType, rank>(Nr, 0, Nr - 4, Nc);
      test_on_view_with_stride<
	ExecutionSpace, KeyType, ValueType, rank>(Nr, 4, Nr - 3, Nc);
    }
  }
}

}  // namespace BinSortWithLayoutStride

TEST(BinSort, UnsignedKeyLayoutStrideValues) {
  using ExeSpace = Kokkos::DefaultExecutionSpace;
  using key_type = unsigned;
  BinSortWithLayoutStride::run_for_rank1<ExeSpace, key_type, int>();
  BinSortWithLayoutStride::run_for_rank1<ExeSpace, key_type, double>();

  BinSortWithLayoutStride::run_for_rank2<ExeSpace, key_type, int>();
  BinSortWithLayoutStride::run_for_rank2<ExeSpace, key_type, double>();
}

}  // namespace Test

#endif
