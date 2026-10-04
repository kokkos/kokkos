// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <TestDefaultDeviceType_Category.hpp>

#include <iostream>
#include <stdint.h>

namespace Test {

template <typename data_type, size_t N>
void print_array(Kokkos::Array<data_type, N> array) {
  for (const auto val : array) {
    std::cout << val << ",";
  }
  std::cout << std::endl;
}

template <typename ReducerValue_t, typename index_t>
struct NoOpFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(index_t, ReducerValue_t&) const {}

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t, index_t, ReducerValue_t&) const {}

  KOKKOS_INLINE_FUNCTION
  void operator()(index_t, index_t, index_t, ReducerValue_t&) const {}
};

template <typename ReducerType, typename index_type, size_t rank>
void test_loc_reducer_array_index() {
  using value_type = typename ReducerType::value_type;

  Kokkos::Array<index_type, rank> lower_bound{};
  Kokkos::Array<index_type, rank> upper_bound{};

  value_type result;

  for (size_t i = 0; i < rank; i++) {
    lower_bound[i] = 0;
    upper_bound[i] = 5;
  }

  NoOpFunctor<value_type, index_type> functor{};

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::Rank<rank>, Kokkos::IndexType<index_type>>(
          lower_bound, upper_bound),
      functor, ReducerType(result));
  Kokkos::fence();
  for (size_t i = 0; i < rank; i++) {
    if constexpr (std::is_same_v<value_type,
                                 Kokkos::MinMaxLocScalar<
                                     float, Kokkos::Array<index_type, rank>>>) {
      ASSERT_EQ(result.min_loc[i],
                Kokkos::reduction_identity<index_type>::min());
      ASSERT_EQ(result.max_loc[i],
                Kokkos::reduction_identity<index_type>::min());
    } else {
      ASSERT_EQ(result.loc[i], Kokkos::reduction_identity<index_type>::min());
    }
  }
}

template <typename IndexType, std::size_t Rank>
void test_loc_array_index_reducers() {
  using index_array = Kokkos::Array<IndexType, Rank>;
  test_loc_reducer_array_index<Kokkos::MinLoc<float, index_array>, IndexType,
                               Rank>();
  test_loc_reducer_array_index<Kokkos::MaxLoc<float, index_array>, IndexType,
                               Rank>();
  test_loc_reducer_array_index<Kokkos::MinMaxLoc<float, index_array>, IndexType,
                               Rank>();
}

template <typename IndexType>
void test_loc_array_index_all_ranks() {
  test_loc_array_index_reducers<IndexType, 1>();
  test_loc_array_index_reducers<IndexType, 2>();
  test_loc_array_index_reducers<IndexType, 3>();
}

TEST(defaultdevicetype, development_test) {
  test_loc_array_index_all_ranks<int32_t>();
  test_loc_array_index_all_ranks<int64_t>();
  test_loc_array_index_all_ranks<uint32_t>();
  test_loc_array_index_all_ranks<uint64_t>();
}

}  // namespace Test
