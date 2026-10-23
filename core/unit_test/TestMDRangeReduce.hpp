// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace {

template <typename T>
void MDRangeReduceTester([[maybe_unused]] int bound, int k) {
  const auto policy_MD = Kokkos::MDRangePolicy<Kokkos::Rank<2>, TEST_EXECSPACE>(
      {0, 0}, {bound, 2});

  // No explicit fence() calls needed because result is in HostSpace
  {
    T lor_MD = 0;
    Kokkos::parallel_reduce(
        policy_MD,
        KOKKOS_LAMBDA(const int i, const int, T& res) { res = res || i == k; },
        Kokkos::LOr<T>(lor_MD));
    EXPECT_EQ(lor_MD, 1);
  }
  {
    // Stick just a few true values in the Logical-OR reduction space,
    // to try to make sure every value is being captured
    T land_MD = 0;
    Kokkos::parallel_reduce(
        policy_MD, KOKKOS_LAMBDA(const int, const int, T& res) { res = 1; },
        Kokkos::LAnd<T>(land_MD));
    EXPECT_EQ(land_MD, 1);
  }
}

TEST(TEST_CATEGORY, mdrange_parallel_reduce_primitive_types) {
  auto doit = [](int bound, int k) {
    MDRangeReduceTester<bool>(bound, k);
    MDRangeReduceTester<signed char>(bound, k);
    MDRangeReduceTester<int8_t>(bound, k);
    MDRangeReduceTester<int16_t>(bound, k);
    MDRangeReduceTester<int32_t>(bound, k);
    MDRangeReduceTester<int64_t>(bound, k);
  };

// FIXME_NEXTSILICON: full sequence very slow in handoff 1.3.0-120
#ifdef KOKKOS_ENABLE_NEXTSILICON
  if constexpr (std::is_same_v<TEST_EXECSPACE,
                               Kokkos::Experimental::NextSilicon>) {
    for (int bound : {0, 7000}) {
      for (int k : {0, 3500, 6999}) {
        if (k < bound) doit(bound, k);
      }
    }
    SUCCEED() << "ran shorter NextSilicon test";
    return;
  }
#endif
  for (int bound : {0, 1, 7, 32, 65, 7000}) {
    for (int k = 0; k < bound; ++k) {
      doit(bound, k);
    }
  }
}

template <typename index_t>
struct MinLocTag {
  using array_type   = Kokkos::Array<index_t, 3>;
  using reducer_type = Kokkos::MinLoc<float, array_type>;
};

template <typename index_t>
struct MaxLocTag {
  using array_type   = Kokkos::Array<index_t, 3>;
  using reducer_type = Kokkos::MaxLoc<float, array_type>;
};

template <typename index_t>
struct MinMaxLocTag {
  using array_type   = Kokkos::Array<index_t, 3>;
  using reducer_type = Kokkos::MinMaxLoc<float, array_type>;
};

template <typename ReducerTag>
void test_loc_reducer_array_rank3() {
  using array_type   = typename ReducerTag::array_type;
  using index_type   = typename array_type::value_type;
  using reducer_type = typename ReducerTag::reducer_type;
  using value_type   = typename reducer_type::value_type;

  constexpr bool is_minmax =
      std::is_same_v<ReducerTag, MinMaxLocTag<index_type>>;
  constexpr bool is_min = std::is_same_v<ReducerTag, MinLocTag<index_type>>;

  constexpr float fill_value = 10.f;
  constexpr float min_value  = 5.f;
  constexpr float max_value  = 15.f;

  const array_type lower{0, 0, 0};
  const array_type upper{29, 31, 33};
  const array_type min_pos{5, 17, 28};
  const array_type max_pos{23, 2, 11};

  Kokkos::View<float***, TEST_EXECSPACE> a("a", upper[0], upper[1], upper[2]);
  auto h_a = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(h_a, fill_value);
  // Set min and max value
  h_a(min_pos[0], min_pos[1], min_pos[2]) = min_value;
  h_a(max_pos[0], max_pos[1], max_pos[2]) = max_value;
  Kokkos::deep_copy(a, h_a);

  value_type result;
  reducer_type reducer(result);
  Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<3>,
                        Kokkos::IndexType<index_type>>
      policy_3d(lower, upper);

  // Lambda cannot first-capture variable in constexpr-if context
  if constexpr (is_minmax) {
    Kokkos::parallel_reduce(
        policy_3d,
        KOKKOS_LAMBDA(index_type i, index_type j, index_type k,
                      value_type & upd) {
          const float v = a(i, j, k);
          const array_type loc{i, j, k};
          reducer.join(upd, value_type{v, v, loc, loc});
        },
        reducer);
  } else {
    Kokkos::parallel_reduce(
        policy_3d,
        KOKKOS_LAMBDA(index_type i, index_type j, index_type k,
                      value_type & upd) {
          reducer.join(upd, value_type{a(i, j, k), array_type{i, j, k}});
        },
        reducer);
  }

  if constexpr (is_minmax) {
    ASSERT_EQ(result.min_val, min_value);
    ASSERT_EQ(result.max_val, max_value);
    for (int d = 0; d < 3; ++d) {
      ASSERT_EQ(result.min_loc[d], min_pos[d]);
      ASSERT_EQ(result.max_loc[d], max_pos[d]);
    }
  } else if constexpr (is_min) {
    ASSERT_EQ(result.val, min_value);
    for (int d = 0; d < 3; ++d) {
      ASSERT_EQ(result.loc[d], min_pos[d]);
    }
  } else {
    ASSERT_EQ(result.val, max_value);
    for (int d = 0; d < 3; ++d) {
      ASSERT_EQ(result.loc[d], max_pos[d]);
    }
  }
}

template <typename IndexType>
void test_loc_array_index_reducers() {
  test_loc_reducer_array_rank3<MinLocTag<IndexType>>();
  test_loc_reducer_array_rank3<MaxLocTag<IndexType>>();
  test_loc_reducer_array_rank3<MinMaxLocTag<IndexType>>();
}

TEST(TEST_CATEGORY, min_max_loc_reduction_3d) {
  test_loc_array_index_reducers<int32_t>();
  test_loc_array_index_reducers<int64_t>();
  test_loc_array_index_reducers<uint32_t>();
  test_loc_array_index_reducers<uint64_t>();
}

}  // namespace
