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

template <typename T>
struct MDReduceFunctor {
  using value_type = T[];

  const int value_count;
  Kokkos::View<T***, Kokkos::DefaultExecutionSpace> m;

  MDReduceFunctor(const Kokkos::View<T***, Kokkos::DefaultExecutionSpace>& m_,
                  int reduce_view_size)
      : value_count(reduce_view_size), m(m_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, value_type sum) const {
    for (int k = 0; k < value_count; ++k) {
      sum[k] += m(i, j, k);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type update) const {
    for (int k = 0; k < value_count; ++k) {
      update[k] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION void final(value_type) const {}
};

template <typename T>
void MDRangeReduceViewTester(const int reduce_view_size) {
  using PolicyType =
      Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>;
  using point_t    = typename PolicyType::point_type;
  using index_type = typename PolicyType::index_type;

  const index_type N(111);

  point_t lower_bound{0, 0};
  point_t upper_bound{N, N};

  Kokkos::View<T***, Kokkos::DefaultExecutionSpace> data_3D("data_3D", N, N,
                                                            reduce_view_size);
  Kokkos::View<T*, Kokkos::DefaultExecutionSpace> data_1D("data_1D",
                                                          reduce_view_size);

  Kokkos::deep_copy(data_1D, static_cast<T>(0.0));
  Kokkos::deep_copy(data_3D, static_cast<T>(1.0));

  // Perform MDRange parallel reduce
  PolicyType policy(lower_bound, upper_bound);
  MDReduceFunctor<T> functor(data_3D, reduce_view_size);
  Kokkos::parallel_reduce(policy, functor, data_1D);
  Kokkos::fence();

  // Verify results
  auto host_data_1D =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data_1D);
  for (int i = 0; i < reduce_view_size; i++) {
    ASSERT_EQ(host_data_1D(i), T(N * N));
  }
}

TEST(TEST_CATEGORY, mdrange_parallel_reduce_primitive_types) {
  for (int bound : {0, 1, 7, 32, 65, 7000}) {
    for (int k = 0; k < bound; ++k) {
      MDRangeReduceTester<bool>(bound, k);
      MDRangeReduceTester<signed char>(bound, k);
      MDRangeReduceTester<int8_t>(bound, k);
      MDRangeReduceTester<int16_t>(bound, k);
      MDRangeReduceTester<int32_t>(bound, k);
      MDRangeReduceTester<int64_t>(bound, k);
    }
  }
}

TEST(TEST_CATEGORY, mdrange_parallel_reduce_view_type) {
  for (int reduce_view_size : {1, 2, 3, 15, 16, 17, 31, 32, 33, 64}) {
    MDRangeReduceViewTester<double>(reduce_view_size);
    MDRangeReduceViewTester<float>(reduce_view_size);
    MDRangeReduceViewTester<int32_t>(reduce_view_size);
    MDRangeReduceViewTester<int16_t>(reduce_view_size);
  }
}

}  // namespace
