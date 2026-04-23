// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// CUDA-specific tests for Kokkos::Experimental::CudaEvent and
// Event<Kokkos::Cuda>.  These test the native cudaEvent_t API surface directly
// and are intentionally separate from the portable TestEvent.hpp tests so they
// are only compiled (and appear in test output) when the CUDA backend is the
// primary backend under test.

#include <TestCuda_Category.hpp>
#include <TestEventHelpers.hpp>
#include <gtest/gtest.h>

namespace Test {

TEST(TEST_CATEGORY, cuda_event_record_and_wait) {
  using view_type = Kokkos::View<int*, Kokkos::CudaSpace>;

  Kokkos::Cuda exec_space;
  const int N = 1000;

  view_type data("data", N);

  Kokkos::parallel_for("fill",
                       Kokkos::RangePolicy<Kokkos::Cuda>(exec_space, 0, N),
                       FillFunctor<view_type>{data});

  auto evt = Kokkos::Experimental::CudaEvent::record_event(exec_space);
  evt.wait();

  auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(h_data(i), i + 1);
  }
}

TEST(TEST_CATEGORY, cuda_event_cross_stream_dependency) {
  using view_type   = Kokkos::View<int*, Kokkos::CudaSpace>;
  using result_type = Kokkos::View<int64_t, Kokkos::CudaSpace>;

  Kokkos::Cuda stream_a;
  Kokkos::Cuda stream_b;
  const int N = 10000;

  view_type data("data", N);

  Kokkos::parallel_for("produce",
                       Kokkos::RangePolicy<Kokkos::Cuda>(stream_a, 0, N),
                       ProduceFunctor<view_type>{data});

  auto evt = Kokkos::Experimental::CudaEvent::record_event(stream_a);
  evt.add_dependency_to(stream_b);

  result_type result("result");
  Kokkos::deep_copy(stream_b, result, static_cast<int64_t>(0));

  Kokkos::parallel_for("consume",
                       Kokkos::RangePolicy<Kokkos::Cuda>(stream_b, 0, N),
                       ConsumeFunctor<view_type, result_type>{data, result});

  stream_b.fence();

  auto h_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

  int64_t expected = 0;
  for (int i = 0; i < N; ++i) expected += i * 2;
  ASSERT_EQ(h_result(), expected);
}

TEST(TEST_CATEGORY, cuda_event_move_semantics) {
  Kokkos::Cuda exec_space;

  Kokkos::Experimental::CudaEvent evt1;
  evt1.record(exec_space);

  Kokkos::Experimental::CudaEvent evt2(std::move(evt1));
  ASSERT_NE(evt2.cuda_event(), nullptr);

  evt2.wait();
  ASSERT_TRUE(evt2.is_complete());
}

TEST(TEST_CATEGORY, cuda_event_alias_template) {
  Kokkos::Cuda exec_space;

  Kokkos::Experimental::Event<Kokkos::Cuda> evt;
  evt.record(exec_space);
  evt.wait();
  ASSERT_TRUE(evt.is_complete());
}

}  // namespace Test
