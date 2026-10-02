// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

#include <concepts>

namespace Test {

template <class ViewType>
struct FillFunctor {
  ViewType data;
  template <std::integral T>
  KOKKOS_FUNCTION void operator()(T i) const {
    data(i) = i + 1;
  }
};

template <class ViewType>
struct ProduceFunctor {
  ViewType data;
  template <std::integral T>
  KOKKOS_FUNCTION void operator()(T i) const {
    data(i) = i * 2;
  }
};

template <class ViewType, class ResultType>
struct ConsumeFunctor {
  ViewType data;
  ResultType result;
  template <std::integral T>
  KOKKOS_FUNCTION void operator()(T i) const {
    Kokkos::atomic_add(&result(),
                       static_cast<typename ResultType::value_type>(data(i)));
  }
};

// ============================================================================
// Portable tests -- run for every enabled backend via TEST_EXECSPACE
// ============================================================================

TEST(TEST_CATEGORY, event_record_and_wait) {
  using exec_space   = TEST_EXECSPACE;
  using memory_space = typename exec_space::memory_space;
  using view_type    = Kokkos::View<int*, memory_space>;

  exec_space space;
  constexpr int N = 1000;

  view_type data(Kokkos::view_alloc(space, "data", Kokkos::WithoutInitializing),
                 N);

  Kokkos::parallel_for("fill", Kokkos::RangePolicy(space, 0, N),
                       FillFunctor<view_type>{data});

  Kokkos::Experimental::Event<exec_space> evt("test_event");
  evt.record(space);
  evt.fence();

  auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(h_data(i), i + 1);
  }
}

TEST(TEST_CATEGORY, event_is_complete) {
  using exec_space = TEST_EXECSPACE;

  exec_space space;

  Kokkos::Experimental::Event<exec_space> evt("test_event");
  evt.record(space);
  evt.fence();

  ASSERT_TRUE(evt.is_complete());
}

TEST(TEST_CATEGORY, event_space_depends_on) {
  using exec_space   = TEST_EXECSPACE;
  using memory_space = typename exec_space::memory_space;
  using view_type    = Kokkos::View<int*, memory_space>;
  using result_type  = Kokkos::View<int, memory_space>;

  exec_space space_a;
  exec_space space_b;
  constexpr int N = 10000;

  view_type data(
      Kokkos::view_alloc(space_a, "data", Kokkos::WithoutInitializing), N);

  Kokkos::parallel_for("produce",
                       Kokkos::RangePolicy<exec_space>(space_a, 0, N),
                       ProduceFunctor<view_type>{data});

  Kokkos::Experimental::Event<exec_space> evt("test_event");
  evt.record(space_a);
  Kokkos::Experimental::space_depends_on(space_b, evt);

  result_type result(Kokkos::view_alloc(space_b, "result"));

  Kokkos::parallel_for("consume",
                       Kokkos::RangePolicy<exec_space>(space_b, 0, N),
                       ConsumeFunctor<view_type, result_type>{data, result});

  space_b.fence();

  auto h_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

  int64_t expected = 0;
  for (int i = 0; i < N; ++i) expected += i * 2;
  ASSERT_EQ(h_result(), expected);
}

TEST(TEST_CATEGORY, event_move_semantics) {
  using exec_space = TEST_EXECSPACE;

  exec_space space;

  Kokkos::Experimental::Event<exec_space> evt1("test_event");
  evt1.record(space);

  Kokkos::Experimental::Event<exec_space> evt2(std::move(evt1));

  evt2.fence();
  ASSERT_TRUE(evt2.is_complete());
}

TEST(TEST_CATEGORY, event_copy_semantics) {
  using exec_space   = TEST_EXECSPACE;
  using memory_space = typename exec_space::memory_space;
  using view_type    = Kokkos::View<int*, memory_space>;

  exec_space space_a;
  exec_space space_b;
  constexpr int N = 1000;

  view_type data(
      Kokkos::view_alloc(space_a, "data", Kokkos::WithoutInitializing), N);

  Kokkos::parallel_for("fill", Kokkos::RangePolicy<exec_space>(space_a, 0, N),
                       FillFunctor<view_type>{data});

  Kokkos::Experimental::Event<exec_space> evt("test_event", space_a);
  auto evt_copy = evt;

  Kokkos::Experimental::space_depends_on(space_b, evt);
  evt_copy.fence();

  ASSERT_TRUE(evt_copy.is_complete());
  ASSERT_TRUE(evt.is_complete());

  auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(h_data(i), i + 1);
  }
}

}  // namespace Test
