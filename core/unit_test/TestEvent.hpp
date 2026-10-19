// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <concepts>
#include <thread>

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

template <class ViewType>
struct EventUpdateFunctor {
  ViewType data;
  int increment;
  KOKKOS_FUNCTION void operator()(int i) const { data(i) += increment; }
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
  using result_type  = Kokkos::View<int64_t, memory_space>;

  auto [space_a, space_b] =
      Kokkos::Experimental::partition_space(exec_space(), 1, 1);
  constexpr int N      = 1000000;
  constexpr int rounds = 32;

  view_type data(
      Kokkos::view_alloc(space_a, "data", Kokkos::WithoutInitializing), N);
  result_type result(Kokkos::view_alloc(space_b, "result"));
  Kokkos::Experimental::Event<exec_space> evt("test_event");

  Kokkos::parallel_for("produce",
                       Kokkos::RangePolicy<exec_space>(space_a, 0, N),
                       ProduceFunctor<view_type>{data});

  evt.record(space_a);
  // Each update and completion marker follows the previous one on A.
  for (int generation = 1; generation <= rounds; ++generation) {
    Kokkos::parallel_for("update", Kokkos::RangePolicy(space_a, 0, N),
                         EventUpdateFunctor<view_type>{data, generation});
    evt.record(space_a);
  }
  auto evt_copy = evt;
  Kokkos::Experimental::space_depends_on(space_b, evt_copy);

  Kokkos::parallel_for("consume",
                       Kokkos::RangePolicy<exec_space>(space_b, 0, N),
                       ConsumeFunctor<view_type, result_type>{data, result});

  space_b.fence();

  auto h_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

  int64_t expected = 0;
  for (int i = 0; i < N; ++i) expected += static_cast<int64_t>(i) * 2;
  constexpr int increment = rounds * (rounds + 1) / 2;
  expected += static_cast<int64_t>(N) * increment;
  ASSERT_EQ(h_result(), expected);
  EXPECT_TRUE(evt_copy.is_complete());
}

TEST(TEST_CATEGORY, event_move_semantics) {
  using exec_space = TEST_EXECSPACE;

  exec_space space;

  Kokkos::Experimental::Event<exec_space> evt1("test_event");
  evt1.record(space);

  Kokkos::Experimental::Event<exec_space> evt2(std::move(evt1));

  evt2.fence();
  ASSERT_TRUE(evt2.is_complete());

  // Copy assignment restores a moved-from handle.
  evt1 = evt2;
  evt1.record(space);
  evt1.fence();
  EXPECT_TRUE(evt1.is_complete());
  EXPECT_EQ(evt1.label(), "test_event");

  // Move assignment transfers the handle again.
  evt2 = std::move(evt1);
  evt2.fence();
  EXPECT_TRUE(evt2.is_complete());
}

TEST(TEST_CATEGORY, event_copy_semantics) {
  using exec_space   = TEST_EXECSPACE;
  using memory_space = typename exec_space::memory_space;
  using view_type    = Kokkos::View<int*, memory_space>;

  auto [space_a, space_b] =
      Kokkos::Experimental::partition_space(exec_space(), 1, 1);
  constexpr int N = 1000;

  view_type data(
      Kokkos::view_alloc(space_a, "data", Kokkos::WithoutInitializing), N);

  Kokkos::parallel_for("fill", Kokkos::RangePolicy<exec_space>(space_a, 0, N),
                       FillFunctor<view_type>{data});

  Kokkos::Experimental::Event<exec_space> evt("test_event", space_a);
  auto evt_copy = evt;

  Kokkos::Experimental::space_depends_on(space_b, evt_copy);
  evt_copy.fence();

  ASSERT_TRUE(evt_copy.is_complete());
  ASSERT_TRUE(evt.is_complete());

  auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(h_data(i), i + 1);
  }
}

template <class ViewType>
struct ThreadEventFillFunctor {
  ViewType data;
  int generation;
  KOKKOS_FUNCTION void operator()(int i) const { data(i) = i + generation; }
};

TEST(TEST_CATEGORY, event_two_threads_query_while_recording) {
  using exec_space     = TEST_EXECSPACE;
  using view_type      = Kokkos::View<int*, typename exec_space::memory_space>;
  constexpr int count  = 1024;
  constexpr int rounds = 256;
  auto [space]         = Kokkos::Experimental::partition_space(exec_space(), 1);
  view_type data(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                    "query while recording data"),
                 count);
  Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> result(
      "query while recording result", count);
  Kokkos::Experimental::Event<exec_space> event("query while recording");
  auto event_copy = event;
  std::barrier start(2);
  std::atomic<bool> polling_started{false};
  std::atomic<bool> final_record_submitted{false};
  int queries         = 0;
  bool final_complete = false;

  std::thread recorder([&] {
    start.arrive_and_wait();
    while (!polling_started.load(std::memory_order_acquire))
      std::this_thread::yield();
    for (int generation = 1; generation <= rounds; ++generation) {
      Kokkos::parallel_for("produce queried generation",
                           Kokkos::RangePolicy(space, 0, count),
                           ThreadEventFillFunctor<view_type>{data, generation});
      event.record(space);
    }
    Kokkos::deep_copy(space, result, data);
    event.record(space);
    final_record_submitted.store(true, std::memory_order_release);
  });
  std::thread poller([&] {
    start.arrive_and_wait();
    (void)event_copy.is_complete();
    ++queries;
    polling_started.store(true, std::memory_order_release);
    do {
      // Either result is allowed during overlap. Completion need not remain
      // true when the event is re-recorded for newer work.
      (void)event_copy.is_complete();
      ++queries;
      std::this_thread::yield();
    } while (!final_record_submitted.load(std::memory_order_acquire));
    event_copy.fence();
    final_complete = event_copy.is_complete();
  });
  recorder.join();
  poller.join();
  EXPECT_GE(queries, 2);
  EXPECT_TRUE(final_complete);
  EXPECT_TRUE(event.is_complete());
  for (int i = 0; i < count; ++i) EXPECT_EQ(result(i), i + rounds);
}

// Exercise the portable fallback with genuinely overlapping operations. Serial
// instances are independent, so only the shared Event resource needs
// protection.
#if TEST_CATEGORY_NUMBER == 0
TEST(TEST_CATEGORY, event_two_threads_record_and_dependency_overlap) {
  auto [producer_a, producer_b, consumer_space] =
      Kokkos::Experimental::partition_space(TEST_EXECSPACE(), 1, 1, 1);
  ASSERT_NE(producer_a, producer_b);
  ASSERT_NE(producer_a, consumer_space);
  ASSERT_NE(producer_b, consumer_space);

  const auto check_overlap = [&](const TEST_EXECSPACE& waiter_space) {
    Kokkos::Experimental::Event<TEST_EXECSPACE> event("overlap", producer_a);
    auto event_copy = event;
    std::barrier handoff(2);
    constexpr int repetitions = 20000;

    std::thread recorder([&] {
      // Deterministic phases cover comparison against both recorded instances.
      for (int phase = 0; phase < 2; ++phase) {
        event.record(phase == 0 ? producer_a : producer_b);
        handoff.arrive_and_wait();
        handoff.arrive_and_wait();
      }
      handoff.arrive_and_wait();
      for (int i = 0; i < repetitions; ++i)
        event.record(i % 2 ? producer_a : producer_b);
    });
    std::thread waiter([&] {
      for (int phase = 0; phase < 2; ++phase) {
        handoff.arrive_and_wait();
        Kokkos::Experimental::space_depends_on(waiter_space, event_copy);
        handoff.arrive_and_wait();
      }
      handoff.arrive_and_wait();
      // Contend on the shared resource while its recorded instance changes.
      for (int i = 0; i < repetitions; ++i)
        Kokkos::Experimental::space_depends_on(waiter_space, event_copy);
    });
    recorder.join();
    waiter.join();
    event.fence();
    EXPECT_TRUE(event.is_complete());
    EXPECT_TRUE(event_copy.is_complete());
  };

  // A exercises matching and different-instance branches; C always differs.
  check_overlap(producer_a);
  check_overlap(consumer_space);
}

#endif

#if (defined(KOKKOS_ENABLE_SERIAL) && defined(KOKKOS_ENABLE_OPENMP) && \
     TEST_CATEGORY_NUMBER == 0) ||                                     \
    (defined(KOKKOS_ENABLE_SERIAL) && defined(KOKKOS_ENABLE_CUDA) &&   \
     TEST_CATEGORY_NUMBER == 5)
template <class ProducerExec, class ConsumerExec>
void test_cross_backend_event_dependency() {
  constexpr int count = 1024;
  ProducerExec producer_space;
  ConsumerExec consumer_space;
  using view_type = Kokkos::View<int*, Kokkos::SharedHostPinnedSpace>;
  view_type data(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                    "cross backend event data"),
                 count);
  Kokkos::Experimental::Event<ProducerExec> event("cross backend event");
  auto event_copy = event;
  std::atomic<bool> recorded{false};

  std::thread producer([&] {
    Kokkos::parallel_for("cross backend produce",
                         Kokkos::RangePolicy(producer_space, 0, count),
                         FillFunctor<view_type>{data});
    event.record(producer_space);
    recorded.store(true, std::memory_order_release);
  });
  std::thread consumer([&] {
    while (!recorded.load(std::memory_order_acquire)) std::this_thread::yield();
    Kokkos::Experimental::space_depends_on(consumer_space, event_copy);
    Kokkos::parallel_for("cross backend consume",
                         Kokkos::RangePolicy(consumer_space, 0, count),
                         EventUpdateFunctor<view_type>{data, 10});
    consumer_space.fence();
  });
  producer.join();
  consumer.join();
  EXPECT_TRUE(event.is_complete());
  for (int i = 0; i < count; ++i) EXPECT_EQ(data(i), i + 11);
}
#endif

#if defined(KOKKOS_ENABLE_SERIAL) && defined(KOKKOS_ENABLE_OPENMP) && \
    TEST_CATEGORY_NUMBER == 0
TEST(TEST_CATEGORY, event_openmp_serial_dependencies) {
  test_cross_backend_event_dependency<Kokkos::OpenMP, Kokkos::Serial>();
  test_cross_backend_event_dependency<Kokkos::Serial, Kokkos::OpenMP>();
}
#endif

#if defined(KOKKOS_ENABLE_SERIAL) && defined(KOKKOS_ENABLE_CUDA) && \
    TEST_CATEGORY_NUMBER == 5
TEST(TEST_CATEGORY, event_cuda_serial_dependencies) {
  test_cross_backend_event_dependency<Kokkos::Cuda, Kokkos::Serial>();
  test_cross_backend_event_dependency<Kokkos::Serial, Kokkos::Cuda>();
}
#endif

}  // namespace Test
