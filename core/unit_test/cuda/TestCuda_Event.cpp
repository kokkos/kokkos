// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <TestCuda_Category.hpp>

#include <atomic>
#include <stdexcept>
#include <thread>

namespace {

struct ProduceEventGeneration {
  Kokkos::View<int**, Kokkos::CudaSpace> data;
  int generation;
  KOKKOS_FUNCTION void operator()(int i) const {
    data(generation, i) = i + generation + 1;
  }
};

struct ConsumeEventGeneration {
  Kokkos::View<int**, Kokkos::CudaSpace> data;
  Kokkos::View<int*, Kokkos::CudaSpace> result;
  int generation;
  KOKKOS_FUNCTION void operator()(int i) const {
    Kokkos::atomic_add(&result(generation), data(generation, i));
  }
};

TEST(TEST_CATEGORY, event_two_threads_acknowledge_wait_before_rerecord) {
  auto [producer_space, consumer_space] =
      Kokkos::Experimental::partition_space(Kokkos::Cuda(), 1, 1);
  ASSERT_NE(producer_space.cuda_stream(), consumer_space.cuda_stream());
  constexpr int rounds = 32;
  constexpr int count  = 1024;
  // Separate rows preserve each generation's data independently of event reuse.
  Kokkos::View<int**, Kokkos::CudaSpace> data(
      Kokkos::view_alloc(producer_space, Kokkos::WithoutInitializing,
                         "event generations"),
      rounds, count);
  Kokkos::View<int*, Kokkos::CudaSpace> result(
      Kokkos::view_alloc(consumer_space, "event generation results"), rounds);
  Kokkos::Experimental::Event<Kokkos::Cuda> event("generation ready");
  auto event_copy = event;
  std::atomic<int> recorded{0};
  std::atomic<int> wait_submitted{0};

  std::thread producer([&] {
    for (int generation = 0; generation < rounds; ++generation) {
      while (wait_submitted.load(std::memory_order_acquire) != generation)
        std::this_thread::yield();
      Kokkos::parallel_for("produce generation",
                           Kokkos::RangePolicy(producer_space, 0, count),
                           ProduceEventGeneration{data, generation});
      event.record(producer_space);
      recorded.store(generation + 1, std::memory_order_release);
    }
  });
  std::thread consumer([&] {
    for (int generation = 0; generation < rounds; ++generation) {
      while (recorded.load(std::memory_order_acquire) != generation + 1)
        std::this_thread::yield();
      Kokkos::Experimental::space_depends_on(consumer_space, event_copy);
      // Acknowledge wait submission before the producer re-records the event
      // so each wait captures its intended generation.
      wait_submitted.store(generation + 1, std::memory_order_release);
      Kokkos::parallel_for("consume generation",
                           Kokkos::RangePolicy(consumer_space, 0, count),
                           ConsumeEventGeneration{data, result, generation});
    }
  });
  producer.join();
  consumer.join();
  // The only explicit completion fence is after all records/waits are
  // submitted.
  consumer_space.fence();
  auto host_result =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  for (int generation = 0; generation < rounds; ++generation) {
    EXPECT_EQ(host_result(generation),
              count * (count - 1) / 2 + count * (generation + 1))
        << "generation " << generation;
  }
}

TEST(TEST_CATEGORY, event_rerecord_independent_instances) {
  auto [space_a, space_b] =
      Kokkos::Experimental::partition_space(Kokkos::Cuda(), 1, 1);
  ASSERT_NE(space_a, space_b);
  Kokkos::Experimental::Event<Kokkos::Cuda> event("re-recorded event");
  Kokkos::Experimental::Event<Kokkos::Cuda> event_a("first instance");
  Kokkos::Experimental::Event<Kokkos::Cuda> event_b("second instance");

  // Block only A's queue. The callback accesses ordinary host memory, so this
  // gate does not depend on host/device visibility of a pinned-memory flag.
  std::atomic<bool> released{false};
  const auto wait_for_release = [](void* state) {
    auto& released = *static_cast<std::atomic<bool>*>(state);
    while (!released.load(std::memory_order_acquire)) std::this_thread::yield();
  };
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaLaunchHostFunc(space_a.cuda_stream(), wait_for_release, &released));

  try {
    event.record(space_a);
    event_a.record(space_a);
    event.record(space_b);
    event_b.record(space_b);
    space_b.fence();

    EXPECT_FALSE(event_a.is_complete());
    EXPECT_TRUE(event_b.is_complete());
    // CUDA replaces the captured state, so fencing E needs only B to finish.
    const bool complete = event.is_complete();
    EXPECT_TRUE(complete);
    // Do not deadlock the test if a regression retains the blocked recording.
    if (complete) event.fence();
    EXPECT_FALSE(event_a.is_complete());
    // The counter fallback instead retains both outstanding recordings and
    // cannot report completion until A also finishes.
  } catch (...) {
    // Release and drain A before the callback's flag goes out of scope.
    released.store(true, std::memory_order_release);
    space_a.fence();
    throw;
  }

  released.store(true, std::memory_order_release);
  space_a.fence();
  event.fence();
  EXPECT_TRUE(event.is_complete());
  EXPECT_TRUE(event_a.is_complete());
}

struct RestoreCudaDevice {
  int device;
  RestoreCudaDevice() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDevice(&device)); }
  ~RestoreCudaDevice() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(device)); }
};

struct ProduceEventData {
  Kokkos::View<int*, Kokkos::CudaSpace> source;
  KOKKOS_FUNCTION void operator()(int i) const { source(i) = i + 1; }
};

TEST(TEST_CATEGORY, event_cross_device_dependency) {
  int num_devices = 0;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) GTEST_SKIP() << "Requires two CUDA devices";

  RestoreCudaDevice restore_device;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(0));
  cudaStream_t producer_stream = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
  const Kokkos::Cuda producer(producer_stream, Kokkos::Impl::ManageStream::yes);

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(1));
  cudaStream_t consumer_stream = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(
      cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking));
  const Kokkos::Cuda consumer(consumer_stream, Kokkos::Impl::ManageStream::yes);
  ASSERT_NE(producer.cuda_device(), consumer.cuda_device());

  constexpr int count = 100000;
  using device_view   = Kokkos::View<int*, Kokkos::CudaSpace>;
  using host_view     = Kokkos::View<int*, Kokkos::CudaHostPinnedSpace>;
  device_view source(
      Kokkos::view_alloc(producer, Kokkos::WithoutInitializing, "source"),
      count);
  device_view destination(
      Kokkos::view_alloc(consumer, Kokkos::WithoutInitializing, "destination"),
      count);
  host_view staging("staging", count);
  host_view result("result", count);
  Kokkos::Experimental::Event<Kokkos::Cuda> event("producer ready", producer);

  // An event can be waited on from another device, but it must be recorded
  // on the device where it was created. Contracts check this precondition;
  // when contracts are disabled, CUDA still rejects the invalid recording.
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(consumer.cuda_device()));
  bool checked_preconditions = false;
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  KOKKOS_EXPECTS(checked_preconditions = true);
  if (checked_preconditions) {
    EXPECT_DEATH(event.record(consumer),
                 "Expected precondition `.*` evaluated false.");
  } else {
    EXPECT_THROW(event.record(consumer), std::runtime_error);
  }

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(producer.cuda_device()));
  Kokkos::parallel_for("produce", Kokkos::RangePolicy(producer, 0, count),
                       ProduceEventData{source});
  Kokkos::deep_copy(producer, staging, source);
  // Recording selects the owning instance's device, even if another device
  // is current on this host thread.
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(consumer.cuda_device()));
  event.record(producer);

  // Only the cross-device event dependency orders these transfers. There is
  // no producer fence, peer-memory access, or default-stream synchronization.
  // The dependency selects the consumer device without caller-side setup.
  Kokkos::Experimental::space_depends_on(consumer, event);
  Kokkos::deep_copy(consumer, destination, staging);
  Kokkos::deep_copy(consumer, result, destination);
  consumer.fence();

  for (int i = 0; i < count; ++i) EXPECT_EQ(result(i), i + 1);
  EXPECT_TRUE(event.is_complete());
}

}  // namespace
