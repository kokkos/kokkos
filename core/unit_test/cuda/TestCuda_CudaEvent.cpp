// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// CUDA-specific tests for Event<Kokkos::Cuda> that verify real
// cross-stream overlap via cudaEvent_t.  Portable correctness tests
// live in TestEvent.hpp; the tests here use wall-clock timing to
// confirm that events enable asynchronous overlap between distinct
// CUDA streams rather than acting as full fences.

#include <TestCuda_Category.hpp>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

namespace Test {

struct BusyFunctor {
  Kokkos::View<double*, Kokkos::CudaSpace> data;
  int iters;
  KOKKOS_FUNCTION void operator()(int i) const {
    double v = static_cast<double>(i);
    for (int k = 0; k < iters; ++k) v = Kokkos::sqrt(v * v + 1.0);
    data(i) = v;
  }
};

// Launch a long-running kernel on stream_a, record an event BEFORE
// the long work, then make stream_b wait on that event and launch
// its own long kernel.  If events enable real overlap the total wall
// time should be significantly less than running both sequentially.
//
// Timeline (overlapping):
//   stream_a: [short produce] | event | [long busy A ~~~~~~~~]
//   stream_b:                   wait -> [long busy B ~~~~~~~~]
//   wall:     |<----------- ~max(A,B) + short ------------>|
//
// Sequential (if event acted as full fence):
//   wall:     |<--- short + A + B --->|
TEST(TEST_CATEGORY, cuda_event_cross_stream_overlap) {
  // Sufficient per-thread work to make this test more robust
  // with a grid small enough that two kernels can run concurrently
  // instead of each filling the device and being serialized.
  const int N          = 8192;
  const int busy_iters = 120000;

  Kokkos::View<double*, Kokkos::CudaSpace> buf_a("buf_a", N);
  Kokkos::View<double*, Kokkos::CudaSpace> buf_b("buf_b", N);

  // Warm up the GPU so first-launch overhead doesn't skew timing.
  {
    Kokkos::Cuda warmup;
    Kokkos::parallel_for("warmup",
                         Kokkos::RangePolicy<Kokkos::Cuda>(warmup, 0, N),
                         BusyFunctor{buf_a, 1});
    warmup.fence();
  }

  // --- Measure sequential baseline: long_a then long_b on one stream ---
  double t_sequential;
  {
    Kokkos::Cuda seq;
    Kokkos::Timer timer;
    Kokkos::parallel_for("seq_a", Kokkos::RangePolicy<Kokkos::Cuda>(seq, 0, N),
                         BusyFunctor{buf_a, busy_iters});
    Kokkos::parallel_for("seq_b", Kokkos::RangePolicy<Kokkos::Cuda>(seq, 0, N),
                         BusyFunctor{buf_b, busy_iters});
    seq.fence();
    t_sequential = timer.seconds();
  }

  // --- Measure overlapped: event dependency between two streams ---
  double t_overlap;
  {
    // Default-constructed Kokkos::Cuda() refers to the same (default) stream;
    // use partition_space to obtain two independent stream instances.
    const auto [stream_a, stream_b] =
        Kokkos::Experimental::partition_space(Kokkos::Cuda(), 1, 1);

    Kokkos::Timer timer;
    // Short produce on stream_a, then record event.
    Kokkos::parallel_for("produce",
                         Kokkos::RangePolicy<Kokkos::Cuda>(stream_a, 0, N),
                         BusyFunctor{buf_a, 1});

    Kokkos::Experimental::Event<Kokkos::Cuda> evt(stream_a);

    // stream_a continues with long work after the event.
    Kokkos::parallel_for("long_a",
                         Kokkos::RangePolicy<Kokkos::Cuda>(stream_a, 0, N),
                         BusyFunctor{buf_a, busy_iters});

    // stream_b waits only for the event (not for long_a).
    Kokkos::Experimental::space_depends_on(stream_b, evt);

    Kokkos::parallel_for("long_b",
                         Kokkos::RangePolicy<Kokkos::Cuda>(stream_b, 0, N),
                         BusyFunctor{buf_b, busy_iters});

    stream_a.fence();
    stream_b.fence();
    t_overlap = timer.seconds();
  }

  // With real overlap the two long kernels run concurrently, so the
  // overlapped time should be well under the sequential time.  Use a
  // conservative 0.85 factor to check for asynchronous overlap.
  EXPECT_LT(t_overlap, t_sequential * 0.85)
      << "Expected cross-stream overlap.  Sequential: " << t_sequential
      << " s, Overlapped: " << t_overlap << " s";
}

// Verify that event.fence() blocks the host only until the recorded
// point, not until all later work on the stream completes.
//
// Timeline:
//   stream: [short kernel] | event | [long kernel ~~~~~~~]
//   host:    event.fence() returns here ^
//            stream.fence() returns here              ^
//
// If event.fence() were a full stream fence, t_event_wait ~ t_fence.
// With a real event, t_event_wait << t_fence.
TEST(TEST_CATEGORY, cuda_event_host_wait_is_not_stream_fence) {
  // Match cuda_event_cross_stream_overlap long-kernel cost so host timers
  // are not dominated by noise on very fast GPUs.
  const int N          = 8192;
  const int busy_iters = 120000;

  Kokkos::View<double*, Kokkos::CudaSpace> buf("buf", N);

  // Warm up.
  {
    Kokkos::Cuda warmup;
    Kokkos::parallel_for("warmup",
                         Kokkos::RangePolicy<Kokkos::Cuda>(warmup, 0, N),
                         BusyFunctor{buf, 1});
    warmup.fence();
  }

  Kokkos::Cuda stream;

  // Short kernel, then record event.
  Kokkos::parallel_for("short", Kokkos::RangePolicy<Kokkos::Cuda>(stream, 0, N),
                       BusyFunctor{buf, 1});

  Kokkos::Experimental::Event<Kokkos::Cuda> evt(stream);

  // Long kernel queued after the event.
  Kokkos::parallel_for("long", Kokkos::RangePolicy<Kokkos::Cuda>(stream, 0, N),
                       BusyFunctor{buf, busy_iters});

  // Host waits on event -- should return before the long kernel finishes.
  Kokkos::Timer timer_evt;
  evt.fence();
  double t_event_wait = timer_evt.seconds();

  // Now fence the whole stream to get total time.
  Kokkos::Timer timer_fence;
  stream.fence();
  double t_fence_after = timer_fence.seconds();

  // The event fence should complete well before the remaining stream work.
  // If the long kernel takes T, event.fence() should return ~0 while
  // stream.fence() still takes ~T.  We check that event.fence() took
  // less than half of the total (event_wait + remaining fence).
  double t_total = t_event_wait + t_fence_after;
  EXPECT_LT(t_event_wait, t_total * 0.5)
      << "event.fence() should return before the long kernel finishes.  "
      << "event.fence(): " << t_event_wait
      << " s, fence after: " << t_fence_after << " s";
}

}  // namespace Test
