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

#include "Kokkos_Core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

#define STREAM_ARRAY_SIZE 100000000
#define STREAM_NTIMES 20

#define HLINE "-------------------------------------------------------------\n"

using StreamDeviceArray =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using StreamHostArray = typename StreamDeviceArray::HostMirror;

using StreamIndex = int;
using Policy      = Kokkos::RangePolicy<Kokkos::IndexType<StreamIndex>>;

void perform_set(StreamDeviceArray& a, const double scalar) {
  Kokkos::parallel_for(
      "set", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a[i] = scalar; });

  Kokkos::fence();
}

void perform_copy(StreamDeviceArray& a, StreamDeviceArray& b) {
  Kokkos::parallel_for(
      "copy", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = a[i]; });

  Kokkos::fence();
}

void perform_scale(StreamDeviceArray& b, StreamDeviceArray& c,
                   const double scalar) {
  Kokkos::parallel_for(
      "scale", Policy(0, b.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = scalar * c[i]; });

  Kokkos::fence();
}

void perform_add(StreamDeviceArray& a, StreamDeviceArray& b,
                 StreamDeviceArray& c) {
  Kokkos::parallel_for(
      "add", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { c[i] = a[i] + b[i]; });

  Kokkos::fence();
}

void perform_triad(StreamDeviceArray& a, StreamDeviceArray& b,
                   StreamDeviceArray& c, const double scalar) {
  Kokkos::parallel_for(
      "triad", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a[i] = b[i] + scalar * c[i]; });

  Kokkos::fence();
}

int perform_validation(StreamHostArray& a, StreamHostArray& b,
                       StreamHostArray& c, const StreamIndex arraySize,
                       const double scalar) {
  double ai = 1.0;
  double bi = 2.0;
  double ci = 0.0;

  for (StreamIndex i = 0; i < STREAM_NTIMES; ++i) {
    ci = ai;
    bi = scalar * ci;
    ci = ai + bi;
    ai = bi + scalar * ci;
  };

  double aError = 0.0;
  double bError = 0.0;
  double cError = 0.0;

  for (StreamIndex i = 0; i < arraySize; ++i) {
    aError = std::abs(a[i] - ai);
    bError = std::abs(b[i] - bi);
    cError = std::abs(c[i] - ci);
  }

  double aAvgError = aError / (double)arraySize;
  double bAvgError = bError / (double)arraySize;
  double cAvgError = cError / (double)arraySize;

  const double epsilon = 1.0e-13;
  int errorCount       = 0;

  if (std::abs(aAvgError / ai) > epsilon) {
    fprintf(stderr, "Error: validation check on View a failed.\n");
    errorCount++;
  }

  if (std::abs(bAvgError / bi) > epsilon) {
    fprintf(stderr, "Error: validation check on View b failed.\n");
    errorCount++;
  }

  if (std::abs(cAvgError / ci) > epsilon) {
    fprintf(stderr, "Error: validation check on View c failed.\n");
    errorCount++;
  }

  if (errorCount == 0) {
    printf("All solutions checked and verified.\n");
  }

  return errorCount;
}

int run_benchmark() {
  printf("Reports fastest timing per kernel\n");
  printf("Creating Views...\n");

  printf("Memory Sizes:\n");
  printf("- Array Size:    %" PRIu64 "\n",
         static_cast<uint64_t>(STREAM_ARRAY_SIZE));
  printf("- Per Array:     %12.2f MB\n",
         1.0e-6 * (double)STREAM_ARRAY_SIZE * (double)sizeof(double));
  printf("- Total:         %12.2f MB\n",
         3.0e-6 * (double)STREAM_ARRAY_SIZE * (double)sizeof(double));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  StreamDeviceArray dev_a("a", STREAM_ARRAY_SIZE);
  StreamDeviceArray dev_b("b", STREAM_ARRAY_SIZE);
  StreamDeviceArray dev_c("c", STREAM_ARRAY_SIZE);

  StreamHostArray a = Kokkos::create_mirror_view(dev_a);
  StreamHostArray b = Kokkos::create_mirror_view(dev_b);
  StreamHostArray c = Kokkos::create_mirror_view(dev_c);

  const double scalar = 3.0;

  double setTime   = std::numeric_limits<double>::max();
  double copyTime  = std::numeric_limits<double>::max();
  double scaleTime = std::numeric_limits<double>::max();
  double addTime   = std::numeric_limits<double>::max();
  double triadTime = std::numeric_limits<double>::max();

  printf("Initializing Views...\n");

  Kokkos::parallel_for(
      "init",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                                                             STREAM_ARRAY_SIZE),
      KOKKOS_LAMBDA(const int i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
      });

  // Copy contents of a (from the host) to the dev_a (device)
  Kokkos::deep_copy(dev_a, a);
  Kokkos::deep_copy(dev_b, b);
  Kokkos::deep_copy(dev_c, c);

  printf("Starting benchmarking...\n");

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    perform_set(dev_c, 1.5);
    setTime = std::min(setTime, timer.seconds());

    timer.reset();
    perform_copy(dev_a, dev_c);
    copyTime = std::min(copyTime, timer.seconds());

    timer.reset();
    perform_scale(dev_b, dev_c, scalar);
    scaleTime = std::min(scaleTime, timer.seconds());

    timer.reset();
    perform_add(dev_a, dev_b, dev_c);
    addTime = std::min(addTime, timer.seconds());

    timer.reset();
    perform_triad(dev_a, dev_b, dev_c, scalar);
    triadTime = std::min(triadTime, timer.seconds());
  }

  Kokkos::deep_copy(a, dev_a);
  Kokkos::deep_copy(b, dev_b);
  Kokkos::deep_copy(c, dev_c);

  printf("Performing validation...\n");
  int rc = perform_validation(a, b, c, STREAM_ARRAY_SIZE, scalar);

  printf(HLINE);

  printf("Set             %11.2f MB/s\n",
         (1.0e-06 * 1.0 * (double)sizeof(double) * (double)STREAM_ARRAY_SIZE) /
             setTime);
  printf("Copy            %11.2f MB/s\n",
         (1.0e-06 * 2.0 * (double)sizeof(double) * (double)STREAM_ARRAY_SIZE) /
             copyTime);
  printf("Scale           %11.2f MB/s\n",
         (1.0e-06 * 2.0 * (double)sizeof(double) * (double)STREAM_ARRAY_SIZE) /
             scaleTime);
  printf("Add             %11.2f MB/s\n",
         (1.0e-06 * 3.0 * (double)sizeof(double) * (double)STREAM_ARRAY_SIZE) /
             addTime);
  printf("Triad           %11.2f MB/s\n",
         (1.0e-06 * 3.0 * (double)sizeof(double) * (double)STREAM_ARRAY_SIZE) /
             triadTime);

  printf(HLINE);

  return rc;
}

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  printf(HLINE);
  printf("Kokkos STREAM Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark();
  Kokkos::finalize();

  return rc;
}
