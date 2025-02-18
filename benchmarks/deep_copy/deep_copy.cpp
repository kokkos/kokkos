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

#define COPY_ARRAY_SIZE 100000000
#define COPY_NTIMES 20

#define HLINE "-------------------------------------------------------------\n"

using DeviceView =
    Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using HostView = typename DeviceView::HostMirror;

using StridedDeviceView = Kokkos::View<double*, Kokkos::LayoutStride>;
using StridedHostView   = typename StridedDeviceView::HostMirror;

int run_benchmark() {
  printf("Reports fastest timing per kernel\n");
  printf("Creating Views...\n");

  printf("Memory Sizes:\n");
  printf("- View Size:    %" PRIu64 "\n",
         static_cast<uint64_t>(COPY_ARRAY_SIZE));
  printf("- Per View:     %12.2f MB\n",
         1.0e-6 * (double)COPY_ARRAY_SIZE * (double)sizeof(double));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         COPY_NTIMES);
  printf("Note: the strided case copies half the view.\n");

  printf(HLINE);

  DeviceView dev_a("a", COPY_ARRAY_SIZE);
  DeviceView dev_b("b", COPY_ARRAY_SIZE);

  HostView host_a = Kokkos::create_mirror_view(dev_a);
  HostView host_b = Kokkos::create_mirror_view(dev_b);

  double scalarToDeviceTime = std::numeric_limits<double>::max();
  double scalarToHostTime   = std::numeric_limits<double>::max();
  double hostToDeviceTime   = std::numeric_limits<double>::max();
  double deviceToHostTime   = std::numeric_limits<double>::max();
  double hostToHostTime     = std::numeric_limits<double>::max();
  double deviceToDeviceTime = std::numeric_limits<double>::max();

  printf("Start benchmarking for contiguous layout...\n");

  Kokkos::Timer timer;

  for (int k = 0; k < COPY_NTIMES; ++k) {
    timer.reset();
    Kokkos::deep_copy(dev_a, 1.0);
    scalarToDeviceTime = std::min(scalarToDeviceTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_a, 2.0);
    scalarToHostTime = std::min(scalarToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(dev_a, host_a);
    hostToDeviceTime = std::min(hostToDeviceTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_a, dev_a);
    deviceToHostTime = std::min(deviceToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_b, host_a);
    hostToHostTime = std::min(hostToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(dev_b, dev_a);
    deviceToDeviceTime = std::min(deviceToDeviceTime, timer.seconds());
  }

  printf(HLINE);

  printf("Scalar to Device             %11.2f ms\n",
         1.0e3 * scalarToDeviceTime);
  printf("Scalar to Host               %11.2f ms\n", 1.0e3 * scalarToHostTime);
  printf("Host to Device               %11.2f ms\n", 1.0e3 * hostToDeviceTime);
  printf("Device to Host               %11.2f ms\n", 1.0e3 * deviceToHostTime);
  printf("Host to Host                 %11.2f ms\n", 1.0e3 * hostToHostTime);
  printf("Device to Device             %11.2f ms\n",
         1.0e3 * deviceToDeviceTime);

  printf(HLINE);

  Kokkos::LayoutStride layout(COPY_ARRAY_SIZE / 2, 2);

  StridedDeviceView dev_odd(dev_a.data(), layout);
  StridedDeviceView dev_even(dev_a.data() + 1, layout);

  StridedHostView host_odd(host_a.data(), layout);
  StridedHostView host_even(host_a.data() + 1, layout);

  printf("Start benchmarking for strided layout...\n");

  for (int k = 0; k < COPY_NTIMES; ++k) {
    timer.reset();
    Kokkos::deep_copy(dev_odd, 3.0);
    scalarToDeviceTime = std::min(scalarToDeviceTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_odd, 3.0);
    scalarToHostTime = std::min(scalarToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(dev_odd, host_odd);
    hostToDeviceTime = std::min(hostToDeviceTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_odd, dev_odd);
    deviceToHostTime = std::min(deviceToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(host_even, host_odd);
    hostToHostTime = std::min(hostToHostTime, timer.seconds());

    timer.reset();
    Kokkos::deep_copy(dev_even, dev_odd);
    deviceToDeviceTime = std::min(deviceToDeviceTime, timer.seconds());
  }

  printf(HLINE);

  printf("Scalar to Device             %11.2f ms\n",
         1.0e3 * scalarToDeviceTime);
  printf("Scalar to Host               %11.2f ms\n", 1.0e3 * scalarToHostTime);
  printf("Host to Device               %11.2f ms\n", 1.0e3 * hostToDeviceTime);
  printf("Device to Host               %11.2f ms\n", 1.0e3 * deviceToHostTime);
  printf("Host to Host                 %11.2f ms\n", 1.0e3 * hostToHostTime);
  printf("Device to Device             %11.2f ms\n",
         1.0e3 * deviceToDeviceTime);

  printf(HLINE);

  return 0;
}

int main(int argc, char* argv[]) {
  printf(HLINE);
  printf("Kokkos deep_copy Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark();
  Kokkos::finalize();

  return rc;
}
