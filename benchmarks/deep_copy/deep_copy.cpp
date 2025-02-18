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

using Scalar = double;

using DeviceView =
    Kokkos::View<Scalar*, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using HostView = typename DeviceView::HostMirror;

using StridedDeviceView = Kokkos::View<Scalar*, Kokkos::LayoutStride>;
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

  double scalarToDeviceTput =
      COPY_ARRAY_SIZE * sizeof(Scalar) / scalarToDeviceTime;
  double scalarToHostTput = COPY_ARRAY_SIZE * sizeof(Scalar) / scalarToHostTime;
  double hostToDeviceTput = COPY_ARRAY_SIZE * sizeof(Scalar) / hostToDeviceTime;
  double deviceToHostTput = COPY_ARRAY_SIZE * sizeof(Scalar) / deviceToHostTime;
  double hostToHostTput   = COPY_ARRAY_SIZE * sizeof(Scalar) / hostToHostTime;
  double deviceToDeviceTput =
      COPY_ARRAY_SIZE * sizeof(Scalar) / deviceToDeviceTime;

  printf(HLINE);

  printf("Scalar to Device     %13.2f ms %17.2f MiB/s\n",
         1.0e3 * scalarToDeviceTime, scalarToDeviceTput / 1024 / 1024);
  printf("Scalar to Host       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * scalarToHostTime, scalarToHostTput / 1024 / 1024);
  printf("Host to Device       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * hostToDeviceTime, hostToDeviceTput / 1024 / 1024);
  printf("Device to Host       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * deviceToHostTime, deviceToHostTput / 1024 / 1024);
  printf("Host to Host         %13.2f ms %17.2f MiB/s\n",
         1.0e3 * hostToHostTime, hostToHostTput / 1024 / 1024);
  printf("Device to Device     %13.2f ms %17.2f MiB/s\n",
         1.0e3 * deviceToDeviceTime, deviceToDeviceTput / 1024 / 1024);

  printf(HLINE);

  Kokkos::LayoutStride layout(COPY_ARRAY_SIZE / 2, 2);

  StridedDeviceView dev_odd(dev_a.data(), layout);
  StridedDeviceView dev_even(dev_a.data() + 1, layout);

  StridedHostView host_odd(host_a.data(), layout);
  StridedHostView host_even(host_a.data() + 1, layout);

  scalarToDeviceTime = std::numeric_limits<double>::max();
  scalarToHostTime   = std::numeric_limits<double>::max();
  hostToDeviceTime   = std::numeric_limits<double>::max();
  deviceToHostTime   = std::numeric_limits<double>::max();
  hostToHostTime     = std::numeric_limits<double>::max();
  deviceToDeviceTime = std::numeric_limits<double>::max();

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

  scalarToDeviceTput =
      COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / scalarToDeviceTime;
  scalarToHostTput = COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / scalarToHostTime;
  hostToDeviceTput = COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / hostToDeviceTime;
  deviceToHostTput = COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / deviceToHostTime;
  hostToHostTput   = COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / hostToHostTime;
  deviceToDeviceTput =
      COPY_ARRAY_SIZE * sizeof(Scalar) / 2 / deviceToDeviceTime;

  printf(HLINE);

  printf("Scalar to Device     %13.2f ms %17.2f MiB/s\n",
         1.0e3 * scalarToDeviceTime, scalarToDeviceTput / 1024 / 1024);
  printf("Scalar to Host       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * scalarToHostTime, scalarToHostTput / 1024 / 1024);
  printf("Host to Device       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * hostToDeviceTime, hostToDeviceTput / 1024 / 1024);
  printf("Device to Host       %13.2f ms %17.2f MiB/s\n",
         1.0e3 * deviceToHostTime, deviceToHostTput / 1024 / 1024);
  printf("Host to Host         %13.2f ms %17.2f MiB/s\n",
         1.0e3 * hostToHostTime, hostToHostTput / 1024 / 1024);
  printf("Device to Device     %13.2f ms %17.2f MiB/s\n",
         1.0e3 * deviceToDeviceTime, deviceToDeviceTput / 1024 / 1024);

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
