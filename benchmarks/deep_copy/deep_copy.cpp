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

#define COPY_ARRAY_SIZE 100000000
#define COPY_NTIMES 20

#define HLINE "-------------------------------------------------------------\n"

using Scalar = double;

using DeviceView =
    Kokkos::View<Scalar *, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using HostView = typename DeviceView::HostMirror;

using StridedDeviceView = Kokkos::View<Scalar *, Kokkos::LayoutStride>;
using StridedHostView   = typename StridedDeviceView::HostMirror;

struct Result {
  double time;  // s
  double tput;  // MiB/s
};

template <typename Dst, typename Src>
static Result bench(const Dst &dst, const Src &src) {
  Kokkos::Timer timer;
  Result r;
  r.time = std::numeric_limits<double>::max();
  for (int k = 0; k < COPY_NTIMES; ++k) {
    timer.reset();
    Kokkos::deep_copy(dst, src);
    r.time = std::min(r.time, timer.seconds());
  }

  static_assert(Dst::rank() == 1,
                "Expected rank-1 view as deep_copy destination");
  r.tput =
      dst.extent(0) * sizeof(typename Dst::value_type) / r.time / 1024 / 1024;

  return r;
}

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

  {
    DeviceView dev_a("a", COPY_ARRAY_SIZE);
    DeviceView dev_b("b", COPY_ARRAY_SIZE);

    HostView host_a = Kokkos::create_mirror_view(dev_a);
    HostView host_b = Kokkos::create_mirror_view(dev_b);

    printf("Start benchmarking for contiguous layout...\n");

    Result scalarToDevice = bench(dev_a, 1.0);
    Result scalarToHost   = bench(host_a, 2.0);
    Result hostToDevice   = bench(dev_a, host_a);
    Result deviceToHost   = bench(host_a, dev_a);
    Result hostToHost     = bench(host_b, host_a);
    Result deviceToDevice = bench(dev_b, dev_a);

    printf(HLINE);

    printf("Scalar to Device     %13.2f ms %17.2f MiB/s\n",
           1.0e3 * scalarToDevice.time, scalarToDevice.tput);
    printf("Scalar to Host       %13.2f ms %17.2f MiB/s\n",
           1.0e3 * scalarToHost.time, scalarToHost.tput);
    printf("Host to Device       %13.2f ms %17.2f MiB/s\n",
           1.0e3 * hostToDevice.time, hostToDevice.tput);
    printf("Device to Host       %13.2f ms %17.2f MiB/s\n",
           1.0e3 * deviceToHost.time, deviceToHost.tput);
    printf("Host to Host         %13.2f ms %17.2f MiB/s\n",
           1.0e3 * hostToHost.time, hostToHost.tput);
    printf("Device to Device     %13.2f ms %17.2f MiB/s\n",
           1.0e3 * deviceToDevice.time, deviceToDevice.tput);

    printf(HLINE);
  }

  {
    Kokkos::LayoutStride layout(COPY_ARRAY_SIZE / 2, 2);

    DeviceView dev_a("a", COPY_ARRAY_SIZE);
    DeviceView dev_b("b", COPY_ARRAY_SIZE);

    StridedDeviceView dev_odd(dev_a.data(), layout);
    StridedDeviceView dev_even(dev_a.data() + 1, layout);

    StridedHostView host_odd  = Kokkos::create_mirror_view(dev_odd);
    StridedHostView host_even = Kokkos::create_mirror_view(dev_even);

    printf("Start benchmarking for strided layout...\n");

    Result scalarToDevice = bench(dev_odd, 3.0);
    Result scalarToHost   = bench(host_odd, 3.0);
    Result hostToHost     = bench(host_even, host_odd);
    Result deviceToDevice = bench(dev_even, dev_odd);

    printf(HLINE);

    printf("Scalar to Device     %13.2f ms %17.2f MiB/s\n",
           1.0e3 * scalarToDevice.time, scalarToDevice.tput);
    printf("Scalar to Host       %13.2f ms %17.2f MiB/s\n",
           1.0e3 * scalarToHost.time, scalarToHost.tput);
    printf("Host to Host         %13.2f ms %17.2f MiB/s\n",
           1.0e3 * hostToHost.time, hostToHost.tput);
    printf("Device to Device     %13.2f ms %17.2f MiB/s\n",
           1.0e3 * deviceToDevice.time, deviceToDevice.tput);

    printf(HLINE);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos deep_copy Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark();
  Kokkos::finalize();

  return rc;
}
