// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include "Kokkos_Core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>

static size_t STREAM_ARRAY_SIZE = 100000000;
#define STREAM_NTIMES 20

#define HLINE "-------------------------------------------------------------\n"

using StreamHostArray = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Restrict>>;
using StreamIndex = int;
using Policy      = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::IndexType<StreamIndex>>;

void perform_set(StreamHostArray& a, const double scalar) {
  Kokkos::parallel_for(
      "set", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { a[i] = scalar; });

  Kokkos::fence();
}

void perform_copy(StreamHostArray& a, StreamHostArray& b) {
  Kokkos::parallel_for(
      "copy", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = a[i]; });

  Kokkos::fence();
}

void perform_scale(StreamHostArray& b, StreamHostArray& c,
                   const double scalar) {
  Kokkos::parallel_for(
      "scale", Policy(0, b.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { b[i] = scalar * c[i]; });

  Kokkos::fence();
}

void perform_add(StreamHostArray& a, StreamHostArray& b,
                 StreamHostArray& c) {
  Kokkos::parallel_for(
      "add", Policy(0, a.extent(0)),
      KOKKOS_LAMBDA(const StreamIndex i) { c[i] = a[i] + b[i]; });

  Kokkos::fence();
}

void perform_triad(StreamHostArray& a, StreamHostArray& b,
                   StreamHostArray& c, const double scalar) {
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

template <typename T,
          std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>, int> = 0>
static T get_env(const char* name, T defaultValue = -1) {
  T value = defaultValue;
  const char *str = std::getenv(name);
  if (nullptr != str) {
    value = static_cast<T>(std::stoi(str));
  }
  return value;
}

int run_benchmark() {
  const char* array_size = std::getenv("STREAM_ARRAY_SIZE");
  if (array_size != nullptr) {
    STREAM_ARRAY_SIZE = std::stoull(array_size);
  }

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

#ifndef KOKKOS_ENABLE_HWLOC
  StreamHostArray a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "a"), STREAM_ARRAY_SIZE);
  StreamHostArray b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), STREAM_ARRAY_SIZE);
  StreamHostArray c(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c"), STREAM_ARRAY_SIZE);
#else
  StreamHostArray a;
  StreamHostArray b;
  StreamHostArray c;

  printf("- Possible NUMA settings by env vars: \n");
  printf("   - <A|B|C>_NUMA_START=<int> <A|B|C>_NUMA_STEP=<int> : based on indexed list of node IDs\n");
  // hwloc_memattr_id_e was introduced in version 2.3.0 (0x00020300)
#if HWLOC_API_VERSION >= 0x00020300
  printf("   - <A|B|C>_NUMA_MEMATTR=<bandwidth|latency> : based on hwloc memory attribute\n");
#endif // HWLOC_API_VERSION
  printf("   - NUMA_POLICY=<default|firsttouch|bind|interleave|nexttouch> : membind policy\n");
  printf("   - NUMA_AWARE=<int> (default = 1) : if zero then disable all NUMA-aware binding (i.e. normal behavior)\n");

  const int numa_aware = get_env<int>("NUMA_AWARE", 1);
  printf("   - env(NUMA_AWARE) = %d (%s)\n", numa_aware, (numa_aware ? "true" : "false"));

  if (!numa_aware) {
    a = StreamHostArray(Kokkos::view_alloc(Kokkos::WithoutInitializing, "a"), STREAM_ARRAY_SIZE);
    b = StreamHostArray(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"), STREAM_ARRAY_SIZE);
    c = StreamHostArray(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c"), STREAM_ARRAY_SIZE);
  } else {
    // example usage of customized hostspace with NUMA capability
    auto my_views = std::array{
      StreamHostArray(),
      StreamHostArray(),
      StreamHostArray()
    };

    #define NUMA_NODES_MAX 8
    static Kokkos::NUMANodes<NUMA_NODES_MAX> numa_ids;

    // if memattr is supported then use memattr by default if NUMA_START not set
    constexpr int numa_start_default =
      (HWLOC_API_VERSION >= 0x00020300) ? -1 : 0;

    const int numa_starts[3] = {
      get_env<int>("A_NUMA_START", numa_start_default),
      get_env<int>("B_NUMA_START", numa_start_default),
      get_env<int>("C_NUMA_START", numa_start_default)
    };
    const int numa_steps[3]  = {
      get_env<int>("A_NUMA_STEP", 1),
      get_env<int>("B_NUMA_STEP", 1),
      get_env<int>("C_NUMA_STEP", 1),
    };
    const char* numa_memattrs[3]  = {
      std::getenv("A_NUMA_MEMATTR"),
      std::getenv("B_NUMA_MEMATTR"),
      std::getenv("C_NUMA_MEMATTR"),
    };
    const char* numa_policy_str = std::getenv("NUMA_POLICY");
    hwloc_membind_policy_t numa_policy = HWLOC_MEMBIND_DEFAULT;
    if (numa_policy_str != nullptr) {
      if (std::strstr(numa_policy_str, "default") || std::strstr(numa_policy_str, "DEFAULT")) {
        numa_policy = HWLOC_MEMBIND_DEFAULT;
      }
      if (std::strstr(numa_policy_str, "firsttouch") || std::strstr(numa_policy_str, "FIRSTTOUCH")) {
        numa_policy = HWLOC_MEMBIND_FIRSTTOUCH;
      }
      if (std::strstr(numa_policy_str, "bind") || std::strstr(numa_policy_str, "BIND")) {
        numa_policy = HWLOC_MEMBIND_BIND;
      }
      if (std::strstr(numa_policy_str, "interleave") || std::strstr(numa_policy_str, "INTERLEAVE")) {
        numa_policy = HWLOC_MEMBIND_INTERLEAVE;
      }
      if (std::strstr(numa_policy_str, "nexttouch") || std::strstr(numa_policy_str, "NEXTTOUCH")) {
        numa_policy = HWLOC_MEMBIND_NEXTTOUCH;
      }
    }

    for (int i = 0; i < 3; ++i) {
      const char* numa_memattr  = numa_memattrs[i];
      int numa_start = numa_starts[i];
      int numa_step  = numa_steps[i];
      char label[8];
      snprintf(label, sizeof(label), "array%d", i);

      if (numa_start >= 0) {
        printf("- Bound NUMA nodes for %s: ", label);
        for (int j = 0; j < NUMA_NODES_MAX; ++j) {
          numa_ids[j] = numa_start + j*numa_step;
          printf("  %2u%s", numa_ids[j], (j == (NUMA_NODES_MAX-1) ? "\n" : ""));
        }
        auto hspace  = Kokkos::HostSpace(numa_ids, numa_policy);
        auto prop = Kokkos::view_alloc(label, hspace, Kokkos::WithoutInitializing);
        my_views[i] = StreamHostArray(prop, STREAM_ARRAY_SIZE);
      }

#if HWLOC_API_VERSION >= 0x00020300
      else {
        hwloc_memattr_id_e memattr = HWLOC_MEMATTR_ID_BANDWIDTH;

        if (numa_memattr != nullptr) {
          if (std::strstr(numa_memattr, "bandwidth") || std::strstr(numa_memattr, "BANDWIDTH")) {
            memattr = HWLOC_MEMATTR_ID_BANDWIDTH;
          }
          if (std::strstr(numa_memattr, "latency") || std::strstr(numa_memattr, "LATENCY")) {
            memattr = HWLOC_MEMATTR_ID_LATENCY;
          }
        }
        const char *memattr_str = memattr == HWLOC_MEMATTR_ID_BANDWIDTH ? "BANDWIDTH" :
                                 (memattr == HWLOC_MEMATTR_ID_LATENCY ?  "LATENCY" :
                                                  "UNKNOWN");

        printf("- Bound NUMA nodes for %s: based on memattr = %s\n", label, memattr_str);
        auto hspace = Kokkos::HostSpace(memattr, numa_policy);
        auto prop = Kokkos::view_alloc(label, hspace, Kokkos::WithoutInitializing);
        my_views[i] = StreamHostArray(prop, STREAM_ARRAY_SIZE);
      }
#endif // HWLOC_API_VERSION

    } // for

    a = my_views[0];
    b = my_views[1];
    c = my_views[2];
  } // if (!numa_aware)

  printf(HLINE);
#endif // KOKKOS_ENABLE_HWLOC

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

  printf("Starting benchmarking...\n");

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    perform_set(c, 1.5);
    setTime = std::min(setTime, timer.seconds());

    timer.reset();
    perform_copy(a, c);
    copyTime = std::min(copyTime, timer.seconds());

    timer.reset();
    perform_scale(b, c, scalar);
    scaleTime = std::min(scaleTime, timer.seconds());

    timer.reset();
    perform_add(a, b, c);
    addTime = std::min(addTime, timer.seconds());

    timer.reset();
    perform_triad(a, b, c, scalar);
    triadTime = std::min(triadTime, timer.seconds());
  }

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
  printf("Kokkos STREAM CPU Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  const int rc = run_benchmark();
  Kokkos::finalize();

  return rc;
}
