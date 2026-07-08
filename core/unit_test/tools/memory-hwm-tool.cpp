// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <tuple>
#include <cstring>
#include <inttypes.h>

#include <unistd.h>

#include <sys/resource.h>

// darwin reports rusage.ru_maxrss in bytes
#if defined(__APPLE__) || defined(__MACH__)
#define RU_MAXRSS_UNITS 1024
#else
#define RU_MAXRSS_UNITS 1
#endif

struct Kokkos_Profiling_KokkosPDeviceInfo;

extern "C" void kokkosp_init_library(
    const int loadSeq, const uint64_t interfaceVer,
    const uint32_t /*devInfoCount*/,
    Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/) {
  (void)interfaceVer;
  (void)loadSeq;
#ifdef KOKKOS_ENABLE_DEBUG
  printf("Memory tracker initialized. \n");
#endif
}

struct SpaceHandle {
  char name[64];
};

constexpr uint64_t WARNING_THRESHOLD = 4ULL * 1024 * 1024 * 1024;
static std::mutex m;
static uint64_t total_allocated = 0;

uint64_t max_mem_usage() {
  struct rusage app_info;
  getrusage(RUSAGE_SELF, &app_info);
  const long max_rssKB = app_info.ru_maxrss;
  return max_rssKB * RU_MAXRSS_UNITS;
}

extern "C" void kokkosp_allocate_data(const SpaceHandle handle,
                                      const char* name, const void* const ptr,
                                      uint64_t size) {
  std::lock_guard<std::mutex> lock(m);
  bool allocation_flag = false;

  if (strcmp(handle.name, "Host") == 0) {
    total_allocated += size;
    allocation_flag = true;
  }

  (void)ptr;
  (void)name;

  if (total_allocated > WARNING_THRESHOLD) {
    fprintf(
        stderr,
        "\n [ WARNING! ] Total allocation (%.4f GB) exceeds %.2f GB limit!\n",
        total_allocated / (1024.0 * 1024.0 * 1024.0),
        WARNING_THRESHOLD / (1024.0 * 1024.0 * 1024.0));
    exit(1);
  }

#ifdef KOKKOS_ENABLE_DEBUG
  if (allocation_flag)
    printf("Allocated %" PRIu64 " kB at %s\n ", max_mem_usage(), handle.name);
#endif
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        const void* ptr, uint64_t size) {
  std::lock_guard<std::mutex> lock(m);
  bool allocation_flag = false;

  (void)ptr;
  (void)name;

  if (strcmp(handle.name, "Host") == 0) {
    total_allocated -= size;
    allocation_flag = true;
  }
#ifdef KOKKOS_ENABLE_DEBUG
  if (allocation_flag)
    printf("De-allocated %" PRIu64 " kB at %s\n ", max_mem_usage(),
           handle.name);
#endif
}

extern "C" void kokkosp_finalize_library() {
#ifdef KOKKOS_ENABLE_DEBUG
  printf("\nKokkosP: Finalization of profiling library.\n");

  printf("KokkosP: High water mark memory consumption: %" PRIu64 " kB\n\n",
         max_mem_usage());
#endif
}
