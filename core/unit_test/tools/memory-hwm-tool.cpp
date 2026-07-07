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
#if defined(KOKKOS_ENABLE_DEBUG)
  printf("Memory tracker initialized. \n");
#endif
}

struct SpaceHandle {
  char name[64];
};

char space_name[16][64];
int num_spaces;
std::vector<std::tuple<uint64_t, uint64_t> > space_size_track[16];
uint64_t space_size[16];
#define WARNING_THRESHOLD 4ULL * 1024 * 1024 * 1024
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

  int space_i = num_spaces;
  for (int s = 0; s < num_spaces; ++s)
    if (strcmp(space_name[s], handle.name) == 0) space_i = s;

  if (space_i == num_spaces) {
    strncpy(space_name[num_spaces], handle.name, 64);
    num_spaces++;
  }
  space_size[space_i] += size;
  total_allocated += size;

  if (total_allocated > WARNING_THRESHOLD) {
    fprintf(
        stderr,
        "\n [ WARNING! ] Total allocation (%.4f GB) exceeds %.2f GB limit!\n",
        total_allocated / (1024.0 * 1024.0 * 1024.0),
        WARNING_THRESHOLD / (1024.0 * 1024.0 * 1024.0));
    exit(1);
  }

  space_size_track[space_i].push_back(
      std::make_tuple(space_size[space_i], max_mem_usage()));
#if defined(KOKKOS_ENABLE_DEBUG)
  printf("Allocated %" PRIu64 " kB\n ",
         std::get<1>(space_size_track[space_i].back()));
#endif
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        const void* ptr, uint64_t size) {
  std::lock_guard<std::mutex> lock(m);

  int space_i = num_spaces;
  for (int s = 0; s < num_spaces; s++)
    if (strcmp(space_name[s], handle.name) == 0) space_i = s;

  if (space_i == num_spaces) {
    strncpy(space_name[num_spaces], handle.name, 64);
    num_spaces++;
  }
  if (space_size[space_i] >= size) {
    space_size[space_i] -= size;
    total_allocated -= size;
    space_size_track[space_i].push_back(
        std::make_tuple(space_size[space_i], max_mem_usage()));
#if defined(KOKKOS_ENABLE_DEBUG)
    printf("De-allocated %" PRIu64 " kB\n ",
           std::get<1>(space_size_track[space_i].back()));
#endif
  }
}

extern "C" void kokkosp_finalize_library() {
#if defined(KOKKOS_ENABLE_DEBUG)
  printf("\nKokkosP: Finalization of profiling library.\n");

  struct rusage sys_resources;
  getrusage(RUSAGE_SELF, &sys_resources);

  printf("KokkosP: High water mark memory consumption: %" PRIu64 " kB\n\n",
         (uint64_t)sys_resources.ru_maxrss * RU_MAXRSS_UNITS);
#endif
}
