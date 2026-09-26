// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cstdint>
#include <cstdio>
#include <mutex>

#include <cstring>
#include <unordered_map>

#include <unistd.h>

struct SpaceHandle {
  char name[64];
};

constexpr uint64_t WARNING_THRESHOLD = 4ULL * 1024 * 1024 * 1024;
static uint64_t total_allocated      = 0;
static std::unordered_map<const void*, uint64_t> host_allocations;
static std::mutex m;

struct Kokkos_Profiling_KokkosPDeviceInfo;

extern "C" void kokkosp_init_library(
    const int loadSeq, const uint64_t interfaceVer,
    const uint32_t /*devInfoCount*/,
    Kokkos_Profiling_KokkosPDeviceInfo* /*deviceInfo*/) {
  (void)interfaceVer;
  (void)loadSeq;

  total_allocated = 0;
  host_allocations.clear();
}

extern "C" void kokkosp_allocate_data(const SpaceHandle handle,
                                      const char* name, const void* const ptr,
                                      uint64_t size) {
  bool exceeded           = false;
  uint64_t reported_total = 0;

  {
    std::lock_guard<std::mutex> lock(m);

    if (strcmp(handle.name, "Host") == 0) {
      auto it = host_allocations.find(ptr);
      if (it != host_allocations.end()) {
        total_allocated -= it->second;
        it->second = size;
      } else {
        host_allocations.emplace(ptr, size);
      }

      total_allocated += size;
      reported_total = total_allocated;
    }

    exceeded = reported_total > WARNING_THRESHOLD;
  }

  if (exceeded) {
    fprintf(
        stderr,
        "\n [ WARNING! ] Total allocation (%.4f GB) exceeds %.2f GB limit!\n",
        reported_total / (1024.0 * 1024.0 * 1024.0),
        WARNING_THRESHOLD / (1024.0 * 1024.0 * 1024.0));
    // using static destructor causes crash, so we use _exit
    fflush(stderr);
    _exit(1);
  }

  (void)name;
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        const void* ptr, uint64_t size) {
  (void)name;
  (void)size;

  std::lock_guard<std::mutex> lock(m);

  if (strcmp(handle.name, "Host") == 0) {
    auto it = host_allocations.find(ptr);
    if (it != host_allocations.end()) {
      total_allocated -= it->second;
      host_allocations.erase(it);
    }
  }
}
