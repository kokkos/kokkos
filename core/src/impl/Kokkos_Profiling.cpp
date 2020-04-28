/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Profiling.hpp>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <dlfcn.h>
#endif
#include <vector>
#include <array>
#include <stack>
#include <chrono>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
namespace Kokkos {
namespace Profiling {

static initFunction initProfileLibrary         = nullptr;
static finalizeFunction finalizeProfileLibrary = nullptr;

static beginFunction beginForCallback    = nullptr;
static beginFunction beginScanCallback   = nullptr;
static beginFunction beginReduceCallback = nullptr;
static endFunction endForCallback        = nullptr;
static endFunction endScanCallback       = nullptr;
static endFunction endReduceCallback     = nullptr;

static pushFunction pushRegionCallback = nullptr;
static popFunction popRegionCallback   = nullptr;

static allocateDataFunction allocateDataCallback     = nullptr;
static deallocateDataFunction deallocateDataCallback = nullptr;

static beginDeepCopyFunction beginDeepCopyCallback = nullptr;
static endDeepCopyFunction endDeepCopyCallback     = nullptr;

static createProfileSectionFunction createSectionCallback   = nullptr;
static startProfileSectionFunction startSectionCallback     = nullptr;
static stopProfileSectionFunction stopSectionCallback       = nullptr;
static destroyProfileSectionFunction destroySectionCallback = nullptr;

static profileEventFunction profileEventCallback = nullptr;
}  // end namespace Profiling

namespace Profiling {

bool profileLibraryLoaded() { return (initProfileLibrary != nullptr); }

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
  if (beginForCallback != nullptr) {
    Kokkos::fence();
    (*beginForCallback)(kernelPrefix.c_str(), devID, kernelID);
  }
}

void endParallelFor(const uint64_t kernelID) {
  if (endForCallback != nullptr) {
    Kokkos::fence();
    (*endForCallback)(kernelID);
  }
}

void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
  if (beginScanCallback != nullptr) {
    Kokkos::fence();
    (*beginScanCallback)(kernelPrefix.c_str(), devID, kernelID);
  }
}

void endParallelScan(const uint64_t kernelID) {
  if (endScanCallback != nullptr) {
    Kokkos::fence();
    (*endScanCallback)(kernelID);
  }
}

void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
  if (beginReduceCallback != nullptr) {
    Kokkos::fence();
    (*beginReduceCallback)(kernelPrefix.c_str(), devID, kernelID);
  }
}

void endParallelReduce(const uint64_t kernelID) {
  if (endReduceCallback != nullptr) {
    Kokkos::fence();
    (*endReduceCallback)(kernelID);
  }
}

void pushRegion(const std::string& kName) {
  if (pushRegionCallback != nullptr) {
    Kokkos::fence();
    (*pushRegionCallback)(kName.c_str());
  }
}

void popRegion() {
  if (popRegionCallback != nullptr) {
    Kokkos::fence();
    (*popRegionCallback)();
  }
}

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size) {
  if (allocateDataCallback != nullptr) {
    (*allocateDataCallback)(space, label.c_str(), ptr, size);
  }
}

void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
  if (deallocateDataCallback != nullptr) {
    (*deallocateDataCallback)(space, label.c_str(), ptr, size);
  }
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  if (beginDeepCopyCallback != nullptr) {
    (*beginDeepCopyCallback)(dst_space, dst_label.c_str(), dst_ptr, src_space,
                             src_label.c_str(), src_ptr, size);
  }
}

void endDeepCopy() {
  if (endDeepCopyCallback != nullptr) {
    (*endDeepCopyCallback)();
  }
}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
  if (createSectionCallback != nullptr) {
    (*createSectionCallback)(sectionName.c_str(), secID);
  }
}

void startSection(const uint32_t secID) {
  if (startSectionCallback != nullptr) {
    (*startSectionCallback)(secID);
  }
}

void stopSection(const uint32_t secID) {
  if (stopSectionCallback != nullptr) {
    (*stopSectionCallback)(secID);
  }
}

void destroyProfileSection(const uint32_t secID) {
  if (destroySectionCallback != nullptr) {
    (*destroySectionCallback)(secID);
  }
}

void markEvent(const std::string& eventName) {
  if (profileEventCallback != nullptr) {
    (*profileEventCallback)(eventName.c_str());
  }
}

SpaceHandle make_space_handle(const char* space_name) {
  SpaceHandle handle;
  strncpy(handle.name, space_name, 63);
  return handle;
}

}  // end namespace Profiling

namespace Profiling {

void initialize() {
  // Make sure initialize calls happens only once
  static int is_initialized = 0;
  if (is_initialized) return;
  is_initialized = 1;

  void* firstProfileLibrary;

  char* envProfileLibrary = getenv("KOKKOS_PROFILE_LIBRARY");

  // If we do not find a profiling library in the environment then exit
  // early.
  if (envProfileLibrary == nullptr) {
    return;
  }

  char* envProfileCopy =
      (char*)malloc(sizeof(char) * (strlen(envProfileLibrary) + 1));
  sprintf(envProfileCopy, "%s", envProfileLibrary);

  char* profileLibraryName = strtok(envProfileCopy, ";");

  if ((profileLibraryName != nullptr) &&
      (strcmp(profileLibraryName, "") != 0)) {
    firstProfileLibrary = dlopen(profileLibraryName, RTLD_NOW | RTLD_GLOBAL);

    if (firstProfileLibrary == nullptr) {
      std::cerr << "Error: Unable to load KokkosP library: "
                << profileLibraryName << std::endl;
      std::cerr << "dlopen(" << profileLibraryName
                << ", RTLD_NOW | RTLD_GLOBAL) failed with " << dlerror()
                << '\n';
    } else {
#ifdef KOKKOS_ENABLE_PROFILING_LOAD_PRINT
      std::cout << "KokkosP: Library Loaded: " << profileLibraryName
                << std::endl;
#endif

      // dlsym returns a pointer to an object, while we want to assign to
      // pointer to function A direct cast will give warnings hence, we have to
      // workaround the issue by casting pointer to pointers.
      auto p1 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_for");
      set_begin_parallel_for_callback(*reinterpret_cast<beginFunction*>(&p1));
      auto p2 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_scan");
      set_begin_parallel_scan_callback(*reinterpret_cast<beginFunction*>(&p2));
      auto p3 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_reduce");
      set_begin_parallel_reduce_callback(
          *reinterpret_cast<beginFunction*>(&p3));

      auto p4 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_scan");
      set_end_parallel_scan_callback(*reinterpret_cast<endFunction*>(&p4));
      auto p5 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_for");
      set_end_parallel_for_callback(*reinterpret_cast<endFunction*>(&p5));
      auto p6 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_reduce");
      set_end_parallel_reduce_callback(*reinterpret_cast<endFunction*>(&p6));

      auto p7 = dlsym(firstProfileLibrary, "kokkosp_init_library");
      set_init_callback(*reinterpret_cast<initFunction*>(&p7));
      auto p8 = dlsym(firstProfileLibrary, "kokkosp_finalize_library");
      set_finalize_callback(*reinterpret_cast<finalizeFunction*>(&p8));

      auto p9 = dlsym(firstProfileLibrary, "kokkosp_push_profile_region");
      set_push_region_callback(*reinterpret_cast<pushFunction*>(&p9));
      auto p10 = dlsym(firstProfileLibrary, "kokkosp_pop_profile_region");
      set_pop_region_callback(*reinterpret_cast<popFunction*>(&p10));

      auto p11 = dlsym(firstProfileLibrary, "kokkosp_allocate_data");
      set_allocate_data_callback(
          *reinterpret_cast<allocateDataFunction*>(&p11));
      auto p12 = dlsym(firstProfileLibrary, "kokkosp_deallocate_data");
      set_deallocate_data_callback(
          *reinterpret_cast<deallocateDataFunction*>(&p12));

      auto p13 = dlsym(firstProfileLibrary, "kokkosp_begin_deep_copy");
      set_begin_deep_copy_callback(
          *reinterpret_cast<beginDeepCopyFunction*>(&p13));
      auto p14 = dlsym(firstProfileLibrary, "kokkosp_end_deep_copy");
      set_end_deep_copy_callback(*reinterpret_cast<endDeepCopyFunction*>(&p14));

      auto p15 = dlsym(firstProfileLibrary, "kokkosp_create_profile_section");
      set_create_profile_section_callback(
          *(reinterpret_cast<createProfileSectionFunction*>(&p15)));
      auto p16 = dlsym(firstProfileLibrary, "kokkosp_start_profile_section");
      set_start_profile_section_callback(
          *reinterpret_cast<startProfileSectionFunction*>(&p16));
      auto p17 = dlsym(firstProfileLibrary, "kokkosp_stop_profile_section");
      set_stop_profile_section_callback(
          *reinterpret_cast<stopProfileSectionFunction*>(&p17));
      auto p18 = dlsym(firstProfileLibrary, "kokkosp_destroy_profile_section");
      set_destroy_profile_section_callback(
          *(reinterpret_cast<destroyProfileSectionFunction*>(&p18)));

      auto p19 = dlsym(firstProfileLibrary, "kokkosp_profile_event");
      set_profile_event_callback(
          *reinterpret_cast<profileEventFunction*>(&p19));
    }
  }

  if (initProfileLibrary != nullptr) {
    (*initProfileLibrary)(0, (uint64_t)KOKKOSP_INTERFACE_VERSION, (uint32_t)0,
                          nullptr);
  }

  free(envProfileCopy);
}

void finalize() {
  // Make sure finalize calls happens only once
  static int is_finalized = 0;
  if (is_finalized) return;
  is_finalized = 1;

  if (finalizeProfileLibrary != nullptr) {
    (*finalizeProfileLibrary)();

    // Set all profile hooks to nullptr to prevent
    // any additional calls. Once we are told to
    // finalize, we mean it
    initProfileLibrary     = nullptr;
    finalizeProfileLibrary = nullptr;

    beginForCallback    = nullptr;
    beginScanCallback   = nullptr;
    beginReduceCallback = nullptr;
    endScanCallback     = nullptr;
    endForCallback      = nullptr;
    endReduceCallback   = nullptr;

    pushRegionCallback = nullptr;
    popRegionCallback  = nullptr;

    allocateDataCallback   = nullptr;
    deallocateDataCallback = nullptr;

    beginDeepCopyCallback = nullptr;
    endDeepCopyCallback   = nullptr;

    createSectionCallback  = nullptr;
    startSectionCallback   = nullptr;
    stopSectionCallback    = nullptr;
    destroySectionCallback = nullptr;

    profileEventCallback = nullptr;
    // TODO DZP: move to its own section
  }
}
static EventSet current_callbacks;
void set_init_callback(initFunction callback) {
  initProfileLibrary     = callback;
  current_callbacks.init = callback;
}
void set_finalize_callback(finalizeFunction callback) {
  finalizeProfileLibrary     = callback;
  current_callbacks.finalize = callback;
}
void set_begin_parallel_for_callback(beginFunction callback) {
  beginForCallback                     = callback;
  current_callbacks.begin_parallel_for = callback;
}
void set_end_parallel_for_callback(endFunction callback) {
  endForCallback                     = callback;
  current_callbacks.end_parallel_for = callback;
}
void set_begin_parallel_reduce_callback(beginFunction callback) {
  beginReduceCallback                     = callback;
  current_callbacks.begin_parallel_reduce = callback;
}
void set_end_parallel_reduce_callback(endFunction callback) {
  endReduceCallback                     = callback;
  current_callbacks.end_parallel_reduce = callback;
}
void set_begin_parallel_scan_callback(beginFunction callback) {
  beginScanCallback                     = callback;
  current_callbacks.begin_parallel_scan = callback;
}
void set_end_parallel_scan_callback(endFunction callback) {
  endScanCallback                     = callback;
  current_callbacks.end_parallel_scan = callback;
}
void set_push_region_callback(pushFunction callback) {
  pushRegionCallback            = callback;
  current_callbacks.push_region = callback;
}
void set_pop_region_callback(popFunction callback) {
  popRegionCallback            = callback;
  current_callbacks.pop_region = callback;
}
void set_allocate_data_callback(allocateDataFunction callback) {
  allocateDataCallback            = callback;
  current_callbacks.allocate_data = callback;
}
void set_deallocate_data_callback(deallocateDataFunction callback) {
  deallocateDataCallback            = callback;
  current_callbacks.deallocate_data = callback;
}
void set_create_profile_section_callback(
    createProfileSectionFunction callback) {
  createSectionCallback                    = callback;
  current_callbacks.create_profile_section = callback;
}
void set_start_profile_section_callback(startProfileSectionFunction callback) {
  startSectionCallback                    = callback;
  current_callbacks.start_profile_section = callback;
}
void set_stop_profile_section_callback(stopProfileSectionFunction callback) {
  stopSectionCallback                    = callback;
  current_callbacks.stop_profile_section = callback;
}
void set_destroy_profile_section_callback(
    destroyProfileSectionFunction callback) {
  destroySectionCallback                    = callback;
  current_callbacks.destroy_profile_section = callback;
}
void set_profile_event_callback(profileEventFunction callback) {
  profileEventCallback            = callback;
  current_callbacks.profile_event = callback;
}
void set_begin_deep_copy_callback(beginDeepCopyFunction callback) {
  beginDeepCopyCallback             = callback;
  current_callbacks.begin_deep_copy = callback;
}
void set_end_deep_copy_callback(endDeepCopyFunction callback) {
  endDeepCopyCallback             = callback;
  current_callbacks.end_deep_copy = callback;
}

void pause_tools() {
  initProfileLibrary     = nullptr;
  finalizeProfileLibrary = nullptr;

  beginForCallback    = nullptr;
  beginScanCallback   = nullptr;
  beginReduceCallback = nullptr;
  endScanCallback     = nullptr;
  endForCallback      = nullptr;
  endReduceCallback   = nullptr;

  pushRegionCallback = nullptr;
  popRegionCallback  = nullptr;

  allocateDataCallback   = nullptr;
  deallocateDataCallback = nullptr;

  beginDeepCopyCallback = nullptr;
  endDeepCopyCallback   = nullptr;

  createSectionCallback  = nullptr;
  startSectionCallback   = nullptr;
  stopSectionCallback    = nullptr;
  destroySectionCallback = nullptr;

  profileEventCallback = nullptr;
}

void resume_tools() {
  initProfileLibrary     = current_callbacks.init;
  finalizeProfileLibrary = current_callbacks.finalize;

  beginForCallback    = current_callbacks.begin_parallel_for;
  beginScanCallback   = current_callbacks.begin_parallel_scan;
  beginReduceCallback = current_callbacks.begin_parallel_reduce;
  endScanCallback     = current_callbacks.end_parallel_scan;
  endForCallback      = current_callbacks.end_parallel_for;
  endReduceCallback   = current_callbacks.end_parallel_reduce;

  pushRegionCallback     = current_callbacks.push_region;
  popRegionCallback      = current_callbacks.pop_region;
  allocateDataCallback   = current_callbacks.allocate_data;
  deallocateDataCallback = current_callbacks.deallocate_data;

  beginDeepCopyCallback = current_callbacks.begin_deep_copy;
  endDeepCopyCallback   = current_callbacks.end_deep_copy;

  createSectionCallback  = current_callbacks.create_profile_section;
  startSectionCallback   = current_callbacks.start_profile_section;
  stopSectionCallback    = current_callbacks.stop_profile_section;
  destroySectionCallback = current_callbacks.destroy_profile_section;

  profileEventCallback = current_callbacks.profile_event;
}
}  // namespace Profiling

}  // namespace Kokkos

#else

// TODO DZP, handle the off case

#include <impl/Kokkos_Profiling_Interface.hpp>
#include <cstring>

namespace Kokkos {
namespace Profiling {

bool profileLibraryLoaded() { return false; }

void beginParallelFor(const std::string&, const uint32_t, uint64_t*) {}
void endParallelFor(const uint64_t) {}
void beginParallelScan(const std::string&, const uint32_t, uint64_t*) {}
void endParallelScan(const uint64_t) {}
void beginParallelReduce(const std::string&, const uint32_t, uint64_t*) {}
void endParallelReduce(const uint64_t) {}

void pushRegion(const std::string&) {}
void popRegion() {}
void createProfileSection(const std::string&, uint32_t*) {}
void startSection(const uint32_t) {}
void stopSection(const uint32_t) {}
void destroyProfileSection(const uint32_t) {}

void markEvent(const std::string&) {}

void allocateData(const SpaceHandle, const std::string, const void*,
                  const uint64_t) {}
void deallocateData(const SpaceHandle, const std::string, const void*,
                    const uint64_t) {}

void beginDeepCopy(const SpaceHandle, const std::string, const void*,
                   const SpaceHandle, const std::string, const void*,
                   const uint64_t) {}
void endDeepCopy() {}

void initialize() {}
void finalize() {}

static EventSet current_callbacks;

void set_init_callback(initFunction callback) {}
void set_finalize_callback(finalizeFunction callback) {}
void set_begin_parallel_for_callback(beginFunction callback) {}
void set_end_parallel_for_callback(endFunction callback) {}
void set_begin_parallel_reduce_callback(beginFunction callback) {}
void set_end_parallel_reduce_callback(endFunction callback) {}
void set_begin_parallel_scan_callback(beginFunction callback) {}
void set_end_parallel_scan_callback(endFunction callback) {}
void set_push_region_callback(pushFunction callback) {}
void set_pop_region_callback(popFunction callback) {}
void set_allocate_data_callback(allocateDataFunction callback) {}
void set_deallocate_data_callback(deallocateDataFunction callback) {}
void set_create_profile_section_callback(
    createProfileSectionFunction callback) {}
void set_start_profile_section_callback(startProfileSectionFunction callback) {}
void set_stop_profile_section_callback(stopProfileSectionFunction callback) {}
void set_destroy_profile_section_callback(
    destroyProfileSectionFunction callback) {}
void set_profile_event_callback(profileEventFunction callback) {}
void set_begin_deep_copy_callback(beginDeepCopyFunction callback) {}
void set_end_deep_copy_callback(endDeepCopyFunction callback) {}

void pause_tools() {}

void resume_tools() {}

}  // namespace Profiling
}  // namespace Kokkos

#endif
