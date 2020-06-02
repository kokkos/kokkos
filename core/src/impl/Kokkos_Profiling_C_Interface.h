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

#ifndef KOKKOS_PROFILING_C_INTERFACE_HPP
#define KOKKOS_PROFILING_C_INTERFACE_HPP

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define KOKKOSP_INTERFACE_VERSION 20191080

struct Kokkos_Profiling_KokkosPDeviceInfo {
  size_t deviceID;
};

struct Kokkos_Profiling_SpaceHandle {
  char name[64];
};

typedef void (*Kokkos_Profiling_initFunction)(
    const int, const uint64_t, const uint32_t,
    Kokkos_Profiling_KokkosPDeviceInfo*);
typedef void (*Kokkos_Profiling_finalizeFunction)();
typedef void (*Kokkos_Profiling_beginFunction)(const char*, const uint32_t,
                                               uint64_t*);
typedef void (*Kokkos_Profiling_endFunction)(uint64_t);

typedef void (*Kokkos_Profiling_pushFunction)(const char*);
typedef void (*Kokkos_Profiling_popFunction)();

typedef void (*Kokkos_Profiling_allocateDataFunction)(
    const Kokkos_Profiling_SpaceHandle, const char*, const void*,
    const uint64_t);
typedef void (*Kokkos_Profiling_deallocateDataFunction)(
    const Kokkos_Profiling_SpaceHandle, const char*, const void*,
    const uint64_t);

typedef void (*Kokkos_Profiling_createProfileSectionFunction)(const char*,
                                                              uint32_t*);
typedef void (*Kokkos_Profiling_startProfileSectionFunction)(const uint32_t);
typedef void (*Kokkos_Profiling_stopProfileSectionFunction)(const uint32_t);
typedef void (*Kokkos_Profiling_destroyProfileSectionFunction)(const uint32_t);

typedef void (*Kokkos_Profiling_profileEventFunction)(const char*);

typedef void (*Kokkos_Profiling_beginDeepCopyFunction)(
    Kokkos_Profiling_SpaceHandle, const char*, const void*,
    Kokkos_Profiling_SpaceHandle, const char*, const void*, uint64_t);
typedef void (*Kokkos_Profiling_endDeepCopyFunction)();

struct Kokkos_Profiling_EventSet {
  Kokkos_Profiling_initFunction init;
  Kokkos_Profiling_finalizeFunction finalize;
  Kokkos_Profiling_beginFunction begin_parallel_for;
  Kokkos_Profiling_endFunction end_parallel_for;
  Kokkos_Profiling_beginFunction begin_parallel_reduce;
  Kokkos_Profiling_endFunction end_parallel_reduce;
  Kokkos_Profiling_beginFunction begin_parallel_scan;
  Kokkos_Profiling_endFunction end_parallel_scan;
  Kokkos_Profiling_pushFunction push_region;
  Kokkos_Profiling_popFunction pop_region;
  Kokkos_Profiling_allocateDataFunction allocate_data;
  Kokkos_Profiling_deallocateDataFunction deallocate_data;
  Kokkos_Profiling_createProfileSectionFunction create_profile_section;
  Kokkos_Profiling_startProfileSectionFunction start_profile_section;
  Kokkos_Profiling_stopProfileSectionFunction stop_profile_section;
  Kokkos_Profiling_destroyProfileSectionFunction destroy_profile_section;
  Kokkos_Profiling_profileEventFunction profile_event;
  Kokkos_Profiling_beginDeepCopyFunction begin_deep_copy;
  Kokkos_Profiling_endDeepCopyFunction end_deep_copy;
  char padding[2048];  // allows us to add another 256 events to the Tools
                       // interface without changing struct layout
};

#endif  // KOKKOS_PROFILING_C_INTERFACE_HPP
