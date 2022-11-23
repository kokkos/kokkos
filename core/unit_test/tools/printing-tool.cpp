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

#include <inttypes.h>
#include <iostream>

struct Kokkos_Profiling_KokkosPDeviceInfo;

// just get the basename for print_help/parse_args
std::string get_basename(char* cmd, int idx = 0) {
  if (idx > 0) return cmd;
  std::string _cmd = cmd;
  auto _pos        = _cmd.find_last_of('/');
  if (_pos != std::string::npos) return _cmd.substr(_pos + 1);
  return _cmd;
}

struct SpaceHandle {
  char name[64];
};

const int parallel_for_id    = 0;
const int parallel_reduce_id = 1;
const int parallel_scan_id   = 2;

extern "C" void kokkosp_init_library(
    const int /*loadSeq*/, const uint64_t /*interfaceVer*/,
    const uint32_t /*devInfoCount*/,
    Kokkos_Profiling_KokkosPDeviceInfo* /* deviceInfo */) {
  std::cout << "kokkosp_init_library::";
}

extern "C" void kokkosp_finalize_library() {
  std::cout << "kokkosp_finalize_library::";
}

extern "C" void kokkosp_print_help(char* exe) {
  std::cout << "kokkosp_print_help:" << get_basename(exe) << "::";
}

extern "C" void kokkosp_parse_args(int argc, char** argv) {
  std::cout << "kokkosp_parse_args:" << argc;
  for (int i = 0; i < argc; ++i) std::cout << ":" << get_basename(argv[i], i);
  std::cout << "::";
}

extern "C" void kokkosp_begin_parallel_for(const char* name,
                                           const uint32_t devID,
                                           uint64_t* kID) {
  *kID = parallel_for_id;
  std::cout << "kokkosp_begin_parallel_for:" << name << ":" << devID << ":"
            << *kID << "::";
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
  std::cout << "kokkosp_end_parallel_for:" << kID << "::";
}

extern "C" void kokkosp_begin_parallel_scan(const char* name,
                                            const uint32_t devID,
                                            uint64_t* kID) {
  *kID = parallel_scan_id;
  std::cout << "kokkosp_begin_parallel_scan:" << name << ":" << devID << ":"
            << *kID << "::";
}

extern "C" void kokkosp_end_parallel_scan(const uint64_t kID) {
  std::cout << "kokkosp_end_parallel_scan:" << kID << "::";
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name,
                                              const uint32_t devID,
                                              uint64_t* kID) {
  *kID = parallel_reduce_id;
  std::cout << "kokkosp_begin_parallel_reduce:" << name << ":" << devID << ":"
            << *kID << "::";
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
  std::cout << "kokkosp_end_parallel_reduce:" << kID << "::";
}

extern "C" void kokkosp_push_profile_region(const char* regionName) {
  std::cout << "kokkosp_push_profile_region:" << regionName << "::";
}

extern "C" void kokkosp_pop_profile_region() {
  std::cout << "kokkosp_pop_profile_region::";
}

extern "C" void kokkosp_allocate_data(SpaceHandle handle, const char* name,
                                      const void* ptr, uint64_t size) {
  std::cout << "kokkosp_allocate_data:" << handle.name << ":" << name << ":"
            << ptr << ":" << size << "::";
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        const void* ptr, uint64_t size) {
  std::cout << "kokkosp_deallocate_data:" << handle.name << ":" << name << ":"
            << ptr << ":" << size << "::";
}

extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char* dst_name,
                                        const void* dst_ptr,
                                        SpaceHandle src_handle,
                                        const char* src_name,
                                        const void* src_ptr, uint64_t size) {
  std::cout << "kokkosp_begin_deep_copy:" << dst_handle.name << ":" << dst_name
            << ":" << dst_ptr << ":" << src_handle.name << ":" << src_name
            << ":" << src_ptr << ":" << size << "::";
}

extern "C" void kokkosp_end_deep_copy() {
  std::cout << "kokkosp_end_deep_copy::";
}

uint32_t section_id = 3;
extern "C" void kokkosp_create_profile_section(const char* name,
                                               uint32_t* sec_id) {
  *sec_id = section_id;
  std::cout << "kokkosp_create_profile_section:" << name << ":" << *sec_id
            << "::";
}

extern "C" void kokkosp_start_profile_section(uint32_t sec_id) {
  std::cout << "kokkosp_start_profile_section:" << sec_id << "::";
}

extern "C" void kokkosp_stop_profile_section(uint32_t sec_id) {
  std::cout << "kokkosp_stop_profile_section:" << sec_id << "::";
}
extern "C" void kokkosp_destroy_profile_section(uint32_t sec_id) {
  std::cout << "kokkosp_destroy_profile_section:" << sec_id << "::";
}

extern "C" void kokkosp_profile_event(const char* name) {
  std::cout << "kokkosp_profile_event:" << name << "::";
}
extern "C" void kokkosp_declare_metadata(const char* key, const char* value) {
  std::cout << "kokkosp_declare_metadata:" << key << ":" << value << "::";
}
