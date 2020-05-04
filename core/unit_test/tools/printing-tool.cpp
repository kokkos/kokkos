
#include <cstdio>
#include <inttypes.h>
#include <vector>
#include <string>

struct SpaceHandle {
  char name[64];
};

constexpr const int parallel_for_id    = 0;
constexpr const int parallel_reduce_id = 1;
constexpr const int parallel_scan_id   = 2;

extern "C" void kokkosp_init_library(const int loadSeq,
                                     const uint64_t interfaceVer,
                                     const uint32_t devInfoCount,
                                     void* deviceInfo) {
  printf("kokkosp_init_library::");
}

extern "C" void kokkosp_finalize_library() {
  printf("kokkosp_finalize_library::");
}

extern "C" void kokkosp_begin_parallel_for(const char* name,
                                           const uint32_t devID,
                                           uint64_t* kID) {
  *kID = parallel_for_id;
  printf("kokkosp_begin_parallel_for:%s:%lu::", name, *kID);
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
  printf("kokkosp_end_parallel_for:%lu::", kID);
}

extern "C" void kokkosp_begin_parallel_scan(const char* name,
                                            const uint32_t devID,
                                            uint64_t* kID) {
  *kID = parallel_scan_id;
  printf("kokkosp_begin_parallel_scan:%s:%lu::", name, *kID);
}

extern "C" void kokkosp_end_parallel_scan(const uint64_t kID) {
  printf("kokkosp_end_parallel_scan:%lu::", kID);
}

extern "C" void kokkosp_begin_parallel_reduce(const char* name,
                                              const uint32_t devID,
                                              uint64_t* kID) {
  *kID = parallel_reduce_id;
  printf("kokkosp_begin_parallel_reduce:%s:%lu::", name, *kID);
}

extern "C" void kokkosp_end_parallel_reduce(const uint64_t kID) {
  printf("kokkosp_end_parallel_reduce:%lu::", kID);
}

extern "C" void kokkosp_push_profile_region(char* regionName) {
  printf("kokkosp_push_profile_region:%s::", regionName);
}

extern "C" void kokkosp_pop_profile_region() {
  printf("kokkosp_pop_profile_region::");
}

extern "C" void kokkosp_allocate_data(SpaceHandle handle, const char* name,
                                      void* ptr, uint64_t size) {
  printf("kokkosp_allocate_data:%s:%s:%p:%lu::", handle.name, name, ptr, size);
}

extern "C" void kokkosp_deallocate_data(SpaceHandle handle, const char* name,
                                        void* ptr, uint64_t size) {
  printf("kokkosp_deallocate_data:%s:%s:%p:%lu::", handle.name, name, ptr,
         size);
}

extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char* dst_name,
                                        const void* dst_ptr,
                                        SpaceHandle src_handle,
                                        const char* src_name,
                                        const void* src_ptr, uint64_t size) {
  printf("kokkosp_begin_deep_copy:%s:%s:%p:%s:%s:%p:%lu::", dst_handle.name,
         dst_name, dst_ptr, src_handle.name, src_name, src_ptr, size);
}

extern "C" void kokkosp_end_deep_copy() { printf("kokkosp_end_deep_copy::"); }

uint32_t section_id = 3;
extern "C" void kokkosp_create_profile_section(const char* name,
                                               uint32_t* sec_id) {
  *sec_id = section_id;
  printf("kokkosp_create_profile_section:%s:%u::", name, *sec_id);
}

extern "C" void kokkosp_start_profile_section(uint32_t sec_id) {
  printf("kokkosp_start_profile_section:%u::", sec_id);
}

extern "C" void kokkosp_stop_profile_section(uint32_t sec_id) {
  printf("kokkosp_stop_profile_section:%u::", sec_id);
}
extern "C" void kokkosp_destroy_profile_section(uint32_t sec_id) {
  printf("kokkosp_destroy_profile_section:%u::", sec_id);
}

extern "C" void kokkosp_profile_event(const char* name) {
  printf("kokkosp_profile_event:%s::", name);
}
