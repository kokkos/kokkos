#include "Kokkos_HostSpace_deepcopy.hpp"
#include "Kokkos_Core.hpp"
namespace Kokkos {

namespace Impl {

#ifndef KOKKOS_IMPL_HOST_DEEP_COPY_SERIAL_LIMIT
#define KOKKOS_IMPL_HOST_DEEP_COPY_SERIAL_LIMIT 10*8192
#endif

void hostspace_parallel_deepcopy(void * dst, const void * src, size_t n) {
  if((n<KOKKOS_IMPL_HOST_DEEP_COPY_SERIAL_LIMIT) || (Kokkos::DefaultHostExecutionSpace().concurrency()==1)) {
    std::memcpy(dst,src,n);
    return;
  }
  if(n%8==0) {
    double* dst_p = (double*)dst;
    const double* src_p = (double*)src;
    Kokkos::parallel_for("Kokkos::Impl::host_space_deepcopy",n/8,[=](const ptrdiff_t i) {
      dst_p[i] = src_p[i];
    });
  } else if(n%4==0) {
    int32_t* dst_p = (int32_t*)dst;
    const int32_t* src_p = (int32_t*)src;
    Kokkos::parallel_for("Kokkos::Impl::host_space_deepcopy",n,[=](const ptrdiff_t i) {
      dst_p[i] = src_p[i];
    });
  } else {
    char* dst_p = (char*)dst;
    const char* src_p = (char*)src;
    Kokkos::parallel_for("Kokkos::Impl::host_space_deepcopy",n,[=](const ptrdiff_t i) {
      dst_p[i] = src_p[i];
    });
  }
}

} // namespace Impl

} // namespace Kokkos

