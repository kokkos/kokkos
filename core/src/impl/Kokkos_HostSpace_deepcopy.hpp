#ifdef _OPENMP
#ifdef JJE_DEBUG_PAR_DEEPCOPY
  #include <cstdio>
#endif

#include <omp.h>
#endif

#define PAR_DEEP_COPY_USE_MEMCPY

namespace Kokkos {

namespace Impl {

void hostspace_parallel_deepcopy(void * dst, const void * src, size_t n);

} // namespace Impl

} // namespace Kokkos

