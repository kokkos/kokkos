#include <Kokkos_Macros.hpp>

#if !defined(KOKKOS_COMPILER_CLANG)
#define KOKKOS_IMPL_CUDA_MAX_SHFL_SIZEOF sizeof(long long)
#else
#define KOKKOS_IMPL_CUDA_MAX_SHFL_SIZEOF sizeof(int)
#endif

#if defined(__CUDA_ARCH__)
#define KOKKOS_IMPL_CUDA_SYNCWARP_OR_RETURN(MSG)                           \
  {                                                                        \
    __syncwarp();                                                          \
    const unsigned b = __activemask();                                     \
    if (b != 0xffffffff) {                                                 \
      printf(" SYNCWARP AT %s (%d,%d,%d) (%d,%d,%d) failed %x\n", MSG,     \
             blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, \
             threadIdx.z, b);                                              \
      return;                                                              \
    }                                                                      \
  }
#else
#define KOKKOS_IMPL_CUDA_SYNCWARP_OR_RETURN(MSG)
#endif
