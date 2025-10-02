#include <cuda.h>

#include <limits>
#if (CUDA_VERSION >= 12080) && not defined(DESUL_CUDA_ARCH_IS_PRE_HOPPER)
// CUDA 12.8 introduced 128-bit compare and swap support on Hopper+ (compute
// capability 9.x and higher)
#define DESUL_HAVE_CUDA_128BIT_CAS
#endif

namespace desul {
namespace Impl {

#include <desul/atomics/cuda/cuda_cc7_asm_exchange.inc>

#ifdef DESUL_HAVE_CUDA_128BIT_CAS
#include <desul/atomics/cuda/cuda_cc9_asm_exchange.inc>
#endif
}  // namespace Impl
}  // namespace desul
