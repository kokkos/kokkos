#include <limits>
namespace desul {
namespace Impl {
#include <desul/atomics/cuda/cuda_cc7_asm_exchange.inc>

#ifndef DESUL_CUDA_ARCH_IS_PRE_HOPPER
// Hopper (CC90) and above has some 128 bit atomic support
#include <desul/atomics/cuda/cuda_cc9_asm_exchange.inc>
#endif
}
}  // namespace desul
