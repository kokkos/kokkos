// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CLOCKTIC_HPP
#define KOKKOS_CLOCKTIC_HPP

#include <Kokkos_Macros.hpp>
#include <stdint.h>
#include <chrono>

// To use OpenCL(TM) built-in intrinsics inside kernels, we have to
// forward-declare their prototype, also see
// https://github.com/intel/pti-gpu/blob/master/chapters/binary_instrumentation/OpenCLBuiltIn.md
#if defined(KOKKOS_ENABLE_SYCL) &&                         \
    defined(KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE) && \
    defined(KOKKOS_ARCH_INTEL_GPU) && defined(__SYCL_DEVICE_ONLY__)
extern SYCL_EXTERNAL unsigned long __attribute__((overloadable))
intel_get_cycle_counter();
#endif

namespace Kokkos {
namespace Impl {

/**\brief  Quick query of clock register tics
 *
 *  Primary use case is to, with low overhead,
 *  obtain a integral value that consistently varies
 *  across concurrent threads of execution within
 *  a parallel algorithm.
 *  This value is often used to "randomly" seed an
 *  attempt to acquire an indexed resource (e.g., bit)
 *  from an array of resources (e.g., bitset) such that
 *  concurrent threads will have high likelihood of
 *  having different index-seed values.
 */

#if defined(__HIP_DEVICE_COMPILE__) || defined(__AMDGCN__)
KOKKOS_IMPL_DEVICE_FUNCTION inline uint64_t amd_get_cycle_counter() noexcept {
  // See: https://gpuopen.com/amd-gpu-architecture-programming-documentation/

  // CDNA and RDNA1 expose a single 64-bit s_memtime instruction.
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) ||  \
    defined(__gfx90c__) || defined(__gfx940__) || defined(__gfx941__) ||  \
    defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1010__) || \
    defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)

#if defined(__has_builtin) && __has_builtin(__builtin_amdgcn_s_memtime)
  return __builtin_amdgcn_s_memtime();
#else
  uint64_t cycles;
  asm volatile("s_memtime %0" : "=s"(cycles));
  return cycles;
#endif

  // RDNA2, RDNA3 and RDNA3.5 replaced s_memtime with a single-instruction
  // 20-bit cycle counter exposed via the SHADER_CYCLES hardware register
  // (id 29). Used for measuring time-delta within a wave, not between waves.
  // Acceptable for the "random" per-thread seed use case documented above.
#elif defined(__gfx1030__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) ||   \
    defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
  uint32_t cycles;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES)" : "=s"(cycles));
  return static_cast<uint64_t>(cycles & 0xFFFFF);

  // RDNA4 splits the counter into SHADER_CYCLES_LO (id 29) and
  // SHADER_CYCLES_HI (id 30). Since the two halves are read by separate
  // instructions, the low half can roll over between the two reads; the
  // ISA doc recommends re-reading the high half and retrying if it changed.
#elif defined(__gfx1200__) || defined(__gfx1201__)
  uint32_t hi0, lo, hi1;
  do {
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)" : "=s"(hi0));
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_LO)" : "=s"(lo));
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES_HI)" : "=s"(hi1));
  } while (hi0 != hi1);
  return (static_cast<uint64_t>(hi1) << 32) | static_cast<uint64_t>(lo);

  // Unsupported/unrecognized AMD architecture.
#else
  return 0;
#endif
}
#endif

KOKKOS_IMPL_DEVICE_FUNCTION inline uint64_t clock_tic_device() noexcept {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)

  // Return value of 64-bit hi-res clock register.
  return clock64();

// FIXME_SYCL We can only return something useful for Intel GPUs and with RDC
#elif defined(KOKKOS_ENABLE_SYCL) &&                       \
    defined(KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE) && \
    defined(KOKKOS_ARCH_INTEL_GPU) && defined(__SYCL_DEVICE_ONLY__)

  return intel_get_cycle_counter();

// Use asm for NVIDIA GPUs when on a different backend than CUDA.
#elif defined(__CUDA_ARCH__) || defined(__NVPTX__) || defined(__PTX_SM__)
  // See:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=clock64#special-registers-clock64
  uint64_t cycles;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(cycles));
  return cycles;

// Use asm for AMD GPUs when on a different backend than HIP.
#elif defined(__HIP_DEVICE_COMPILE__) || defined(__AMDGCN__)

  return amd_get_cycle_counter();

#else

  return 0;

#endif
}

KOKKOS_IMPL_HOST_FUNCTION inline uint64_t clock_tic_host() noexcept {
#if defined(__i386__) || defined(__x86_64)

  // Return value of 64-bit hi-res clock register.

  unsigned a = 0, d = 0;

  __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));

  return ((uint64_t)a) | (((uint64_t)d) << 32);

#elif defined(__powerpc64__) || defined(__ppc64__)

  unsigned long cycles = 0;

  asm volatile("mftb %0" : "=r"(cycles));

  return (uint64_t)cycles;

#elif defined(__ppc__)
  // see : pages.cs.wisc.edu/~legault/miniproj-736.pdf or
  // cmssdt.cern.ch/lxr/source/FWCore/Utilities/interface/HRRealTime.h

  uint64_t result = 0;
  uint32_t upper, lower, tmp;

  __asm__ volatile(
      "0: \n"
      "\tmftbu %0     \n"
      "\tmftb  %1     \n"
      "\tmftbu %2     \n"
      "\tcmpw  %2, %0 \n"
      "\tbne   0b     \n"
      : "=r"(upper), "=r"(lower), "=r"(tmp));
  result = upper;
  result = result << 32;
  result = result | lower;

  return (result);

#else

  return std::chrono::high_resolution_clock::now().time_since_epoch().count();

#endif
}

KOKKOS_FORCEINLINE_FUNCTION
uint64_t clock_tic() noexcept {
  KOKKOS_IF_ON_DEVICE((return clock_tic_device();))
  KOKKOS_IF_ON_HOST((return clock_tic_host();))
  KOKKOS_IMPL_UNREACHABLE();
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_CLOCKTIC_HPP
