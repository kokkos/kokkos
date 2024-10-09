function(KOKKOS_ARCH_OPTION SUFFIX DEV_TYPE DESCRIPTION DEPENDENCY)
  #all optimizations off by default
  kokkos_dependent_option(ARCH_${SUFFIX} "Optimize for ${DESCRIPTION} (${DEV_TYPE})" OFF "${DEPENDENCY}" OFF)
  set(KOKKOS_ARCH_${SUFFIX} ${KOKKOS_ARCH_${SUFFIX}} PARENT_SCOPE)
  set(KOKKOS_OPTION_KEYS ${KOKKOS_OPTION_KEYS} PARENT_SCOPE)
  set(KOKKOS_OPTION_VALUES ${KOKKOS_OPTION_VALUES} PARENT_SCOPE)
  set(KOKKOS_OPTION_TYPES ${KOKKOS_OPTION_TYPES} PARENT_SCOPE)
  if(KOKKOS_ARCH_${SUFFIX})
    list(APPEND KOKKOS_ENABLED_ARCH_LIST ${SUFFIX})
    set(KOKKOS_ENABLED_ARCH_LIST ${KOKKOS_ENABLED_ARCH_LIST} PARENT_SCOPE)
  endif()
endfunction()

# Make sure devices and compiler ID are done
kokkos_cfg_depends(ARCH COMPILER_ID)
kokkos_cfg_depends(ARCH DEVICES)
kokkos_cfg_depends(ARCH OPTIONS)

kokkos_check_deprecated_options(
  ARCH_EPYC "Please replace EPYC with ZEN or ZEN2, depending on your platform" ARCH_RYZEN
  "Please replace RYZEN with ZEN or ZEN2, depending on your platform"
)

#-------------------------------------------------------------------------------
# List of possible host architectures.
#-------------------------------------------------------------------------------
set(KOKKOS_ARCH_LIST)

include(CheckCXXCompilerFlag)

kokkos_deprecated_list(ARCH ARCH)

set(HOST_ARCH_ALREADY_SPECIFIED "")
macro(DECLARE_AND_CHECK_HOST_ARCH ARCH LABEL)
  kokkos_arch_option(${ARCH} HOST "${LABEL}" TRUE)
  if(KOKKOS_ARCH_${ARCH})
    if(HOST_ARCH_ALREADY_SPECIFIED)
      message(
        FATAL_ERROR
          "Multiple host architectures given! Already have ${HOST_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again."
      )
    endif()
    set(HOST_ARCH_ALREADY_SPECIFIED ${ARCH})
  endif()
endmacro()

declare_and_check_host_arch(NATIVE "local machine")
declare_and_check_host_arch(AMDAVX "AMD chip")
declare_and_check_host_arch(ARMV80 "ARMv8.0 Compatible CPU")
declare_and_check_host_arch(ARMV81 "ARMv8.1 Compatible CPU")
declare_and_check_host_arch(ARMV8_THUNDERX "ARMv8 Cavium ThunderX CPU")
declare_and_check_host_arch(ARMV8_THUNDERX2 "ARMv8 Cavium ThunderX2 CPU")
declare_and_check_host_arch(A64FX "ARMv8.2 with SVE Support")
declare_and_check_host_arch(ARMV9_GRACE "ARMv9 NVIDIA Grace CPU")
declare_and_check_host_arch(SNB "Intel Sandy/Ivy Bridge CPUs")
declare_and_check_host_arch(HSW "Intel Haswell CPUs")
declare_and_check_host_arch(BDW "Intel Broadwell Xeon E-class CPUs")
declare_and_check_host_arch(ICL "Intel Ice Lake Client CPUs (AVX512)")
declare_and_check_host_arch(ICX "Intel Ice Lake Xeon Server CPUs (AVX512)")
declare_and_check_host_arch(SKL "Intel Skylake Client CPUs")
declare_and_check_host_arch(SKX "Intel Skylake Xeon Server CPUs (AVX512)")
declare_and_check_host_arch(KNC "Intel Knights Corner Xeon Phi")
declare_and_check_host_arch(KNL "Intel Knights Landing Xeon Phi")
declare_and_check_host_arch(SPR "Intel Sapphire Rapids Xeon Server CPUs (AVX512)")
declare_and_check_host_arch(POWER8 "IBM POWER8 CPUs")
declare_and_check_host_arch(POWER9 "IBM POWER9 CPUs")
declare_and_check_host_arch(ZEN "AMD Zen architecture")
declare_and_check_host_arch(ZEN2 "AMD Zen2 architecture")
declare_and_check_host_arch(ZEN3 "AMD Zen3 architecture")
declare_and_check_host_arch(RISCV_SG2042 "SG2042 (RISC-V) CPUs")
declare_and_check_host_arch(RISCV_RVA22V "RVA22V (RISC-V) CPUs")

if(Kokkos_ENABLE_CUDA
   OR Kokkos_ENABLE_OPENMPTARGET
   OR Kokkos_ENABLE_OPENACC
   OR Kokkos_ENABLE_SYCL
)
  set(KOKKOS_SHOW_CUDA_ARCHS ON)
endif()

kokkos_arch_option(KEPLER30 GPU "NVIDIA Kepler generation CC 3.0" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(KEPLER32 GPU "NVIDIA Kepler generation CC 3.2" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(KEPLER35 GPU "NVIDIA Kepler generation CC 3.5" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(KEPLER37 GPU "NVIDIA Kepler generation CC 3.7" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(MAXWELL50 GPU "NVIDIA Maxwell generation CC 5.0" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(MAXWELL52 GPU "NVIDIA Maxwell generation CC 5.2" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(MAXWELL53 GPU "NVIDIA Maxwell generation CC 5.3" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(PASCAL60 GPU "NVIDIA Pascal generation CC 6.0" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(PASCAL61 GPU "NVIDIA Pascal generation CC 6.1" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(VOLTA70 GPU "NVIDIA Volta generation CC 7.0" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(VOLTA72 GPU "NVIDIA Volta generation CC 7.2" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(TURING75 GPU "NVIDIA Turing generation CC 7.5" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(AMPERE80 GPU "NVIDIA Ampere generation CC 8.0" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(AMPERE86 GPU "NVIDIA Ampere generation CC 8.6" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(ADA89 GPU "NVIDIA Ada generation CC 8.9" "KOKKOS_SHOW_CUDA_ARCHS")
kokkos_arch_option(HOPPER90 GPU "NVIDIA Hopper generation CC 9.0" "KOKKOS_SHOW_CUDA_ARCHS")

if(Kokkos_ENABLE_HIP
   OR Kokkos_ENABLE_OPENMPTARGET
   OR Kokkos_ENABLE_OPENACC
   OR Kokkos_ENABLE_SYCL
)
  set(KOKKOS_SHOW_HIP_ARCHS ON)
endif()

# AMD archs ordered in decreasing priority of autodetection
list(APPEND SUPPORTED_AMD_GPUS MI300 MI300)
list(APPEND SUPPORTED_AMD_ARCHS AMD_GFX942 AMD_GFX940)
list(APPEND CORRESPONDING_AMD_FLAGS gfx942 gfx940)
list(APPEND SUPPORTED_AMD_GPUS MI200 MI200 MI100 MI100)
list(APPEND SUPPORTED_AMD_ARCHS VEGA90A AMD_GFX90A VEGA908 AMD_GFX908)
list(APPEND CORRESPONDING_AMD_FLAGS gfx90a gfx90a gfx908 gfx908)
list(APPEND SUPPORTED_AMD_GPUS MI50/60 MI50/60)
list(APPEND SUPPORTED_AMD_ARCHS VEGA906 AMD_GFX906)
list(APPEND CORRESPONDING_AMD_FLAGS gfx906 gfx906)
list(APPEND SUPPORTED_AMD_GPUS PHOENIX RX7900XTX V620/W6800 V620/W6800)
list(APPEND SUPPORTED_AMD_ARCHS AMD_GFX1103 AMD_GFX1100 NAVI1030 AMD_GFX1030)
list(APPEND CORRESPONDING_AMD_FLAGS gfx1103 gfx1100 gfx1030 gfx1030)

#FIXME CAN BE REPLACED WITH LIST_ZIP IN CMAKE 3.17
foreach(ARCH IN LISTS SUPPORTED_AMD_ARCHS)
  list(FIND SUPPORTED_AMD_ARCHS ${ARCH} LIST_INDEX)
  list(GET SUPPORTED_AMD_GPUS ${LIST_INDEX} GPU)
  list(GET CORRESPONDING_AMD_FLAGS ${LIST_INDEX} FLAG)
  kokkos_arch_option(${ARCH} GPU "AMD GPU ${GPU} ${FLAG}" "KOKKOS_SHOW_HIP_ARCHS")
endforeach()

if(Kokkos_ENABLE_SYCL OR Kokkos_ENABLE_OPENMPTARGET)
  set(KOKKOS_SHOW_SYCL_ARCHS ON)
endif()

kokkos_arch_option(INTEL_GEN GPU "SPIR64-based devices, e.g. Intel GPUs, using JIT" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_DG1 GPU "Intel Iris XeMAX GPU" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_GEN9 GPU "Intel GPU Gen9" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_GEN11 GPU "Intel GPU Gen11" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_GEN12LP GPU "Intel GPU Gen12LP" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_XEHP GPU "Intel GPU Xe-HP" "KOKKOS_SHOW_SYCL_ARCHS")
kokkos_arch_option(INTEL_PVC GPU "Intel GPU Ponte Vecchio" "KOKKOS_SHOW_SYCL_ARCHS")

if(KOKKOS_ENABLE_COMPILER_WARNINGS)
  set(COMMON_WARNINGS
      "-Wall"
      "-Wextra"
      "-Wunused-parameter"
      "-Wshadow"
      "-pedantic"
      "-Wsign-compare"
      "-Wtype-limits"
      "-Wuninitialized"
      "-Wsuggest-override"
  )

  # NOTE KOKKOS_ prefixed variable (all uppercase) is not set yet because TPLs are processed after ARCH
  if(Kokkos_ENABLE_LIBQUADMATH)
    # warning: non-standard suffix on floating constant [-Wpedantic]
    list(REMOVE_ITEM COMMON_WARNINGS "-pedantic")
  endif()

  # NVHPC compiler does not support -Wtype-limits.
  if(KOKKOS_ENABLE_OPENACC)
    if(KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
      list(REMOVE_ITEM COMMON_WARNINGS "-Wtype-limits")
    endif()
  endif()

  if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
    list(APPEND COMMON_WARNINGS "-Wimplicit-fallthrough")
  endif()

  set(GNU_WARNINGS "-Wempty-body" "-Wclobbered" "-Wignored-qualifiers" ${COMMON_WARNINGS})
  if(KOKKOS_CXX_COMPILER_ID STREQUAL GNU)
    list(APPEND GNU_WARNINGS "-Wimplicit-fallthrough")
  endif()

  # Not using COMPILER_SPECIFIC_FLAGS function so the warning flags are not passed downstream
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    string(REPLACE ";" " " WARNING_FLAGS "${GNU_WARNINGS}")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
    # FIXME_NVHPC
  else()
    string(REPLACE ";" " " WARNING_FLAGS "${COMMON_WARNINGS}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
endif()

#------------------------------- KOKKOS_CUDA_OPTIONS ---------------------------
#clear anything that might be in the cache
global_set(KOKKOS_CUDA_OPTIONS)
# Construct the Makefile options
if(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  global_append(KOKKOS_CUDA_OPTIONS "-extended-lambda")
  global_append(KOKKOS_CUDA_OPTIONS "-Wext-lambda-captures-this")
endif()

if(KOKKOS_ENABLE_CUDA_CONSTEXPR)
  if(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    global_append(KOKKOS_CUDA_OPTIONS "-expt-relaxed-constexpr")
  endif()
endif()

if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  set(CUDA_ARCH_FLAG "--cuda-gpu-arch")
  global_append(KOKKOS_CUDA_OPTIONS -x cuda)
  # Kokkos_CUDA_DIR has priority over CUDAToolkit_BIN_DIR
  if(Kokkos_CUDA_DIR)
    global_append(KOKKOS_CUDA_OPTIONS --cuda-path=${Kokkos_CUDA_DIR})
  elseif(CUDAToolkit_BIN_DIR)
    global_append(KOKKOS_CUDA_OPTIONS --cuda-path=${CUDAToolkit_BIN_DIR}/..)
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  set(CUDA_ARCH_FLAG "-arch")
endif()

if(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _UPPERCASE_CMAKE_BUILD_TYPE)
  if(KOKKOS_ENABLE_DEBUG OR _UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    global_append(KOKKOS_CUDA_OPTIONS -lineinfo)
  endif()
  unset(_UPPERCASE_CMAKE_BUILD_TYPE)
endif()

#------------------------------- KOKKOS_HIP_OPTIONS ---------------------------
kokkos_option(IMPL_AMDGPU_FLAGS "" STRING "Set compiler flags for AMD GPUs")
kokkos_option(IMPL_AMDGPU_LINK "" STRING "Set linker flags for AMD GPUs")
mark_as_advanced(Kokkos_IMPL_AMDGPU_FLAGS)
mark_as_advanced(Kokkos_IMPL_AMDGPU_LINK)

#clear anything that might be in the cache
global_set(KOKKOS_AMDGPU_OPTIONS)
if(KOKKOS_ENABLE_HIP)
  set(AMDGPU_ARCH_FLAG "--offload-arch")
  if(NOT KOKKOS_CXX_COMPILER_ID STREQUAL HIPCC)
    if(NOT CMAKE_CXX_STANDARD)
      message(FATAL_ERROR "Kokkos requires CMAKE_CXX_STANDARD to set to 17 or higher")
    endif()
    global_append(KOKKOS_AMDGPU_OPTIONS -xhip)
    if(DEFINED ENV{ROCM_PATH})
      global_append(KOKKOS_AMDGPU_OPTIONS --rocm-path=$ENV{ROCM_PATH})
    endif()
  endif()
endif()

if(KOKKOS_ARCH_NATIVE)
  if(KOKKOS_CXX_HOST_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "MSVC doesn't support ARCH_NATIVE!")
  endif()

  string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" KOKKOS_UC_SYSTEM_PROCESSOR)
  if(KOKKOS_UC_SYSTEM_PROCESSOR MATCHES "(X86)|(AMD64)")
    set(KOKKOS_NATIVE_FLAGS "-march=native;-mtune=native")
  else()
    set(KOKKOS_NATIVE_FLAGS "-mcpu=native")
  endif()
  compiler_specific_flags(COMPILER_ID KOKKOS_CXX_HOST_COMPILER_ID NVHPC -tp=native DEFAULT ${KOKKOS_NATIVE_FLAGS})
endif()

if(KOKKOS_ARCH_ARMV80)
  set(KOKKOS_ARCH_ARM_NEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    MSVC
    /arch:armv8.0
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -march=armv8-a
  )
endif()

if(KOKKOS_ARCH_ARMV81)
  set(KOKKOS_ARCH_ARM_NEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    MSVC
    /arch:armv8.1
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -march=armv8.1-a
  )
endif()

if(KOKKOS_ARCH_ARMV8_THUNDERX)
  set(KOKKOS_ARCH_ARM_NEON ON)
  set(KOKKOS_ARCH_ARMV80 ON) #Not a cache variable
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    MSVC
    /arch:armv8.0
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -march=armv8-a
    -mtune=thunderx
  )
endif()

if(KOKKOS_ARCH_ARMV8_THUNDERX2)
  set(KOKKOS_ARCH_ARM_NEON ON)
  set(KOKKOS_ARCH_ARMV81 ON) #Not a cache variable
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    MSVC
    /arch:armv8.1
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -mcpu=thunderx2t99
    -mtune=thunderx2t99
  )
endif()

if(KOKKOS_ARCH_A64FX)
  set(KOKKOS_ARCH_ARM_NEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Clang
    -march=armv8.2-a+sve
    -msve-vector-bits=512
    GNU
    -march=armv8.2-a+sve
    -msve-vector-bits=512
    MSVC
    NO-VALUE-SPECIFIED
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -march=armv8.2-a+sve
  )
endif()

if(KOKKOS_ARCH_ARMV9_GRACE)
  set(KOKKOS_ARCH_ARM_NEON ON)
  check_cxx_compiler_flag("-mcpu=neoverse-n2" COMPILER_SUPPORTS_NEOVERSE_N2)
  check_cxx_compiler_flag("-msve-vector-bits=128" COMPILER_SUPPORTS_SVE_VECTOR_BITS)
  if(COMPILER_SUPPORTS_NEOVERSE_N2 AND COMPILER_SUPPORTS_SVE_VECTOR_BITS)
    compiler_specific_flags(COMPILER_ID KOKKOS_CXX_HOST_COMPILER_ID DEFAULT -mcpu=neoverse-n2 -msve-vector-bits=128)
  else()
    message(WARNING "Compiler does not support ARMv9 Grace architecture")
  endif()
endif()

if(KOKKOS_ARCH_ZEN)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Intel
    -mavx2
    MSVC
    /arch:AVX2
    NVHPC
    -tp=zen
    DEFAULT
    -march=znver1
    -mtune=znver1
  )
  set(KOKKOS_ARCH_AMD_ZEN ON)
  set(KOKKOS_ARCH_AVX2 ON)
endif()

if(KOKKOS_ARCH_ZEN2)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Intel
    -mavx2
    MSVC
    /arch:AVX2
    NVHPC
    -tp=zen2
    DEFAULT
    -march=znver2
    -mtune=znver2
  )
  set(KOKKOS_ARCH_AMD_ZEN2 ON)
  set(KOKKOS_ARCH_AVX2 ON)
endif()

if(KOKKOS_ARCH_ZEN3)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Intel
    -mavx2
    MSVC
    /arch:AVX2
    NVHPC
    -tp=zen2
    DEFAULT
    -march=znver3
    -mtune=znver3
  )
  set(KOKKOS_ARCH_AMD_ZEN3 ON)
  set(KOKKOS_ARCH_AVX2 ON)
endif()

if(KOKKOS_ARCH_SNB OR KOKKOS_ARCH_AMDAVX)
  set(KOKKOS_ARCH_AVX ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -mavx
    MSVC
    /arch:AVX
    NVHPC
    -tp=sandybridge
    DEFAULT
    -mavx
  )
endif()

if(KOKKOS_ARCH_HSW)
  set(KOKKOS_ARCH_AVX2 ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -xCORE-AVX2
    MSVC
    /arch:AVX2
    NVHPC
    -tp=haswell
    DEFAULT
    -march=core-avx2
    -mtune=core-avx2
  )
endif()

if(KOKKOS_ARCH_RISCV_SG2042)
  if(NOT (KOKKOS_CXX_COMPILER_ID STREQUAL GNU AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
     OR (KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)
  )
    message(SEND_ERROR "Only gcc >= 12 and clang >= 14 support RISC-V.")
  endif()
  compiler_specific_flags(COMPILER_ID KOKKOS_CXX_HOST_COMPILER_ID DEFAULT -march=rv64imafdcv)
endif()

if(KOKKOS_ARCH_RISCV_RVA22V)
  if(NOT (KOKKOS_CXX_COMPILER_ID STREQUAL GNU AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
     OR (KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14)
  )
    message(SEND_ERROR "Only gcc >= 12 and clang >= 14 support RISC-V.")
  endif()
  compiler_specific_flags(
    COMPILER_ID KOKKOS_CXX_HOST_COMPILER_ID DEFAULT
    -march=rv64imafdcv_sscofpmf_sstc_svpbmt_zicbom_zicboz_zicbop_zihintpause
  )
endif()

if(KOKKOS_ARCH_BDW)
  set(KOKKOS_ARCH_AVX2 ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -xCORE-AVX2
    MSVC
    /arch:AVX2
    NVHPC
    -tp=haswell
    DEFAULT
    -march=core-avx2
    -mtune=core-avx2
    -mrtm
  )
endif()

if(KOKKOS_ARCH_KNL)
  #avx512-mic
  set(KOKKOS_ARCH_AVX512MIC ON) #not a cache variable
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -xMIC-AVX512
    MSVC
    /arch:AVX512
    NVHPC
    -tp=knl
    DEFAULT
    -march=knl
    -mtune=knl
  )
endif()

if(KOKKOS_ARCH_KNC)
  compiler_specific_flags(COMPILER_ID KOKKOS_CXX_HOST_COMPILER_ID MSVC NO-VALUE-SPECIFIED DEFAULT -mmic)
endif()

if(KOKKOS_ARCH_SKL)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -xSKYLAKE
    MSVC
    /arch:AVX2
    NVHPC
    -tp=skylake
    DEFAULT
    -march=skylake
    -mtune=skylake
  )
endif()

if(KOKKOS_ARCH_SKX)
  set(KOKKOS_ARCH_AVX512XEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    Cray
    NO-VALUE-SPECIFIED
    Intel
    -xCORE-AVX512
    MSVC
    /arch:AVX512
    NVHPC
    -tp=skylake
    DEFAULT
    -march=skylake-avx512
    -mtune=skylake-avx512
  )
endif()

if(KOKKOS_ARCH_ICL)
  set(KOKKOS_ARCH_AVX512XEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    /arch:AVX512
    DEFAULT
    -march=icelake-client
    -mtune=icelake-client
  )
endif()

if(KOKKOS_ARCH_ICX)
  set(KOKKOS_ARCH_AVX512XEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    /arch:AVX512
    DEFAULT
    -march=icelake-server
    -mtune=icelake-server
  )
endif()

if(KOKKOS_ARCH_SPR)
  set(KOKKOS_ARCH_AVX512XEON ON)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    /arch:AVX512
    DEFAULT
    -march=sapphirerapids
    -mtune=sapphirerapids
  )
endif()

if(KOKKOS_ARCH_POWER7)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    NO-VALUE-SPECIFIED
    NVHPC
    NO-VALUE-SPECIFIED
    DEFAULT
    -mcpu=power7
    -mtune=power7
  )
endif()

if(KOKKOS_ARCH_POWER8)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    NO-VALUE-SPECIFIED
    NVHPC
    -tp=pwr8
    DEFAULT
    -mcpu=power8
    -mtune=power8
  )
endif()

if(KOKKOS_ARCH_POWER9)
  compiler_specific_flags(
    COMPILER_ID
    KOKKOS_CXX_HOST_COMPILER_ID
    MSVC
    NO-VALUE-SPECIFIED
    NVHPC
    -tp=pwr9
    DEFAULT
    -mcpu=power9
    -mtune=power9
  )
endif()

# If Kokkos_ARCH_NATIVE is enabled, we are trying to autodetect
# the SIMD capabilities based on compiler macros.
if(KOKKOS_ARCH_NATIVE)
  # Make sure to rerun the checks if compile options have changed
  if(NOT "${KOKKOS_COMPILE_OPTIONS}" STREQUAL "${KOKKOS_COMPILE_OPTIONS_SAVED}")
    set(KOKKOS_COMPILE_OPTIONS_SAVED "${KOKKOS_COMPILE_OPTIONS}" CACHE INTERNAL "")

    set(CMAKE_REQUIRED_QUIET ON)
    set(CMAKE_REQUIRED_FLAGS "${KOKKOS_COMPILE_OPTIONS}")
    include(CheckCXXSymbolExists)

    unset(KOKKOS_COMPILER_HAS_AVX512 CACHE)
    check_cxx_symbol_exists(__AVX512F__ "" KOKKOS_COMPILER_HAS_AVX512)
    unset(KOKKOS_COMPILER_HAS_AVX2 CACHE)
    check_cxx_symbol_exists(__AVX2__ "" KOKKOS_COMPILER_HAS_AVX2)
    unset(KOKKOS_COMPILER_HAS_ARM_NEON CACHE)
    check_cxx_symbol_exists(__ARM_NEON "" KOKKOS_COMPILER_HAS_ARM_NEON)
    unset(KOKKOS_COMPILER_HAS_AVX CACHE)
    check_cxx_symbol_exists(__AVX__ "" KOKKOS_COMPILER_HAS_AVX)
    set(CMAKE_REQUIRED_FLAGS "${KOKKOS_COMPILE_OPTIONS}")

    unset(CMAKE_REQUIRED_QUIET)
    unset(CMAKE_REQUIRED_FLAGS)
  endif()

  # Only define one of these macros for now
  # to be uniform with what we are doing for other architectures.
  if(KOKKOS_COMPILER_HAS_AVX512)
    message(STATUS "SIMD: AVX512 detected")
    set(KOKKOS_ARCH_AVX512XEON ON)
  elseif(KOKKOS_COMPILER_HAS_AVX2)
    message(STATUS "SIMD: AVX2 detected")
    set(KOKKOS_ARCH_AVX2 ON)
  elseif(KOKKOS_COMPILER_HAS_ARM_NEON)
    message(STATUS "SIMD: ARM_NEON detected")
    set(KOKKOS_ARCH_ARM_NEON ON)
  elseif(KOKKOS_COMPILER_HAS_AVX)
    message(STATUS "SIMD: AVX detected")
    set(KOKKOS_ARCH_AVX ON)
  endif()
endif()

# FIXME_NVHPC nvc++ doesn't seem to support AVX512.
if(KOKKOS_CXX_HOST_COMPILER_ID STREQUAL NVHPC)
  set(KOKKOS_ARCH_AVX512XEON OFF)
endif()

# FIXME_NVCC nvcc doesn't seem to support Arm Neon.
if(KOKKOS_ARCH_ARM_NEON AND KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  unset(KOKKOS_ARCH_ARM_NEON)
endif()

if(NOT KOKKOS_COMPILE_LANGUAGE STREQUAL CUDA)
  if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
    compiler_specific_flags(Clang -fcuda-rdc NVIDIA --relocatable-device-code=true)
  endif()
endif()

# Clang needs mcx16 option enabled for Windows atomic functions
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang AND WIN32)
  compiler_specific_options(Clang -mcx16)
endif()

# MSVC ABI has many deprecation warnings, so ignore them
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
  compiler_specific_defs(Clang _CRT_SECURE_NO_WARNINGS)
endif()

#Right now we cannot get the compiler ID when cross-compiling, so just check
#that HIP is enabled
if(KOKKOS_ENABLE_HIP)
  if(KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
    compiler_specific_flags(DEFAULT -fgpu-rdc)
    if(NOT KOKKOS_CXX_COMPILER_ID STREQUAL HIPCC AND NOT KOKKOS_IMPL_AMDGPU_FLAGS)
      compiler_specific_link_options(DEFAULT --hip-link)
    endif()
  else()
    compiler_specific_flags(DEFAULT -fno-gpu-rdc)
  endif()
endif()

if(KOKKOS_ENABLE_SYCL)
  compiler_specific_flags(DEFAULT -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-dead-args-optimization)
  compiler_specific_options(DEFAULT -fsycl-unnamed-lambda)
  if(KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 2024.1.0)
    # Before oneAPI 2024.1.0 passing -fno-sycl didn't work properly
    if(NOT KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
      message(FATAL_ERROR "Kokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=OFF requires oneAPI 2024.1.0 or later")
    endif()
  elseif(KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
    compiler_specific_options(DEFAULT -fsycl-rdc)
  else()
    compiler_specific_options(DEFAULT -fno-sycl-rdc)
  endif()
endif()

# Check support for device_global variables
# FIXME_SYCL If SYCL_EXT_ONEAPI_DEVICE_GLOBAL is defined, we can use device
#   global variables with shared libraries using the "non-separable compilation"
#   implementation. Otherwise, the feature is not supported when building shared
#   libraries. Thus, we don't even check for support if shared libraries are
#   requested and SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined.
if(KOKKOS_ENABLE_SYCL)
  string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${KOKKOS_COMPILE_OPTIONS}")
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(SYCL_EXT_ONEAPI_DEVICE_GLOBAL "sycl/sycl.hpp" KOKKOS_IMPL_HAVE_SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
  if(KOKKOS_IMPL_HAVE_SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    set(KOKKOS_IMPL_SYCL_DEVICE_GLOBAL_SUPPORTED ON)
    # Use the non-separable compilation implementation to support shared libraries as well.
    compiler_specific_flags(DEFAULT -DDESUL_SYCL_DEVICE_GLOBAL_SUPPORTED)
  elseif(NOT BUILD_SHARED_LIBS AND KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
      "
      #include <sycl/sycl.hpp>
      using namespace sycl::ext::oneapi::experimental;
      using namespace sycl;

      SYCL_EXTERNAL device_global<int, decltype(properties(device_image_scope))> Foo;

      void bar(queue q) {
        q.single_task([=] {
        Foo = 42;
      });
      }

      int main(){ return 0; }
      "
      KOKKOS_IMPL_SYCL_DEVICE_GLOBAL_SUPPORTED
    )

    if(KOKKOS_IMPL_SYCL_DEVICE_GLOBAL_SUPPORTED)
      # Only the separable compilation implementation is supported.
      compiler_specific_flags(DEFAULT -fsycl-device-code-split=off -DDESUL_SYCL_DEVICE_GLOBAL_SUPPORTED)
    endif()
  endif()

  check_cxx_symbol_exists(SYCL_EXT_ONEAPI_GRAPH "sycl/sycl.hpp" KOKKOS_IMPL_HAVE_SYCL_EXT_ONEAPI_GRAPH)
endif()

set(CUDA_ARCH_ALREADY_SPECIFIED "")
function(CHECK_CUDA_ARCH ARCH FLAG)
  if(KOKKOS_ARCH_${ARCH})
    if(CUDA_ARCH_ALREADY_SPECIFIED)
      message(
        FATAL_ERROR
          "Multiple GPU architectures given! Already have ${CUDA_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again."
      )
    endif()
    set(CUDA_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
    if(NOT KOKKOS_ENABLE_CUDA
       AND NOT KOKKOS_ENABLE_OPENMPTARGET
       AND NOT KOKKOS_ENABLE_SYCL
       AND NOT KOKKOS_ENABLE_OPENACC
    )
      message(
        WARNING
          "Given CUDA arch ${ARCH}, but Kokkos_ENABLE_CUDA, Kokkos_ENABLE_SYCL, Kokkos_ENABLE_OPENACC, and Kokkos_ENABLE_OPENMPTARGET are OFF. Option will be ignored."
      )
      unset(KOKKOS_ARCH_${ARCH} PARENT_SCOPE)
    else()
      if(KOKKOS_ENABLE_CUDA)
        string(REPLACE "sm_" "" CMAKE_ARCH ${FLAG})
        set(KOKKOS_CUDA_ARCHITECTURES ${CMAKE_ARCH})
        set(KOKKOS_CUDA_ARCHITECTURES ${CMAKE_ARCH} PARENT_SCOPE)
      endif()
      set(KOKKOS_CUDA_ARCH_FLAG ${FLAG} PARENT_SCOPE)
      if(KOKKOS_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
        set(CMAKE_CUDA_ARCHITECTURES ${KOKKOS_CUDA_ARCHITECTURES} PARENT_SCOPE)
      else()
        global_append(KOKKOS_CUDA_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
        if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OR KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
          global_append(KOKKOS_LINK_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
        endif()
      endif()
    endif()
  endif()
  list(APPEND KOKKOS_CUDA_ARCH_FLAGS ${FLAG})
  set(KOKKOS_CUDA_ARCH_FLAGS ${KOKKOS_CUDA_ARCH_FLAGS} PARENT_SCOPE)
  list(APPEND KOKKOS_CUDA_ARCH_LIST ${ARCH})
  set(KOKKOS_CUDA_ARCH_LIST ${KOKKOS_CUDA_ARCH_LIST} PARENT_SCOPE)
endfunction()

#These will define KOKKOS_CUDA_ARCH_FLAG
#to the corresponding flag name if ON
check_cuda_arch(KEPLER30 sm_30)
check_cuda_arch(KEPLER32 sm_32)
check_cuda_arch(KEPLER35 sm_35)
check_cuda_arch(KEPLER37 sm_37)
check_cuda_arch(MAXWELL50 sm_50)
check_cuda_arch(MAXWELL52 sm_52)
check_cuda_arch(MAXWELL53 sm_53)
check_cuda_arch(PASCAL60 sm_60)
check_cuda_arch(PASCAL61 sm_61)
check_cuda_arch(VOLTA70 sm_70)
check_cuda_arch(VOLTA72 sm_72)
check_cuda_arch(TURING75 sm_75)
check_cuda_arch(AMPERE80 sm_80)
check_cuda_arch(AMPERE86 sm_86)
check_cuda_arch(ADA89 sm_89)
check_cuda_arch(HOPPER90 sm_90)

set(AMDGPU_ARCH_ALREADY_SPECIFIED "")
function(CHECK_AMDGPU_ARCH ARCH FLAG)
  if(KOKKOS_ARCH_${ARCH})
    if(AMDGPU_ARCH_ALREADY_SPECIFIED)
      message(
        FATAL_ERROR
          "Multiple GPU architectures given! Already have ${AMDGPU_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again."
      )
    endif()
    set(AMDGPU_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
    if(NOT KOKKOS_ENABLE_HIP
       AND NOT KOKKOS_ENABLE_OPENMPTARGET
       AND NOT KOKKOS_ENABLE_OPENACC
       AND NOT KOKKOS_ENABLE_SYCL
    )
      message(
        WARNING
          "Given AMD GPU architecture ${ARCH}, but Kokkos_ENABLE_HIP, Kokkos_ENABLE_SYCL, Kokkos_ENABLE_OPENACC, and Kokkos_ENABLE_OPENMPTARGET are OFF. Option will be ignored."
      )
      unset(KOKKOS_ARCH_${ARCH} PARENT_SCOPE)
    else()
      if(KOKKOS_ENABLE_HIP)
        set(KOKKOS_HIP_ARCHITECTURES ${FLAG} PARENT_SCOPE)
      endif()
      if(NOT KOKKOS_IMPL_AMDGPU_FLAGS)
        set(KOKKOS_AMDGPU_ARCH_FLAG ${FLAG} PARENT_SCOPE)
        global_append(KOKKOS_AMDGPU_OPTIONS "${AMDGPU_ARCH_FLAG}=${FLAG}")
      endif()
      if(KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
        global_append(KOKKOS_LINK_OPTIONS "${AMDGPU_ARCH_FLAG}=${FLAG}")
      endif()
    endif()
  endif()
endfunction()

#These will define KOKKOS_AMDGPU_ARCH_FLAG
#to the corresponding flag name if ON
foreach(ARCH IN LISTS SUPPORTED_AMD_ARCHS)
  list(FIND SUPPORTED_AMD_ARCHS ${ARCH} LIST_INDEX)
  list(GET CORRESPONDING_AMD_FLAGS ${LIST_INDEX} FLAG)
  check_amdgpu_arch(${ARCH} ${FLAG})
endforeach()

if(KOKKOS_IMPL_AMDGPU_FLAGS)
  if(NOT AMDGPU_ARCH_ALREADY_SPECIFIED)
    message(FATAL_ERROR "When IMPL_AMDGPU_FLAGS is set the architecture autodectection is disabled. "
                        "Please explicitly set the GPU architecture."
    )
  endif()
  global_append(KOKKOS_AMDGPU_OPTIONS "${KOKKOS_IMPL_AMDGPU_FLAGS}")
  global_append(KOKKOS_LINK_OPTIONS "${KOKKOS_IMPL_AMDGPU_LINK}")
endif()

macro(SET_AND_CHECK_AMD_ARCH ARCH FLAG)
  kokkos_set_option(ARCH_${ARCH} ON)
  check_amdgpu_arch(${ARCH} ${FLAG})
  list(APPEND KOKKOS_ENABLED_ARCH_LIST ${ARCH})
endmacro()

macro(CHECK_MULTIPLE_INTEL_ARCH)
  if(KOKKOS_ARCH_INTEL_GPU)
    message(FATAL_ERROR "Specifying multiple Intel GPU architectures is not allowed!")
  endif()
  set(KOKKOS_ARCH_INTEL_GPU ON)
endmacro()

if(KOKKOS_ARCH_INTEL_GEN)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_DG1)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_GEN9)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_GEN11)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_GEN12LP)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_XEHP)
  check_multiple_intel_arch()
endif()
if(KOKKOS_ARCH_INTEL_PVC)
  check_multiple_intel_arch()
endif()

if(KOKKOS_ENABLE_OPENMP)
  compiler_specific_link_options(CrayClang -fopenmp)
endif()

if(KOKKOS_ENABLE_OPENMPTARGET)
  set(CLANG_CUDA_ARCH ${KOKKOS_CUDA_ARCH_FLAG})
  if(CLANG_CUDA_ARCH)
    if(KOKKOS_CLANG_IS_CRAY)
      compiler_specific_flags(Cray -fopenmp)
    else()
      string(REPLACE "sm_" "cc" NVHPC_CUDA_ARCH ${CLANG_CUDA_ARCH})
      compiler_specific_flags(
        Clang -Xopenmp-target -march=${CLANG_CUDA_ARCH} -fopenmp-targets=nvptx64 NVHPC -gpu=${NVHPC_CUDA_ARCH}
      )
    endif()
  endif()
  set(CLANG_AMDGPU_ARCH ${KOKKOS_AMDGPU_ARCH_FLAG})
  if(CLANG_AMDGPU_ARCH)
    compiler_specific_flags(
      Clang -Xopenmp-target=amdgcn-amd-amdhsa -march=${CLANG_AMDGPU_ARCH} -fopenmp-targets=amdgcn-amd-amdhsa
    )
  endif()
  if(KOKKOS_ARCH_INTEL_GEN)
    compiler_specific_flags(IntelLLVM -fopenmp-targets=spir64 -D__STRICT_ANSI__)
  else()
    compiler_specific_options(IntelLLVM -fopenmp-targets=spir64_gen -D__STRICT_ANSI__)
    if(KOKKOS_ARCH_INTEL_GEN9)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device gen9")
    elseif(KOKKOS_ARCH_INTEL_GEN11)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device gen11")
    elseif(KOKKOS_ARCH_INTEL_GEN12LP)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device gen12lp")
    elseif(KOKKOS_ARCH_INTEL_DG1)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device dg1")
    elseif(KOKKOS_ARCH_INTEL_XEHP)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device 12.50.4")
    elseif(KOKKOS_ARCH_INTEL_PVC)
      compiler_specific_link_options(IntelLLVM -fopenmp-targets=spir64_gen -Xopenmp-target-backend "-device 12.60.7")
    endif()
  endif()
endif()

if(KOKKOS_ENABLE_OPENACC)
  if(KOKKOS_CUDA_ARCH_FLAG)
    if(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
      message(
        FATAL_ERROR
          "If a GPU architecture is specified, Kokkos_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE option cannot be used. Disable the Kokkos_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE option."
      )
    endif()
    set(CLANG_CUDA_ARCH ${KOKKOS_CUDA_ARCH_FLAG})
    string(REPLACE "sm_" "cc" NVHPC_CUDA_ARCH ${KOKKOS_CUDA_ARCH_FLAG})
    compiler_specific_flags(
      NVHPC
      -acc
      -gpu=${NVHPC_CUDA_ARCH}
      Clang
      -Xopenmp-target=nvptx64-nvidia-cuda
      -march=${CLANG_CUDA_ARCH}
      -fopenmp-targets=nvptx64-nvidia-cuda
    )
    if(DEFINED ENV{CUDA_PATH})
      compiler_specific_link_options(Clang -L$ENV{CUDA_PATH}/lib64)
    endif()
    compiler_specific_libs(Clang -lcudart NVHPC -cuda)
  elseif(KOKKOS_AMDGPU_ARCH_FLAG)
    if(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
      message(
        FATAL_ERROR
          "If a GPU architecture is specified, Kokkos_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE option cannot be used. Disable the Kokkos_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE option."
      )
    endif()
    compiler_specific_flags(
      Clang -Xopenmp-target=amdgcn-amd-amdhsa -march=${KOKKOS_AMDGPU_ARCH_FLAG} -fopenmp-targets=amdgcn-amd-amdhsa
    )
    if(DEFINED ENV{ROCM_PATH})
      compiler_specific_flags(Clang -I$ENV{ROCM_PATH}/include)
      compiler_specific_link_options(Clang -L$ENV{ROCM_PATH}/lib)
    endif()
    compiler_specific_libs(Clang -lamdhip64)
  elseif(KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE)
    # Compile for kernel execution on the host. In that case,
    # memory is shared between the OpenACC space and the host space.
    compiler_specific_flags(NVHPC -acc=multicore)
  else()
    # Automatic fallback mode; try to offload any available GPU, and fall back
    # to the host CPU if no available GPU is found.
    compiler_specific_flags(NVHPC -acc=gpu,multicore)
    message(
      STATUS
        "No OpenACC target device is specificed; the OpenACC backend will be executed in an automatic fallback mode."
    )
  endif()
endif()

if(KOKKOS_ENABLE_SYCL)
  if(CUDA_ARCH_ALREADY_SPECIFIED)
    if(KOKKOS_ENABLE_UNSUPPORTED_ARCHS)
      compiler_specific_flags(
        DEFAULT -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda
        --cuda-gpu-arch=${KOKKOS_CUDA_ARCH_FLAG}
      )
    else()
      message(
        SEND_ERROR "Setting a CUDA architecture for SYCL is only allowed with Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON!"
      )
    endif()
  elseif(AMDGPU_ARCH_ALREADY_SPECIFIED)
    if(KOKKOS_ENABLE_UNSUPPORTED_ARCHS)
      compiler_specific_flags(
        DEFAULT -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${KOKKOS_AMDGPU_ARCH_FLAG}
      )
    else()
      message(
        SEND_ERROR "Setting a AMDGPU architecture for SYCL is only allowed with Kokkos_ENABLE_UNSUPPORTED_ARCHS=ON!"
      )
    endif()
  elseif(KOKKOS_ARCH_INTEL_GEN)
    compiler_specific_flags(DEFAULT -fsycl-targets=spir64)
  elseif(KOKKOS_ARCH_INTEL_GPU)
    set(SYCL_TARGET_FLAG -fsycl-targets=spir64_gen)

    if(KOKKOS_ARCH_INTEL_GEN9)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device gen9")
    elseif(KOKKOS_ARCH_INTEL_GEN11)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device gen11")
    elseif(KOKKOS_ARCH_INTEL_GEN12LP)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device gen12lp")
    elseif(KOKKOS_ARCH_INTEL_DG1)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device dg1")
    elseif(KOKKOS_ARCH_INTEL_XEHP)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device 12.50.4")
    elseif(KOKKOS_ARCH_INTEL_PVC)
      set(SYCL_TARGET_BACKEND_FLAG -Xsycl-target-backend "-device 12.60.7")
    endif()

    if(Kokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
      compiler_specific_options(DEFAULT ${SYCL_TARGET_FLAG})
      compiler_specific_link_options(DEFAULT ${SYCL_TARGET_FLAG} ${SYCL_TARGET_BACKEND_FLAG})
    else()
      compiler_specific_options(DEFAULT ${SYCL_TARGET_FLAG} ${SYCL_TARGET_BACKEND_FLAG})
    endif()
  endif()
endif()

if(KOKKOS_ENABLE_CUDA AND NOT CUDA_ARCH_ALREADY_SPECIFIED)
  # Try to autodetect the CUDA Compute Capability by asking the device
  set(_BINARY_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/cmake/compile_tests/CUDAComputeCapabilityWorkdir)
  file(REMOVE_RECURSE ${_BINARY_TEST_DIR})
  file(MAKE_DIRECTORY ${_BINARY_TEST_DIR})

  try_run(_RESULT _COMPILE_RESULT ${_BINARY_TEST_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc COMPILE_DEFINITIONS -DSM_ONLY
          RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY
  )

  # if user is using kokkos_compiler_launcher, above will fail.
  if(NOT _COMPILE_RESULT OR NOT _RESULT EQUAL 0)
    # check to see if CUDA is not already enabled (may happen when Kokkos is subproject)
    get_property(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    # language has to be fully enabled, just checking for CMAKE_CUDA_COMPILER isn't enough
    if(NOT "CUDA" IN_LIST _ENABLED_LANGUAGES)
      # make sure the user knows that we aren't using CUDA compiler for anything else
      message(
        STATUS
          "CUDA auto-detection of architecture failed with ${CMAKE_CXX_COMPILER}. Enabling CUDA language ONLY to auto-detect architecture..."
      )
      include(CheckLanguage)
      check_language(CUDA)
      if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
      else()
        message(STATUS "CUDA language could not be enabled")
      endif()
    endif()

    # if CUDA was enabled, this will be defined
    if(CMAKE_CUDA_COMPILER)
      # copy our test to .cu so cmake compiles as CUDA
      configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc
        ${CMAKE_CURRENT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu COPYONLY
      )
      # run test again
      try_run(_RESULT _COMPILE_RESULT ${_BINARY_TEST_DIR}
              ${CMAKE_CURRENT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu COMPILE_DEFINITIONS -DSM_ONLY
              RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY
      )
    endif()
  endif()

  list(FIND KOKKOS_CUDA_ARCH_FLAGS sm_${_CUDA_COMPUTE_CAPABILITY} FLAG_INDEX)
  if(_COMPILE_RESULT AND _RESULT EQUAL 0 AND NOT FLAG_INDEX EQUAL -1)
    message(STATUS "Detected CUDA Compute Capability ${_CUDA_COMPUTE_CAPABILITY}")
    list(GET KOKKOS_CUDA_ARCH_LIST ${FLAG_INDEX} ARCHITECTURE)
    kokkos_set_option(ARCH_${ARCHITECTURE} ON)
    check_cuda_arch(${ARCHITECTURE} sm_${_CUDA_COMPUTE_CAPABILITY})
    list(APPEND KOKKOS_ENABLED_ARCH_LIST ${ARCHITECTURE})
  else()
    message(
      SEND_ERROR
        "CUDA enabled but no NVIDIA GPU architecture currently enabled and auto-detection failed. "
        "Please give one -DKokkos_ARCH_{..}=ON' to enable an NVIDIA GPU architecture.\n"
        "You can yourself try to compile ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compile_tests/cuda_compute_capability.cc and run the executable. "
        "If you are cross-compiling, you should try to do this on a compute node."
    )
  endif()
endif()

#Regardless of version, make sure we define the general architecture name
if(KOKKOS_ARCH_KEPLER30
   OR KOKKOS_ARCH_KEPLER32
   OR KOKKOS_ARCH_KEPLER35
   OR KOKKOS_ARCH_KEPLER37
)
  set(KOKKOS_ARCH_KEPLER ON)
endif()

#Regardless of version, make sure we define the general architecture name
if(KOKKOS_ARCH_MAXWELL50 OR KOKKOS_ARCH_MAXWELL52 OR KOKKOS_ARCH_MAXWELL53)
  set(KOKKOS_ARCH_MAXWELL ON)
endif()

#Regardless of version, make sure we define the general architecture name
if(KOKKOS_ARCH_PASCAL60 OR KOKKOS_ARCH_PASCAL61)
  set(KOKKOS_ARCH_PASCAL ON)
endif()

#Regardless of version, make sure we define the general architecture name
if(KOKKOS_ARCH_VOLTA70 OR KOKKOS_ARCH_VOLTA72)
  set(KOKKOS_ARCH_VOLTA ON)
endif()

if(KOKKOS_ARCH_AMPERE80 OR KOKKOS_ARCH_AMPERE86)
  set(KOKKOS_ARCH_AMPERE ON)
endif()

if(KOKKOS_ARCH_HOPPER90)
  set(KOKKOS_ARCH_HOPPER ON)
endif()

#HIP detection of gpu arch
if(KOKKOS_ENABLE_HIP AND NOT AMDGPU_ARCH_ALREADY_SPECIFIED AND NOT KOKKOS_IMPL_AMDGPU_FLAGS)
  find_program(ROCM_ENUMERATOR rocm_agent_enumerator)
  if(NOT ROCM_ENUMERATOR)
    message(
      FATAL_ERROR "Autodetection of AMD GPU architecture not possible as " "rocm_agent_enumerator could not be found. "
                  "Please specify an arch manually via -DKokkos_ARCH_{..}=ON"
    )
  else()
    execute_process(COMMAND ${ROCM_ENUMERATOR} OUTPUT_VARIABLE GPU_ARCHS)
    string(LENGTH "${GPU_ARCHS}" len_str)
    # enumerator always output gfx000 as the first line
    if(${len_str} LESS 8)
      message(SEND_ERROR "HIP enabled but no AMD GPU architecture could be automatically detected. "
                         "Please manually specify one AMD GPU architecture via -DKokkos_ARCH_{..}=ON'."
      )
      # check for known gpu archs, otherwise error out
    else()
      set(AMD_ARCH_DETECTED "")
      foreach(ARCH IN LISTS SUPPORTED_AMD_ARCHS)
        list(FIND SUPPORTED_AMD_ARCHS ${ARCH} LIST_INDEX)
        list(GET CORRESPONDING_AMD_FLAGS ${LIST_INDEX} FLAG)
        string(REGEX MATCH "(${FLAG})" DETECTED_GPU_ARCH ${GPU_ARCHS})
        if("${DETECTED_GPU_ARCH}" STREQUAL "${FLAG}")
          set_and_check_amd_arch(${ARCH} ${FLAG})
          set(AMD_ARCH_DETECTED ${ARCH})
          break()
        endif()
      endforeach()
      if("${AMD_ARCH_DETECTED}" STREQUAL "")
        message(FATAL_ERROR "HIP enabled but no automatically detected AMD GPU architecture " "is supported. "
                            "Please manually specify one AMD GPU architecture via -DKokkos_ARCH_{..}=ON'."
        )
      endif()
    endif()
  endif()
endif()

foreach(ARCH IN LISTS SUPPORTED_AMD_ARCHS)
  if(KOKKOS_ARCH_${ARCH})
    string(REGEX MATCH "90A" IS_90A ${ARCH})
    if(IS_90A)
      set(KOKKOS_ARCH_AMD_GFX90A ON)
      set(KOKKOS_ARCH_VEGA90A ON)
      break()
    endif()
    string(REGEX MATCH "908" IS_908 ${ARCH})
    if(IS_908)
      set(KOKKOS_ARCH_AMD_GFX908 ON)
      set(KOKKOS_ARCH_VEGA908 ON)
      break()
    endif()
    string(REGEX MATCH "906" IS_906 ${ARCH})
    if(IS_906)
      set(KOKKOS_ARCH_AMD_GFX906 ON)
      set(KOKKOS_ARCH_VEGA906 ON)
      break()
    endif()
    string(REGEX MATCH "1100" IS_1100 ${ARCH})
    if(IS_1100)
      set(KOKKOS_ARCH_AMD_GFX1100 ON)
      set(KOKKOS_ARCH_NAVI1100 ON)
      break()
    endif()
    string(REGEX MATCH "1030" IS_1030 ${ARCH})
    if(IS_1030)
      set(KOKKOS_ARCH_AMD_GFX1030 ON)
      set(KOKKOS_ARCH_NAVI1030 ON)
      break()
    endif()
  endif()
endforeach()

#Regardless of version, make sure we define the general architecture name
foreach(ARCH IN LISTS SUPPORTED_AMD_ARCHS)
  if(KOKKOS_ARCH_${ARCH})
    list(FIND SUPPORTED_AMD_ARCHS ${ARCH} LIST_INDEX)
    list(GET CORRESPONDING_AMD_FLAGS ${LIST_INDEX} FLAG)
    set(KOKKOS_ARCH_AMD_GPU "${FLAG}")
    string(REGEX MATCH "(VEGA)" IS_VEGA ${ARCH})
    if(IS_VEGA)
      set(KOKKOS_ARCH_VEGA ON)
      break()
    endif()
    string(REGEX MATCH "(NAVI)" IS_NAVI ${ARCH})
    if(IS_NAVI)
      set(KOKKOS_ARCH_NAVI ON)
      break()
    endif()
  endif()
endforeach()

#CMake verbose is kind of pointless
#Let's just always print things
message(STATUS "Built-in Execution Spaces:")

foreach(_BACKEND Cuda OpenMPTarget HIP SYCL OpenACC)
  string(TOUPPER ${_BACKEND} UC_BACKEND)
  if(KOKKOS_ENABLE_${UC_BACKEND})
    if(_DEVICE_PARALLEL)
      message(
        FATAL_ERROR
          "Multiple device parallel execution spaces are not allowed! "
          "Trying to enable execution space ${_BACKEND}, "
          "but execution space ${_DEVICE_PARALLEL} is already enabled. "
          "Remove the CMakeCache.txt file and re-configure."
      )
    endif()
    if(${_BACKEND} STREQUAL "Cuda")
      if(KOKKOS_ENABLE_CUDA_UVM)
        message(
          DEPRECATION
            "Setting Kokkos_ENABLE_CUDA_UVM is deprecated - use the portable Kokkos::SharedSpace as an explicit memory space in your code instead"
        )
        if(NOT KOKKOS_ENABLE_DEPRECATED_CODE_4)
          message(FATAL_ERROR "Kokkos_ENABLE_DEPRECATED_CODE_4 must be set to use Kokkos_ENABLE_CUDA_UVM")
        endif()
      endif()
      set(_DEVICE_PARALLEL "Kokkos::${_BACKEND}")
    elseif(${_BACKEND} STREQUAL "HIP" OR ${_BACKEND} STREQUAL "SYCL")
      set(_DEVICE_PARALLEL "Kokkos::${_BACKEND}")
    else()
      set(_DEVICE_PARALLEL "Kokkos::Experimental::${_BACKEND}")
    endif()
  endif()
endforeach()
if(NOT _DEVICE_PARALLEL)
  set(_DEVICE_PARALLEL "NoTypeDefined")
endif()
message(STATUS "    Device Parallel: ${_DEVICE_PARALLEL}")

foreach(_BACKEND OpenMP Threads HPX)
  string(TOUPPER ${_BACKEND} UC_BACKEND)
  if(KOKKOS_ENABLE_${UC_BACKEND})
    if(_HOST_PARALLEL)
      message(
        FATAL_ERROR
          "Multiple host parallel execution spaces are not allowed! " "Trying to enable execution space ${_BACKEND}, "
          "but execution space ${_HOST_PARALLEL} is already enabled. "
          "Remove the CMakeCache.txt file and re-configure."
      )
    endif()
    if(${_BACKEND} STREQUAL "HPX")
      set(_HOST_PARALLEL "Kokkos::Experimental::${_BACKEND}")
    else()
      set(_HOST_PARALLEL "Kokkos::${_BACKEND}")
    endif()
  endif()
endforeach()

if(NOT _HOST_PARALLEL AND NOT KOKKOS_ENABLE_SERIAL)
  message(FATAL_ERROR "At least one host execution space must be enabled, "
                      "but no host parallel execution space was requested " "and Kokkos_ENABLE_SERIAL=OFF."
  )
endif()

if(_HOST_PARALLEL)
  message(STATUS "    Host Parallel: ${_HOST_PARALLEL}")
else()
  set(_HOST_PARALLEL "NoTypeDefined")
  message(STATUS "    Host Parallel: NoTypeDefined")
endif()

if(KOKKOS_ENABLE_SERIAL)
  message(STATUS "      Host Serial: SERIAL")
else()
  message(STATUS "      Host Serial: NONE")
endif()

message(STATUS "")
message(STATUS "Architectures:")
foreach(Arch ${KOKKOS_ENABLED_ARCH_LIST})
  message(STATUS " ${Arch}")
endforeach()

if(KOKKOS_ENABLE_ATOMICS_BYPASS)
  if(NOT _HOST_PARALLEL STREQUAL "NoTypeDefined" OR NOT _DEVICE_PARALLEL STREQUAL "NoTypeDefined")
    message(
      FATAL_ERROR
        "Disabling atomics (via -DKokkos_ENABLE_ATOMICS_BYPASS=ON) is not allowed if a host parallel or a device backend is enabled!"
    )
  endif()
  if(NOT KOKKOS_ENABLE_SERIAL)
    message(FATAL_ERROR "Implementation bug") # safeguard
  endif()
  message(STATUS "Atomics: **DISABLED**")
endif()
