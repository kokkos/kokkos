
FUNCTION(KOKKOS_ARCH_OPTION CAMEL_SUFFIX DOCSTRING)
  #all optimizations off by default
  kokkos_option(ARCH_${CAMEL_SUFFIX} OFF BOOL ${DOCSTRING})
ENDFUNCTION()

FUNCTION(ARCH_FLAGS)
  SET(COMPILERS NVIDIA PGI XL DEFAULT Cray Intel Clang AppleClang GNU)
  CMAKE_PARSE_ARGUMENTS(
    PARSE
    "LINK_ONLY;COMPILE_ONLY"
    ""
    "${COMPILERS}"
    ${ARGN})

  SET(COMPILER ${KOKKOS_CXX_COMPILER_ID})

  SET(FLAGS)
  SET(NEW_COMPILE_OPTIONS)
  SET(NEW_XCOMPILER_OPTIONS)
  SET(NEW_LINK_OPTIONS)
  LIST(APPEND NEW_XCOMPILER_OPTIONS ${KOKKOS_XCOMPILER_OPTIONS})
  LIST(APPEND NEW_COMPILE_OPTIONS ${KOKKOS_COMPILE_OPTIONS})
  LIST(APPEND NEW_LINK_OPTIONS ${KOKKOS_LINK_OPTIONS})
  FOREACH(COMP ${COMPILERS})
    IF (COMPILER STREQUAL "${COMP}")
      IF (PARSE_${COMPILER})
        SET(FLAGS ${PARSE_${COMPILER}})
      ELSEIF(PARSE_DEFAULT)
        SET(FLAGS ${PARSE_DEFAULT})
      ENDIF()
    ENDIF()
  ENDFOREACH()

  IF (NOT LINK_ONLY)
    # The funky logic here is for future handling of argument deduplication
    # If we naively pass multiple -Xcompiler flags to target_compile_options
    # -Xcompiler will get deduplicated and break the build
    IF ("-Xcompiler" IN_LIST FLAGS)
      LIST(REMOVE_ITEM FLAGS "-Xcompiler")
      LIST(APPEND NEW_XCOMPILER_OPTIONS ${FLAGS})
      GLOBAL_SET(KOKKOS_XCOMPILER_OPTIONS ${NEW_XCOMPILER_OPTIONS})
    ELSE()
      LIST(APPEND NEW_COMPILE_OPTIONS   ${FLAGS})
      GLOBAL_SET(KOKKOS_COMPILE_OPTIONS ${NEW_COMPILE_OPTIONS})
    ENDIF()
  ENDIF()

  IF (NOT COMPILE_ONLY)
    LIST(APPEND NEW_LINK_OPTIONS ${FLAGS})
    GLOBAL_SET(KOKKOS_LINK_OPTIONS ${NEW_LINK_OPTIONS})
  ENDIF()
ENDFUNCTION()

# Setting this variable to a value other than "None" can improve host
# performance by turning on architecture specific code.
# NOT SET is used to determine if the option is passed in.  It is reset to
# default "None" down below.
SET(KOKKOS_ARCH "NOT_SET" CACHE STRING 
    "Optimize for specific host architecture. Options are: ${KOKKOS_INTERNAL_ARCH_DOCSTR}")

# Make sure KOKKOS_ARCH is set to something
IF ("${KOKKOS_ARCH}" STREQUAL "NOT_SET")
  SET(KOKKOS_ARCH "None")
ENDIF()

#-------------------------------------------------------------------------------
# List of possible host architectures.
#-------------------------------------------------------------------------------
SET(KOKKOS_ARCH_LIST)
LIST(APPEND KOKKOS_ARCH_LIST
     None            # No architecture optimization
     AMDAVX          # (HOST) AMD chip
     ARMv80          # (HOST) ARMv8.0 Compatible CPU
     ARMv81          # (HOST) ARMv8.1 Compatible CPU
     ARMv8_ThunderX  # (HOST) ARMv8 Cavium ThunderX CPU
     ARMv8_TX2       # (HOST) ARMv8 Cavium ThunderX2 CPU
     WSM             # (HOST) Intel Westmere CPU
     SNB             # (HOST) Intel Sandy/Ivy Bridge CPUs
     HSW             # (HOST) Intel Haswell CPUs
     BDW             # (HOST) Intel Broadwell Xeon E-class CPUs
     SKX             # (HOST) Intel Sky Lake Xeon E-class HPC CPUs (AVX512)
     KNC             # (HOST) Intel Knights Corner Xeon Phi
     KNL             # (HOST) Intel Knights Landing Xeon Phi
     BGQ             # (HOST) IBM Blue Gene Q
     Power7          # (HOST) IBM POWER7 CPUs
     Power8          # (HOST) IBM POWER8 CPUs
     Power9          # (HOST) IBM POWER9 CPUs
     Kepler          # (GPU) NVIDIA Kepler default (generation CC 3.5)
     Kepler30        # (GPU) NVIDIA Kepler generation CC 3.0
     Kepler32        # (GPU) NVIDIA Kepler generation CC 3.2
     Kepler35        # (GPU) NVIDIA Kepler generation CC 3.5
     Kepler37        # (GPU) NVIDIA Kepler generation CC 3.7
     Maxwell         # (GPU) NVIDIA Maxwell default (generation CC 5.0)
     Maxwell50       # (GPU) NVIDIA Maxwell generation CC 5.0
     Maxwell52       # (GPU) NVIDIA Maxwell generation CC 5.2
     Maxwell53       # (GPU) NVIDIA Maxwell generation CC 5.3
     Pascal60        # (GPU) NVIDIA Pascal generation CC 6.0
     Pascal61        # (GPU) NVIDIA Pascal generation CC 6.1
     Volta70         # (GPU) NVIDIA Volta generation CC 7.0
     Volta72         # (GPU) NVIDIA Volta generation CC 7.2
     Turing75         # (GPU) NVIDIA Turing generation CC 7.5
     Ryzen
     Epyc
     Kaveri
     Carrizo
     Fiji
     Vega
     GFX901
    )



# Ensure that KOKKOS_ARCH is in the ARCH_LIST
IF (KOKKOS_ARCH MATCHES ",")
  MESSAGE(WARNING "-- Detected a comma in: KOKKOS_ARCH=`${KOKKOS_ARCH}`")
  MESSAGE("-- Although we prefer KOKKOS_ARCH to be semicolon-delimited, we do allow")
  MESSAGE("-- comma-delimited values for compatibility with scripts (see github.com/trilinos/Trilinos/issues/2330)")
  STRING(REPLACE "," ";" KOKKOS_ARCH "${KOKKOS_ARCH}")
  MESSAGE("-- Commas were changed to semicolons, now KOKKOS_ARCH=`${KOKKOS_ARCH}`")
ENDIF()

IF (KOKKOS_ARCH MATCHES "-")
  string(REPLACE "-" "_" KOKKOS_ARCH "${KOKKOS_ARCH}")
ENDIF()

FOREACH(Arch ${KOKKOS_ARCH_LIST})
  KOKKOS_ARCH_OPTION(${Arch} "Whether to optimize for the ${Arch} architecture")
ENDFOREACH()

set(tmpr "\n       ")
string(REPLACE ";" ${tmpr} KOKKOS_INTERNAL_ARCH_DOCSTR "${KOKKOS_ARCH_LIST}")
set(KOKKOS_INTERNAL_ARCH_DOCSTR "${tmpr}${KOKKOS_INTERNAL_ARCH_DOCSTR}")

FOREACH(Arch ${KOKKOS_ARCH})
  STRING(TOUPPER ${Arch} ARCH)
  #force on all the architectures in the list
  SET(KOKKOS_ARCH_${ARCH} ON CACHE BOOL "optimize for architecture ${Arch}" FORCE)
  IF (NOT ${Arch} IN_LIST KOKKOS_ARCH_LIST)
    MESSAGE(FATAL_ERROR "`${arch}` is not an accepted value in KOKKOS_ARCH=`${KOKKOS_ARCH}`."
      "  Please pick from these choices: ${KOKKOS_INTERNAL_ARCH_DOCSTR}")
  ENDIF()
ENDFOREACH()

IF(KOKKOS_ENABLE_COMPILER_WARNINGS)
  SET(COMMON_WARNINGS
    "-Wall" "-Wshadow" "-pedantic" 
    "-Wsign-compare" "-Wtype-limits" "-Wuninitialized")

  SET(GNU_WARNINGS "-Wempty-body" "-Wclobbered" "-Wignored-qualifiers"
    ${COMMON_WARNINGS})

  ARCH_FLAGS(
    PGI ""
    GNU     ${GNU_WARNINGS}
    DEFAULT ${COMMON_WARNINGS}
  )
ENDIF()


#------------------------------- KOKKOS_CUDA_OPTIONS ---------------------------
#clear anything that might be in the cache
GLOBAL_SET(KOKKOS_CUDA_OPTIONS "")
# Construct the Makefile options
IF (KOKKOS_ENABLE_CUDA_LAMBDA)
  GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS "-expt-extended-lambda")
ENDIF()

IF (KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  SET(CUDA_ARCH_FLAG "--cuda-gpu-arch")
  GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS -x cuda)
  SET(KOKKOS_IMPL_CUDA_CLANG_WORKAROUND ON CACHE BOOL "enable CUDA Clang workarounds" FORCE)
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  SET(CUDA_ARCH_FLAG "-arch")
ENDIF()

IF (KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF (KOKKOS_ENABLE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS -lineinfo)
  ENDIF()
  IF (KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER 9.0 OR KOKKOS_CXX_COMPILER_VERSION VERSION_EQUAL 9.0)
    GLOBAL_APPEND(KOKKOS_CUDAFE_OPTIONS --diag_suppress=esa_on_defaulted_function_ignored) 
  ENDIF()
ENDIF()

IF(KOKKOS_ENABLE_OPENMP)
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL AppleClang)
    MESSAGE(FATAL_ERROR "Apple Clang does not support OpenMP. Use native clang instead")
  ENDIF()
  ARCH_FLAGS(
    Clang      -fopenmp=libomp
    PGI        -mp
    NVIDIA     -Xcompiler -fopenmp
    Cray       ""
    XL         -qsmp=omp
    DEFAULT    -fopenmp 
  )
ENDIF()

IF (KOKKOS_ARCH_ARMV81)
  ARCH_FLAGS(
    Cray ""
    PGI  ""
    DEFAULT -march=armv8.1-a
  )
ENDIF()

IF (KOKKOS_ARCH_ARMV8_THUNDERX)
  SET(KOKKOS_ARCH_ARMV80 ON CACHE BOOL "enable armv80" FORCE)
  ARCH_FLAGS(
    Cray ""
    PGI  ""
    DEFAULT -march=armv8-a -mtune=thunderx
  )
ENDIF()

IF (KOKKOS_ARCH_ARMV8_THUNDERX2)
  SET(KOKKOS_ARCH_ARMV81 ON CACHE BOOL "enable armv80" FORCE)
  ARCH_FLAGS(
    Cray ""
    PGI  ""
    DEFAULT -march=thunderx2t99 -mtune=thunderx2t99
  )
ENDIF()

IF (KOKKOS_ARCH_EPYC)
  ARCH_FLAGS(
    Intel   -mavx2
    DEFAULT -march=znver1 -mtune=znver1
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
ENDIF()

IF (KOKKOS_ARCH_WSM)
  ARCH_FLAGS(
    Intel   -xSSE4.2
    PGI     -tp=nehalem
    Cray    ""
    DEFAULT -msse4.2
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
ENDIF()

IF (KOKKOS_ARCH_SNB OR KOKKOS_ARCH_AMDAVX)
  ARCH_FLAGS(
    Intel   -mavx
    PGI     -tp=sandybridge
    Cray    ""
    DEFAULT -mavx
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
ENDIF()

IF (KOKKOS_ARCH_HSW OR KOKKOS_ARCH_BDW)
  SET(KOKKOS_ARCH_AVX2 ON CACHE BOOL "enable avx2" FORCE)
  ARCH_FLAGS(
    Intel   -xCORE-AVX2
    PGI     -tp=haswell
    Cray    ""
    DEFAULT -march=core-avx2 -mtune=core-avx2
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
  IF (KOKKOS_ARCH_BDW)
    SET(KOKKOS_ENABLE_TM ON CACHE INTERNAL "whether transactional memory supported")
  ENDIF()
ENDIF()

IF (KOKKOS_ARCH_KNL)
  #avx512-mic
  SET(KOKKOS_ARCH_AVX512MIC ON CACHE BOOL "enable avx-512 MIC" FORCE)
  ARCH_FLAGS(
    Intel   -xMIC-AVX512
    PGI     ""
    Cray    ""
    DEFAULT -march=knl -mtune=knl
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
ENDIF()

IF (KOKKOS_ARCH_SKX)
  #avx512-xeon
  SET(KOKKOS_ARCH_AVX512XEON ON CACHE BOOL "enable avx-512 Xeon" FORCE)
  ARCH_FLAGS(
    Intel   -xCORE-AVX512
    PGI     ""
    Cray    ""
    DEFAULT -march=skylake-avx512 -march=skylake-avx512 -mrtm
  )
  SET(KOKKOS_USE_ISA_X86_64 ON CACHE INTERNAL "x86-64 architecture")
  SET(KOKKOS_ENABLE_TM ON CACHE INTERNAL "whether transactional memory supported")
ENDIF()

IF (KOKKOS_ARCH_POWER7)
  ARCH_FLAGS(
    PGI     ""
    DEFAULT -mcpu=power7 -mtune=power7
  )
  SET(KOKKOS_USE_ISA_POWERPCBE ON CACHE INTERNAL "Power PC Architecture")
ENDIF()

IF (KOKKOS_ARCH_POWER8)
  ARCH_FLAGS(
    PGI     ""
    NVIDIA  ""
    DEFAULT -mcpu=power8 -mtune=power8
  )
  SET(KOKKOS_USE_ISA_POWERPCLE ON CACHE INTERNAL "Power PC Architecture")
ENDIF()

IF (KOKKOS_ARCH_POWER9)
  ARCH_FLAGS(
    PGI     ""
    NVIDIA  ""
    DEFAULT -mcpu=power9 -mtune=power9
  )
  SET(KOKKOS_USE_ISA_POWERPCLE ON CACHE INTERNAL "Power PC Architecture")
ENDIF()


IF (KOKKOS_ARCH_KAVERI)
  SET(KOKKOS_ARCH_ROCM 701 CACHE STRING "rocm arch" FORCE)
ENDIF()

IF (KOKKOS_ARCH_CARRIZO)
  SET(KOKKOS_ARCH_ROCM 801 CACHE STRING "rocm arch" FORCE)
ENDIF()

IF (KOKKOS_ARCH_FIJI)
  SET(KOKKOS_ARCH_ROCM 803 CACHE STRING "rocm arch" FORCE)
ENDIF()

IF (KOKKOS_ARCH_VEGA)
  SET(KOKKOS_ARCH_ROCM 900 CACHE STRING "rocm arch" FORCE)
ENDIF()

IF (KOKKOS_ARCH_GFX901)
  SET(KOKKOS_ARCH_ROCM 901 CACHE STRING "rocm arch" FORCE)
ENDIF()

IF (KOKKOS_ARCH_RYZEN)
ENDIF()

IF (KOKKOS_ENABLE_CUDA_RELOCATED_DEVICE_CODE)
  ARCH_FLAGS(
    Clang  -fcuda-rdc
    NVIDIA --relocatable-device-code=true
  )
ENDIF()


SET(CUDA_ARCH_ALREADY_SPECIFIED "")
FUNCTION(CHECK_CUDA_ARCH ARCH FLAG)
IF(KOKKOS_ARCH_${ARCH})
  IF(CUDA_ARCH_ALREADY_SPECIFIED)
    MESSAGE(FATAL_ERROR "Multiple GPU architectures given! Already have ${CUDA_ARCH_ALREADY_SPECIFIED}, but trying to add ${ARCH}. If you are re-running CMake, try clearing the cache and running again.")
  ENDIF()
  SET(CUDA_ARCH_ALREADY_SPECIFIED ${ARCH} PARENT_SCOPE)
  GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS "${CUDA_ARCH_FLAG}=${FLAG}")
ENDIF()
ENDFUNCTION(CHECK_CUDA_ARCH ARCH)


CHECK_CUDA_ARCH(KEPLER30  sm_30)
CHECK_CUDA_ARCH(KEPLER32  sm_32)
CHECK_CUDA_ARCH(KEPLER35  sm_35)
CHECK_CUDA_ARCH(KEPLER37  sm_37)
CHECK_CUDA_ARCH(MAXWELL50 sm_50)
CHECK_CUDA_ARCH(MAXWELL52 sm_52)
CHECK_CUDA_ARCH(MAXWELL53 sm_53)
CHECK_CUDA_ARCH(PASCAL60  sm_60)
CHECK_CUDA_ARCH(PASCAL61  sm_61)
CHECK_CUDA_ARCH(VOLTA70  sm_70)
CHECK_CUDA_ARCH(VOLTA72  sm_72)
CHECK_CUDA_ARCH(TURING75  sm_75)


#CMake verbose is kind of pointless
#Let's just always print things
MESSAGE(STATUS "Execution Spaces:")
IF(KOKKOS_ENABLE_CUDA)
  MESSAGE(STATUS "  Device Parallel: Cuda")
ELSE()
  MESSAGE(STATUS "  Device Parallel: None")
ENDIF()

IF(KOKKOS_ENABLE_OPENMP)
  MESSAGE(STATUS "    Host Parallel: OpenMP")
ELSEIF(KOKKOS_ENABLE_PTHREAD)
  MESSAGE(STATUS "    Host Parallel: Pthread")
ELSEIF(KOKKOS_ENABLE_QTHREADS)
  MESSAGE(STATUS "    Host Parallel: Qthreads")
ELSEIF(KOKKOS_ENABLE_HPX)
  MESSAGE(STATUS "    Host Parallel: HPX")
ELSE()
  MESSAGE(STATUS "    Host Parallel: None")
ENDIF()

IF(KOKKOS_ENABLE_SERIAL)
  MESSAGE(STATUS "      Host Serial: Serial")
ELSE()
  MESSAGE(STATUS "      Host Serial: None")
ENDIF()

MESSAGE(STATUS "")
MESSAGE(STATUS "Architectures:")
FOREACH(Arch ${KOKKOS_ARCH_LIST})
  STRING(TOUPPER ${Arch} ARCH)
  IF (KOKKOS_ARCH_${ARCH})
    MESSAGE(STATUS " ${Arch}")
  ENDIF()
ENDFOREACH()
