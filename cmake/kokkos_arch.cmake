
FUNCTION(KOKKOS_ARCH_OPTION SUFFIX DEV_TYPE DESCRIPTION)
  #all optimizations off by default
  KOKKOS_OPTION(ARCH_${SUFFIX} OFF BOOL "Optimize for ${DESCRIPTION} (${DEV_TYPE})")
  IF (KOKKOS_ARCH_${SUFFIX})
    LIST(APPEND KOKKOS_ENABLED_ARCH_LIST ${SUFFIX})
    SET(KOKKOS_ENABLED_ARCH_LIST ${KOKKOS_ENABLED_ARCH_LIST} PARENT_SCOPE)
  ENDIF()
  SET(KOKKOS_ARCH_${SUFFIX} ${KOKKOS_ARCH_${SUFFIX}} PARENT_SCOPE)
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

# Make sure devices and compiler ID are done
KOKKOS_CFG_DEPENDS(ARCH COMPILER_ID)
KOKKOS_CFG_DEPENDS(ARCH DEVICES)
KOKKOS_CFG_DEPENDS(ARCH OPTIONS)


#-------------------------------------------------------------------------------
# List of possible host architectures.
#-------------------------------------------------------------------------------
SET(KOKKOS_ARCH_LIST)


KOKKOS_ARCH_OPTION(AMDAVX          HOST "AMD chip")
KOKKOS_ARCH_OPTION(ARMV80          HOST "ARMv8.0 Compatible CPU")
KOKKOS_ARCH_OPTION(ARMV81          HOST "ARMv8.1 Compatible CPU")
KOKKOS_ARCH_OPTION(ARMV8_THUNDERX  HOST "ARMv8 Cavium ThunderX CPU")
KOKKOS_ARCH_OPTION(ARMV8_TX2       HOST "ARMv8 Cavium ThunderX2 CPU")
KOKKOS_ARCH_OPTION(WSM             HOST "Intel Westmere CPU")
KOKKOS_ARCH_OPTION(SNB             HOST "Intel Sandy/Ivy Bridge CPUs")
KOKKOS_ARCH_OPTION(HSW             HOST "Intel Haswell CPUs")
KOKKOS_ARCH_OPTION(BDW             HOST "Intel Broadwell Xeon E-class CPUs")
KOKKOS_ARCH_OPTION(SKX             HOST "Intel Sky Lake Xeon E-class HPC CPUs (AVX512)")
KOKKOS_ARCH_OPTION(KNC             HOST "Intel Knights Corner Xeon Phi")
KOKKOS_ARCH_OPTION(KNL             HOST "Intel Knights Landing Xeon Phi")
KOKKOS_ARCH_OPTION(BGQ             HOST "IBM Blue Gene Q")
KOKKOS_ARCH_OPTION(POWER7          HOST "IBM POWER7 CPUs")
KOKKOS_ARCH_OPTION(POWER8          HOST "IBM POWER8 CPUs")
KOKKOS_ARCH_OPTION(POWER9          HOST "IBM POWER9 CPUs")
KOKKOS_ARCH_OPTION(KEPLER30        GPU  "NVIDIA Kepler generation CC 3.0")
KOKKOS_ARCH_OPTION(KEPLER32        GPU  "NVIDIA Kepler generation CC 3.2")
KOKKOS_ARCH_OPTION(KEPLER35        GPU  "NVIDIA Kepler generation CC 3.5")
KOKKOS_ARCH_OPTION(KEPLER37        GPU  "NVIDIA Kepler generation CC 3.7")
KOKKOS_ARCH_OPTION(MAXWELL50       GPU  "NVIDIA Maxwell generation CC 5.0")
KOKKOS_ARCH_OPTION(MAXWELL52       GPU  "NVIDIA Maxwell generation CC 5.2")
KOKKOS_ARCH_OPTION(MAXWELL53       GPU  "NVIDIA Maxwell generation CC 5.3")
KOKKOS_ARCH_OPTION(PASCAL60        GPU  "NVIDIA Pascal generation CC 6.0")
KOKKOS_ARCH_OPTION(PASCAL61        GPU  "NVIDIA Pascal generation CC 6.1")
KOKKOS_ARCH_OPTION(VOLTA70         GPU  "NVIDIA Volta generation CC 7.0")
KOKKOS_ARCH_OPTION(VOLTA72         GPU  "NVIDIA Volta generation CC 7.2")
KOKKOS_ARCH_OPTION(TURING75        GPU  "NVIDIA Turing generation CC 7.5")
KOKKOS_ARCH_OPTION(RYZEN           HOST "AMD Ryzen architecture")
KOKKOS_ARCH_OPTION(EPYC            HOST "AMD Epyc architecture")
KOKKOS_ARCH_OPTION(KAVERI          APU  "AMD Kaveri architecture")
KOKKOS_ARCH_OPTION(CARRIZO         APU  "AMD Carrizo architecture")
KOKKOS_ARCH_OPTION(FIJI            GPU  "AMD Fiji architecture")
KOKKOS_ARCH_OPTION(VEGA            GPU  "AMD Vega architecture")
KOKKOS_ARCH_OPTION(GFX901          GPU  "AMD GFX architecture")


IF (KOKKOS_ENABLE_CUDA)
 #Regardless of version, make sure we define the general architecture name
 IF (KOKKOS_ARCH_KEPLER30 OR KOKKOS_ARCH_KEPLER32 OR KOKKOS_ARCH_KEPLER35 OR KOKKOS_ARCH_KEPLER37)
   SET(KOKKOS_ARCH_KEPLER ON)
 ENDIF()
 
 #Regardless of version, make sure we define the general architecture name
 IF (KOKKOS_ARCH_MAXWELL50 OR KOKKOS_ARCH_MAXWELL52 OR KOKKOS_ARCH_MAXWELL53)
   SET(KOKKOS_ARCH_MAXWELL ON)
 ENDIF()

 #Regardless of version, make sure we define the general architecture name
 IF (KOKKOS_ARCH_PASCAL60 OR KOKKOS_ARCH_PASCAL61)
   SET(KOKKOS_ARCH_PASCAL ON)
 ENDIF()

  #Regardless of version, make sure we define the general architecture name
  IF (KOKKOS_ARCH_VOLTA70 OR KOKKOS_ARCH_VOLTA72)
    SET(KOKKOS_ARCH_VOLTA ON)
  ENDIF()
ENDIF()


KOKKOS_DEPRECATED_LIST(ARCH ARCH)

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
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS "-expt-extended-lambda")
  ENDIF()
ENDIF()

IF (KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  SET(CUDA_ARCH_FLAG "--cuda-gpu-arch")
  GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS -x cuda)
  IF (KOKKOS_ENABLE_CUDA)
     SET(KOKKOS_IMPL_CUDA_CLANG_WORKAROUND ON CACHE BOOL "enable CUDA Clang workarounds" FORCE)
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  SET(CUDA_ARCH_FLAG "-arch")
ENDIF()

IF (KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  STRING(TOUPPER "${CMAKE_BUILD_TYPE}" _UPPERCASE_CMAKE_BUILD_TYPE)
  IF (KOKKOS_ENABLE_DEBUG OR _UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    GLOBAL_APPEND(KOKKOS_CUDA_OPTIONS -lineinfo)
  ENDIF()
  UNSET(_UPPERCASE_CMAKE_BUILD_TYPE)
  IF (KOKKOS_CXX_COMPILER_VERSION VERSION_GREATER 9.0 OR KOKKOS_CXX_COMPILER_VERSION VERSION_EQUAL 9.0)
    GLOBAL_APPEND(KOKKOS_CUDAFE_OPTIONS --diag_suppress=esa_on_defaulted_function_ignored) 
  ENDIF()
ENDIF()

IF(KOKKOS_ENABLE_OPENMP)
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL AppleClang)
    MESSAGE(FATAL_ERROR "Apple Clang does not support OpenMP. Use native Clang instead")
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

IF (KOKKOS_ARCH_ARMV80)
  ARCH_FLAGS(
    Cray ""
    PGI  ""
    DEFAULT -march=armv8-a
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
  SET(KOKKOS_ARCH_ARMV80 ON) #Not a cache variable
  ARCH_FLAGS(
    Cray ""
    PGI  ""
    DEFAULT -march=armv8-a -mtune=thunderx
  )
ENDIF()

IF (KOKKOS_ARCH_ARMV8_THUNDERX2)
  SET(KOKKOS_ARCH_ARMV81 ON) #Not a cache variable
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
  SET(KOKKOS_ARCH_AMD_EPYC ON)
  SET(KOKKOS_ARCH_AMD_AVX2 ON)
ENDIF()

IF (KOKKOS_ARCH_WSM)
  ARCH_FLAGS(
    Intel   -xSSE4.2
    PGI     -tp=nehalem
    Cray    ""
    DEFAULT -msse4.2
  )
  SET(KOKKOS_ARCH_SSE42 ON)
ENDIF()

IF (KOKKOS_ARCH_SNB OR KOKKOS_ARCH_AMDAVX)
  SET(KOKKOS_ARCH_AVX ON)
  ARCH_FLAGS(
    Intel   -mavx
    PGI     -tp=sandybridge
    Cray    ""
    DEFAULT -mavx
  )
ENDIF()

IF (KOKKOS_ARCH_HSW)
  SET(KOKKOS_ARCH_AVX2 ON)
  ARCH_FLAGS(
    Intel   -xCORE-AVX2
    PGI     -tp=haswell
    Cray    ""
    DEFAULT -march=core-avx2 -mtune=core-avx2
  )
ENDIF()

IF (KOKKOS_ARCH_BDW)
  SET(KOKKOS_ARCH_AVX2 ON)
  ARCH_FLAGS(
    Intel   -xCORE-AVX2
    PGI     -tp=haswell
    Cray    ""
    DEFAULT -march=core-avx2 -mtune=core-avx2 -mrtm
  )
ENDIF()

IF (KOKKOS_ARCH_EPYC)
  SET(KOKKOS_ARCH_AMD_AVX2 ON)
  ARCH_FLAGS(
    Intel   -mvax2
    DEFAULT  -march=znver1 -mtune=znver1
  )
ENDIF()

IF (KOKKOS_ARCH_KNL)
  #avx512-mic
  SET(KOKKOS_ARCH_AVX512MIC ON) #not a cache variable
  ARCH_FLAGS(
    Intel   -xMIC-AVX512
    PGI     ""
    Cray    ""
    DEFAULT -march=knl -mtune=knl
  )
ENDIF()

IF (KOKKOS_ARCH_KNC)
  SET(KOKKOS_USE_ISA_KNC ON)
  ARCH_FLAGS(
    DEFAULT -mmic
  )
ENDIF()

IF (KOKKOS_ARCH_SKX)
  #avx512-xeon
  SET(KOKKOS_ARCH_AVX512XEON ON)
  ARCH_FLAGS(
    Intel   -xCORE-AVX512
    PGI     ""
    Cray    ""
    DEFAULT -march=skylake-avx512 -mtune=skylake-avx512 -mrtm
  )
ENDIF()

IF (KOKKOS_ARCH_WSM OR KOKKOS_ARCH_SNB OR KOKKOS_ARCH_HSW OR KOKKOS_ARCH_BDW OR KOKKOS_ARCH_KNL OR KOKKOS_ARCH_SKX OR KOKKOS_ARCH_EPYC)
  SET(KOKKOS_USE_ISA_X86_64 ON)
ENDIF()

IF (KOKKOS_ARCH_BDW OR KOKKOS_ARCH_SKX)
  SET(KOKKOS_ENABLE_TM ON) #not a cache variable
ENDIF()

IF (KOKKOS_ARCH_POWER7)
  ARCH_FLAGS(
    PGI     ""
    DEFAULT -mcpu=power7 -mtune=power7
  )
  SET(KOKKOS_USE_ISA_POWERPCBE ON)
ENDIF()

IF (KOKKOS_ARCH_POWER8)
  ARCH_FLAGS(
    PGI     ""
    NVIDIA  ""
    DEFAULT -mcpu=power8 -mtune=power8
  )
ENDIF()

IF (KOKKOS_ARCH_POWER9)
  ARCH_FLAGS(
    PGI     ""
    NVIDIA  ""
    DEFAULT -mcpu=power9 -mtune=power9
  )
ENDIF()

IF (KOKKOS_ARCH_POWER8 OR KOKKOS_ARCH_POWER9)
  SET(KOKKOS_USE_ISA_POWERPCLE ON)
ENDIF()


IF (KOKKOS_ARCH_KAVERI)
  SET(KOKKOS_ARCH_ROCM 701)
ENDIF()

IF (KOKKOS_ARCH_CARRIZO)
  SET(KOKKOS_ARCH_ROCM 801)
ENDIF()

IF (KOKKOS_ARCH_FIJI)
  SET(KOKKOS_ARCH_ROCM 803)
ENDIF()

IF (KOKKOS_ARCH_VEGA)
  SET(KOKKOS_ARCH_ROCM 900)
ENDIF()

IF (KOKKOS_ARCH_GFX901)
  SET(KOKKOS_ARCH_ROCM 901)
ENDIF()

IF (KOKKOS_ARCH_RYZEN)
ENDIF()

IF (Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
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
  IF (NOT KOKKOS_ENABLE_CUDA)
    MESSAGE(WARNING "Given CUDA arch ${ARCH}, but Kokkos_ENABLE_CUDA is OFF. Option will be ignored.")
    UNSET(KOKKOS_ARCH_${ARCH} PARENT_SCOPE)
  ENDIF()
ENDIF()
ENDFUNCTION()


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

IF (KOKKOS_ENABLE_CUDA)
  SET(KOKKOS_COMPILER_CUDA_VERSION "${KOKKOS_VERSION_MAJOR}${KOKKOS_VERSION_MINOR}")
ENDIF()

#CMake verbose is kind of pointless
#Let's just always print things
MESSAGE(STATUS "Execution Spaces:")
IF(KOKKOS_ENABLE_CUDA)
  MESSAGE(STATUS "  Device Parallel: CUDA")
ELSE()
  MESSAGE(STATUS "  Device Parallel: NONE")
ENDIF()

FOREACH (_BACKEND OPENMP PTHREAD QTHREAD HPX)
  IF(KOKKOS_ENABLE_${_BACKEND})
    IF(_HOST_PARALLEL)
      MESSAGE(FATAL_ERROR "Multiple host parallel execution spaces are not allowed! "
                          "Trying to enable execution space ${_BACKEND}, "
                          "but execution space ${_HOST_PARALLEL} is already enabled. "
                          "Remove the CMakeCache.txt file and re-configure.")
    ENDIF()
    SET(_HOST_PARALLEL ${_BACKEND})
  ENDIF()
ENDFOREACH()

IF(NOT _HOST_PARALLEL AND NOT KOKKOS_ENABLE_SERIAL)
  MESSAGE(FATAL_ERROR "At least one host execution space must be enabled, "
                      "but no host parallel execution space was requested "
                      "and Kokkos_ENABLE_SERIAL=OFF.")
ENDIF()

IF(NOT _HOST_PARALLEL)
  SET(_HOST_PARALLEL "NONE")
ENDIF()
MESSAGE(STATUS "    Host Parallel: ${_HOST_PARALLEL}")
UNSET(_HOST_PARALLEL)

IF(KOKKOS_ENABLE_PTHREAD)
  SET(KOKKOS_ENABLE_THREADS ON)
ENDIF()

IF(KOKKOS_ENABLE_SERIAL)
  MESSAGE(STATUS "      Host Serial: SERIAL")
ELSE()
  MESSAGE(STATUS "      Host Serial: NONE")
ENDIF()

MESSAGE(STATUS "")
MESSAGE(STATUS "Architectures:")
FOREACH(Arch ${KOKKOS_ENABLED_ARCH_LIST})
  MESSAGE(STATUS " ${Arch}")
ENDFOREACH()

