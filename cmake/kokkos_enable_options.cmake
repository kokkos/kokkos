########################## NOTES ###############################################
#  List the options for configuring kokkos using CMake method of doing it.

########################## AVAILABLE OPTIONS ###################################
# Use lists for documentation, verification, and programming convenience

function(KOKKOS_ENABLE_OPTION SUFFIX DEFAULT DOCSTRING)
  kokkos_option(ENABLE_${SUFFIX} ${DEFAULT} BOOL ${DOCSTRING})
  string(TOUPPER ${SUFFIX} UC_NAME)
  if(KOKKOS_ENABLE_${UC_NAME} AND NOT "Kokkos_ENABLE_${UC_NAME}" IN_LIST Kokkos_OPTIONS_NOT_TO_EXPORT)
    list(APPEND KOKKOS_ENABLED_OPTIONS ${UC_NAME})
    #I hate that CMake makes me do this
    set(KOKKOS_ENABLED_OPTIONS ${KOKKOS_ENABLED_OPTIONS} PARENT_SCOPE)
  endif()
  set(KOKKOS_ENABLE_${UC_NAME} ${KOKKOS_ENABLE_${UC_NAME}} PARENT_SCOPE)
endfunction()

# Certain defaults will depend on knowing the enabled devices
kokkos_cfg_depends(OPTIONS DEVICES)
kokkos_cfg_depends(OPTIONS COMPILER_ID)

# Put a check in just in case people are using this option
kokkos_deprecated_list(OPTIONS ENABLE)

kokkos_enable_option(CUDA_RELOCATABLE_DEVICE_CODE OFF "Whether to enable relocatable device code (RDC) for CUDA")
kokkos_enable_option(CUDA_UVM OFF "Whether to use unified memory (UM) for CUDA by default")
kokkos_enable_option(CUDA_LDG_INTRINSIC OFF "Whether to use CUDA LDG intrinsics")
# In contrast to other CUDA-dependent, options CUDA_LAMBDA is ON by default.
# That is problematic when CUDA is not enabled because this not only yields a
# bogus warning, but also exports the Kokkos_ENABLE_CUDA_LAMBDA variable and
# sets it to ON.
kokkos_enable_option(
  CUDA_LAMBDA ${KOKKOS_ENABLE_CUDA} "Whether to allow lambda expressions on the device with NVCC **DEPRECATED**"
)

# As of 09/2024, cudaMallocAsync causes issues with ICP and older version of UCX
# as MPI communication layer.
kokkos_enable_option(IMPL_CUDA_MALLOC_ASYNC OFF "Whether to enable CudaMallocAsync (requires CUDA Toolkit 11.2)")
kokkos_enable_option(IMPL_NVHPC_AS_DEVICE_COMPILER OFF "Whether to allow nvc++ as Cuda device compiler")
kokkos_enable_option(IMPL_CUDA_UNIFIED_MEMORY OFF "Whether to leverage unified memory architectures for CUDA")

kokkos_enable_option(DEPRECATED_CODE_4 ON "Whether code deprecated in major release 4 is available")
kokkos_enable_option(DEPRECATION_WARNINGS ON "Whether to emit deprecation warnings")
kokkos_enable_option(HIP_RELOCATABLE_DEVICE_CODE OFF "Whether to enable relocatable device code (RDC) for HIP")

# Disabling RDC only works properly since oneAPI 2024.1.0
if(KOKKOS_ENABLE_SYCL AND KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND KOKKOS_CXX_COMPILER_VERSION VERSION_LESS
                                                                        2024.1.0
)
  set(SYCL_RDC_DEFAULT ON)
else()
  set(SYCL_RDC_DEFAULT OFF)
endif()
kokkos_enable_option(
  SYCL_RELOCATABLE_DEVICE_CODE ${SYCL_RDC_DEFAULT} "Whether to enable relocatable device code (RDC) for SYCL"
)
kokkos_enable_option(TESTS OFF "Whether to build the unit tests")
kokkos_enable_option(BENCHMARKS OFF "Whether to build the benchmarks")
kokkos_enable_option(EXAMPLES OFF "Whether to build the examples")
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  kokkos_enable_option(DEBUG ON "Whether to activate extra debug features - may increase compile times")
  kokkos_enable_option(DEBUG_DUALVIEW_MODIFY_CHECK ON "Debug check on dual views")
else()
  kokkos_enable_option(DEBUG OFF "Whether to activate extra debug features - may increase compile times")
  kokkos_enable_option(DEBUG_DUALVIEW_MODIFY_CHECK OFF "Debug check on dual views")
endif()
unset(_UPPERCASE_CMAKE_BUILD_TYPE)
kokkos_enable_option(LARGE_MEM_TESTS OFF "Whether to perform extra large memory tests")
kokkos_enable_option(DEBUG_BOUNDS_CHECK OFF "Whether to use bounds checking - will increase runtime")
kokkos_enable_option(COMPILER_WARNINGS OFF "Whether to print all compiler warnings")
kokkos_enable_option(TUNING OFF "Whether to create bindings for tuning tools")
kokkos_enable_option(AGGRESSIVE_VECTORIZATION OFF "Whether to aggressively vectorize loops")
kokkos_enable_option(COMPILE_AS_CMAKE_LANGUAGE OFF "Whether to use native cmake language support")
kokkos_enable_option(
  HIP_MULTIPLE_KERNEL_INSTANTIATIONS OFF
  "Whether multiple kernels are instantiated at compile time - improve performance but increase compile time"
)
kokkos_enable_option(IMPL_HIP_UNIFIED_MEMORY OFF "Whether to leverage unified memory architectures for HIP")
kokkos_enable_option(OPENACC_FORCE_HOST_AS_DEVICE OFF "Whether to force to use host as a target device for OpenACC")

# This option will go away eventually, but allows fallback to old implementation when needed.
kokkos_enable_option(DESUL_ATOMICS_EXTERNAL OFF "Whether to use an external desul installation")
kokkos_enable_option(
  ATOMICS_BYPASS OFF "**NOT RECOMMENDED** Whether to make atomics non-atomic for non-threaded MPI-only use cases"
)
kokkos_enable_option(
  IMPL_REF_COUNT_BRANCH_UNLIKELY ON "Whether to use the C++20 `[[unlikely]]` attribute in the view reference counting"
)
mark_as_advanced(Kokkos_ENABLE_IMPL_REF_COUNT_BRANCH_UNLIKELY)
kokkos_enable_option(
  IMPL_VIEW_OF_VIEWS_DESTRUCTOR_PRECONDITION_VIOLATION_WORKAROUND OFF
  "Whether to enable a workaround for invalid use of View of Views that causes program hang on destruction."
)
mark_as_advanced(Kokkos_ENABLE_IMPL_VIEW_OF_VIEWS_DESTRUCTOR_PRECONDITION_VIOLATION_WORKAROUND)

kokkos_enable_option(IMPL_MDSPAN ON "Whether to enable experimental mdspan support")
kokkos_enable_option(MDSPAN_EXTERNAL OFF BOOL "Whether to use an external version of mdspan")
kokkos_enable_option(
  IMPL_SKIP_COMPILER_MDSPAN ON BOOL "Whether to use an internal version of mdspan even if the compiler supports mdspan"
)
mark_as_advanced(Kokkos_ENABLE_IMPL_MDSPAN)
mark_as_advanced(Kokkos_ENABLE_MDSPAN_EXTERNAL)
mark_as_advanced(Kokkos_ENABLE_IMPL_SKIP_COMPILER_MDSPAN)

kokkos_enable_option(COMPLEX_ALIGN ON "Whether to align Kokkos::complex to 2*alignof(RealType)")

if(KOKKOS_ENABLE_TESTS)
  set(HEADER_SELF_CONTAINMENT_TESTS_DEFAULT ON)
else()
  set(HEADER_SELF_CONTAINMENT_TESTS_DEFAULT OFF)
endif()
kokkos_enable_option(
  HEADER_SELF_CONTAINMENT_TESTS ${HEADER_SELF_CONTAINMENT_TESTS_DEFAULT} "Enable header self-containment unit tests"
)
if(NOT KOKKOS_ENABLE_TESTS AND KOKKOS_ENABLE_HEADER_SELF_CONTAINMENT_TESTS)
  message(
    WARNING "Kokkos_ENABLE_HEADER_SELF_CONTAINMENT_TESTS is ON but Kokkos_ENABLE_TESTS is OFF. Option will be ignored."
  )
endif()

if(KOKKOS_ENABLE_CUDA AND (KOKKOS_CXX_COMPILER_ID STREQUAL Clang))
  set(CUDA_CONSTEXPR_DEFAULT ON)
else()
  set(CUDA_CONSTEXPR_DEFAULT OFF)
endif()
kokkos_enable_option(
  CUDA_CONSTEXPR ${CUDA_CONSTEXPR_DEFAULT} "Whether to activate experimental relaxed constexpr functions"
)

if(KOKKOS_ENABLE_HPX)
  set(HPX_ASYNC_DISPATCH_DEFAULT ON)
else()
  set(HPX_ASYNC_DISPATCH_DEFAULT OFF)
endif()
kokkos_enable_option(IMPL_HPX_ASYNC_DISPATCH ${HPX_ASYNC_DISPATCH_DEFAULT} "Whether HPX supports asynchronous dispatch")

kokkos_enable_option(UNSUPPORTED_ARCHS OFF "Whether to allow architectures in backends Kokkos doesn't optimize for")

function(check_device_specific_options)
  cmake_parse_arguments(SOME "" "DEVICE" "OPTIONS" ${ARGN})
  if(NOT KOKKOS_ENABLE_${SOME_DEVICE})
    foreach(OPTION ${SOME_OPTIONS})
      if(NOT DEFINED CACHE{Kokkos_ENABLE_${OPTION}} OR NOT DEFINED CACHE{Kokkos_ENABLE_${SOME_DEVICE}})
        message(FATAL_ERROR "Internal logic error: option '${OPTION}' or device '${SOME_DEVICE}' not recognized.")
      endif()
      if(KOKKOS_ENABLE_${OPTION})
        message(
          WARNING "Kokkos_ENABLE_${OPTION} is ON but ${SOME_DEVICE} backend is not enabled. Option will be ignored."
        )
        unset(KOKKOS_ENABLE_${OPTION} PARENT_SCOPE)
      endif()
    endforeach()
  endif()
endfunction()

check_device_specific_options(
  DEVICE
  CUDA
  OPTIONS
  CUDA_UVM
  CUDA_RELOCATABLE_DEVICE_CODE
  CUDA_LAMBDA
  CUDA_CONSTEXPR
  CUDA_LDG_INTRINSIC
  IMPL_CUDA_MALLOC_ASYNC
  IMPL_CUDA_UNIFIED_MEMORY
)
check_device_specific_options(
  DEVICE HIP OPTIONS HIP_RELOCATABLE_DEVICE_CODE HIP_MULTIPLE_KERNEL_INSTANTIATIONS IMPL_HIP_UNIFIED_MEMORY
)
check_device_specific_options(DEVICE HPX OPTIONS IMPL_HPX_ASYNC_DISPATCH)
check_device_specific_options(DEVICE OPENACC OPTIONS OPENACC_FORCE_HOST_AS_DEVICE)

# Needed due to change from deprecated name to new header define name
if(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION)
  set(KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION ON)
endif()

# Force consistency of KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
# and CMAKE_CUDA_SEPARABLE_COMPILATION when we are compiling
# using the CMake CUDA language support.
# Either one being on will turn the other one on.
if(KOKKOS_COMPILE_LANGUAGE STREQUAL CUDA)
  if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
    if(NOT CMAKE_CUDA_SEPARABLE_COMPILATION)
      message(
        STATUS
          "Setting CMAKE_CUDA_SEPARABLE_COMPILATION=ON since Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE is true. When compiling Kokkos with CMake language CUDA, please use CMAKE_CUDA_SEPARABLE_COMPILATION to control RDC support"
      )
      set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    endif()
  else()
    if(CMAKE_CUDA_SEPARABLE_COMPILATION)
      set(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON)
    endif()
  endif()
endif()

# This is known to occur with Clang 9. We would need to use nvcc as the linker
# http://lists.llvm.org/pipermail/cfe-dev/2018-June/058296.html
# TODO: Through great effort we can use a different linker by hacking
# CMAKE_CXX_LINK_EXECUTABLE in a future release
if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE AND KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  message(
    FATAL_ERROR "Relocatable device code is currently not supported with Clang - must use nvcc_wrapper or turn off RDC"
  )
endif()

if(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Relocatable device code requires static libraries.")
endif()

if(Kokkos_ENABLE_CUDA_LDG_INTRINSIC)
  if(KOKKOS_ENABLE_DEPRECATED_CODE_4)
    message(DEPRECATION "Setting Kokkos_ENABLE_CUDA_LDG_INTRINSIC is deprecated. LDG intrinsics are always enabled.")
  else()
    message(FATAL_ERROR "Kokkos_ENABLE_CUDA_LDG_INTRINSIC has been removed. LDG intrinsics are always enabled.")
  endif()
endif()
if(Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_CUDA_LAMBDA)
  if(KOKKOS_ENABLE_DEPRECATED_CODE_4)
    message(
      DEPRECATION
        "Setting Kokkos_ENABLE_CUDA_LAMBDA is deprecated. Lambda expressions in device code are always enabled. Forcing -DKokkos_ENABLE_CUDA_LAMBDA=ON"
    )
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Kokkos turned Cuda lambda support ON!" FORCE)
    set(KOKKOS_ENABLE_CUDA_LAMBDA ON)
  else()
    message(FATAL_ERROR "Kokkos_ENABLE_CUDA_LAMBDA has been removed. Lambda expressions in device code always enabled.")
  endif()
endif()

if(DEFINED Kokkos_ENABLE_IMPL_DESUL_ATOMICS)
  message(WARNING "Kokkos_ENABLE_IMPL_DESUL_ATOMICS option has been removed. Desul atomics cannot be disabled.")
endif()
