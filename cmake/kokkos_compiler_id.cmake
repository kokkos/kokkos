kokkos_cfg_depends(COMPILER_ID NONE)

set(KOKKOS_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(KOKKOS_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
set(KOKKOS_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
set(KOKKOS_BACKEND_COMPILER ${CMAKE_CXX_COMPILER})

macro(kokkos_internal_have_compiler_nvcc)
  # Check if the compiler is nvcc (which really means nvcc_wrapper).
  execute_process(COMMAND ${ARGN} --version OUTPUT_VARIABLE INTERNAL_COMPILER_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\n" " - " INTERNAL_COMPILER_VERSION_ONE_LINE ${INTERNAL_COMPILER_VERSION})
  string(FIND ${INTERNAL_COMPILER_VERSION_ONE_LINE} "nvcc" INTERNAL_COMPILER_VERSION_CONTAINS_NVCC)
  string(REGEX REPLACE "^ +" "" INTERNAL_HAVE_COMPILER_NVCC "${INTERNAL_HAVE_COMPILER_NVCC}")
  if(${INTERNAL_COMPILER_VERSION_CONTAINS_NVCC} GREATER -1)
    set(INTERNAL_HAVE_COMPILER_NVCC true)
  else()
    set(INTERNAL_HAVE_COMPILER_NVCC false)
  endif()
endmacro()

if(Kokkos_ENABLE_CUDA)
  # kokkos_enable_options is not yet called so use lower case here
  if(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
    kokkos_internal_have_compiler_nvcc(${CMAKE_CUDA_COMPILER})
  else()
    # find kokkos_launch_compiler
    find_program(
      Kokkos_COMPILE_LAUNCHER
      NAMES kokkos_launch_compiler
      HINTS ${PROJECT_SOURCE_DIR}
      PATHS ${PROJECT_SOURCE_DIR}
      PATH_SUFFIXES bin
    )

    find_program(
      Kokkos_NVCC_WRAPPER
      NAMES nvcc_wrapper
      HINTS ${PROJECT_SOURCE_DIR}
      PATHS ${PROJECT_SOURCE_DIR}
      PATH_SUFFIXES bin
    )

    # Check if compiler was set to nvcc_wrapper
    kokkos_internal_have_compiler_nvcc(${CMAKE_CXX_COMPILER})
    # If launcher was found and nvcc_wrapper was not specified as
    # compiler and `CMAKE_CXX_COMPILIER_LAUNCHER` is not set, set to use launcher.
    # Will ensure CMAKE_CXX_COMPILER is replaced by nvcc_wrapper
    if(Kokkos_COMPILE_LAUNCHER AND NOT INTERNAL_HAVE_COMPILER_NVCC AND NOT KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
      if(CMAKE_CXX_COMPILER_LAUNCHER)
        message(
          FATAL_ERROR
            "Cannot use CMAKE_CXX_COMPILER_LAUNCHER if the CMAKE_CXX_COMPILER is not able to compile CUDA code, i.e. nvcc_wrapper or clang++!"
        )
      endif()
      # the first argument to launcher is always the C++ compiler defined by cmake
      # if the second argument matches the C++ compiler, it forwards the rest of the
      # args to nvcc_wrapper
      kokkos_internal_have_compiler_nvcc(
        ${Kokkos_COMPILE_LAUNCHER} ${Kokkos_NVCC_WRAPPER} ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER}
        -DKOKKOS_DEPENDENCE
      )
      set(INTERNAL_USE_COMPILER_LAUNCHER true)
      set(KOKKOS_BACKEND_COMPILER ${Kokkos_NVCC_WRAPPER})
    endif()
  endif()
endif()

if(INTERNAL_HAVE_COMPILER_NVCC)
  # Save the host compiler id before overwriting it.
  set(KOKKOS_CXX_HOST_COMPILER_ID ${KOKKOS_CXX_COMPILER_ID})

  # SET the compiler id to nvcc.  We use the value used by CMake 3.8.
  set(KOKKOS_CXX_COMPILER_ID NVIDIA CACHE STRING INTERNAL FORCE)

  string(REGEX MATCH "V[0-9]+\\.[0-9]+\\.[0-9]+" TEMP_CXX_COMPILER_VERSION ${INTERNAL_COMPILER_VERSION_ONE_LINE})
  string(SUBSTRING ${TEMP_CXX_COMPILER_VERSION} 1 -1 TEMP_CXX_COMPILER_VERSION)
  set(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  message(STATUS "Compiler Version: ${KOKKOS_CXX_COMPILER_VERSION}")
  if(INTERNAL_USE_COMPILER_LAUNCHER)
    message(STATUS "kokkos_launch_compiler (${Kokkos_COMPILE_LAUNCHER}) is enabled...")
    kokkos_compilation(GLOBAL)
  endif()
endif()

if(Kokkos_ENABLE_HIP)
  # get HIP version
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTERNAL_COMPILER_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REPLACE "\n" " - " INTERNAL_COMPILER_VERSION_ONE_LINE ${INTERNAL_COMPILER_VERSION})

  string(FIND ${INTERNAL_COMPILER_VERSION_ONE_LINE} "HIP version" INTERNAL_COMPILER_VERSION_CONTAINS_HIP)
  if(INTERNAL_COMPILER_VERSION_CONTAINS_HIP GREATER -1)
    set(KOKKOS_CXX_COMPILER_ID HIPCC CACHE STRING INTERNAL FORCE)
  endif()

  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" TEMP_CXX_COMPILER_VERSION ${INTERNAL_COMPILER_VERSION_ONE_LINE})
  set(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  message(STATUS "Compiler Version: ${KOKKOS_CXX_COMPILER_VERSION}")
endif()

if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  # The Cray compiler reports as Clang to most versions of CMake
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    COMMAND grep -c Cray
    OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    set(KOKKOS_CLANG_IS_CRAY TRUE)
    set(KOKKOS_CXX_COMPILER_ID CrayClang)
  endif()
  # The clang based Intel compiler reports as Clang to most versions of CMake
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version
    COMMAND grep -c "DPC++\\|icpx"
    OUTPUT_VARIABLE INTERNAL_HAVE_INTEL_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(INTERNAL_HAVE_INTEL_COMPILER) #not actually Clang
    set(KOKKOS_CLANG_IS_INTEL TRUE)
    set(KOKKOS_CXX_COMPILER_ID IntelLLVM CACHE STRING INTERNAL FORCE)
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" KOKKOS_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  endif()
endif()

if(KOKKOS_CXX_COMPILER_ID STREQUAL Cray OR KOKKOS_CLANG_IS_CRAY)
  # SET Cray's compiler version.
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  if(KOKKOS_CLANG_IS_CRAY)
    set(KOKKOS_CLANG_CRAY_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION})
  else()
    set(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  endif()
endif()

if(KOKKOS_CXX_COMPILER_ID STREQUAL Fujitsu)
  # SET Fujitsus compiler version which is not detected by CMake
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  set(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
endif()

# Enforce the minimum compilers supported by Kokkos.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_CXX_STANDARD EQUAL 17)
  set(KOKKOS_CLANG_CPU_MINIMUM 8.0.0)
  set(KOKKOS_CLANG_CUDA_MINIMUM 10.0.0)
  set(KOKKOS_CLANG_OPENMPTARGET_MINIMUM 15.0.0)
  set(KOKKOS_GCC_MINIMUM 8.2.0)
  set(KOKKOS_INTEL_LLVM_CPU_MINIMUM 2021.1.1)
  set(KOKKOS_INTEL_LLVM_SYCL_MINIMUM 2024.2.1)
  set(KOKKOS_NVCC_MINIMUM 11.0.0)
  set(KOKKOS_HIPCC_MINIMUM 5.2.0)
  set(KOKKOS_NVHPC_MINIMUM 22.3)
  set(KOKKOS_MSVC_MINIMUM 19.29)
else()
  set(KOKKOS_CLANG_CPU_MINIMUM 14.0.0)
  set(KOKKOS_CLANG_CUDA_MINIMUM 14.0.0)
  set(KOKKOS_CLANG_OPENMPTARGET_MINIMUM 15.0.0)
  set(KOKKOS_GCC_MINIMUM 10.1.0)
  set(KOKKOS_INTEL_LLVM_CPU_MINIMUM 2022.0.0)
  set(KOKKOS_INTEL_LLVM_SYCL_MINIMUM 2024.2.1)
  set(KOKKOS_NVCC_MINIMUM 12.0.0)
  set(KOKKOS_HIPCC_MINIMUM 5.2.0)
  set(KOKKOS_NVHPC_MINIMUM 22.3)
  set(KOKKOS_MSVC_MINIMUM 19.30)
endif()

set(KOKKOS_MESSAGE_TEXT
    "Compiler not supported by Kokkos for C++${CMAKE_CXX_STANDARD}. Required minimum compiler versions:"
)
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang(CPU)          ${KOKKOS_CLANG_CPU_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang(CUDA)         ${KOKKOS_CLANG_CUDA_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang(OpenMPTarget) ${KOKKOS_CLANG_OPENMPTARGET_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    GCC                 ${KOKKOS_GCC_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Intel               not supported")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    IntelLLVM(CPU)      ${KOKKOS_INTEL_LLVM_CPU_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    IntelLLVM(SYCL)     ${KOKKOS_INTEL_LLVM_SYCL_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVCC                ${KOKKOS_NVCC_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    HIPCC               ${KOKKOS_HIPCC_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVHPC/PGI           ${KOKKOS_NVHPC_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    MSVC                ${KOKKOS_MSVC_MINIMUM}")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    XL/XLClang          not supported")
set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\nCompiler: ${KOKKOS_CXX_COMPILER_ID} ${KOKKOS_CXX_COMPILER_VERSION}\n")

if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND NOT Kokkos_ENABLE_CUDA)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_CLANG_CPU_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND Kokkos_ENABLE_CUDA)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_CLANG_CUDA_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL GNU)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_GCC_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
  message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND NOT Kokkos_ENABLE_SYCL)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_INTEL_LLVM_CPU_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND Kokkos_ENABLE_SYCL)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_INTEL_LLVM_SYCL_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_NVCC_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
  set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Kokkos turns off CXX extensions" FORCE)
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL HIPCC)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_HIPCC_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL PGI OR KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_NVHPC_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
  # Treat PGI internally as NVHPC to simplify handling both compilers.
  # Before CMake 3.20 NVHPC was identified as PGI, nvc++ is
  # backward-compatible to pgc++.
  set(KOKKOS_CXX_COMPILER_ID NVHPC CACHE STRING INTERNAL FORCE)
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL "MSVC")
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS ${KOKKOS_MSVC_MINIMUM})
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL XL OR KOKKOS_CXX_COMPILER_ID STREQUAL XLClang)
  message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
elseif(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND Kokkos_ENABLE_OPENMPTARGET)
  if(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS KOKKOS_CLANG_OPENMPTARGET_MINIMUM)
    message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  endif()
endif()

if(NOT DEFINED KOKKOS_CXX_HOST_COMPILER_ID)
  set(KOKKOS_CXX_HOST_COMPILER_ID ${KOKKOS_CXX_COMPILER_ID})
elseif(KOKKOS_CXX_HOST_COMPILER_ID STREQUAL PGI)
  set(KOKKOS_CXX_HOST_COMPILER_ID NVHPC CACHE STRING INTERNAL FORCE)
endif()

string(REPLACE "." ";" VERSION_LIST ${KOKKOS_CXX_COMPILER_VERSION})
list(GET VERSION_LIST 0 KOKKOS_COMPILER_VERSION_MAJOR)
list(GET VERSION_LIST 1 KOKKOS_COMPILER_VERSION_MINOR)
list(LENGTH VERSION_LIST LIST_LENGTH)

# On Android, the compiler doesn't have a patch version, just a major/minor
if(LIST_LENGTH GREATER 2)
  list(GET VERSION_LIST 2 KOKKOS_COMPILER_VERSION_PATCH)
else()
  set(KOKKOS_COMPILER_VERSION_PATCH 0)
endif()
