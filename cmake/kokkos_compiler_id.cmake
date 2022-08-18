KOKKOS_CFG_DEPENDS(COMPILER_ID NONE)

SET(KOKKOS_CXX_COMPILER ${CMAKE_CXX_COMPILER})
SET(KOKKOS_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
SET(KOKKOS_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})

MACRO(kokkos_internal_have_compiler_nvcc)
  # Check if the compiler is nvcc (which really means nvcc_wrapper).
  EXECUTE_PROCESS(COMMAND ${ARGN} --version
                  OUTPUT_VARIABLE INTERNAL_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  STRING(REPLACE "\n" " - " INTERNAL_COMPILER_VERSION_ONE_LINE ${INTERNAL_COMPILER_VERSION} )
  STRING(FIND ${INTERNAL_COMPILER_VERSION_ONE_LINE} "nvcc" INTERNAL_COMPILER_VERSION_CONTAINS_NVCC)
  STRING(REGEX REPLACE "^ +" "" INTERNAL_HAVE_COMPILER_NVCC "${INTERNAL_HAVE_COMPILER_NVCC}")
  IF(${INTERNAL_COMPILER_VERSION_CONTAINS_NVCC} GREATER -1)
    SET(INTERNAL_HAVE_COMPILER_NVCC true)
  ELSE()
    SET(INTERNAL_HAVE_COMPILER_NVCC false)
  ENDIF()
ENDMACRO()

IF(Kokkos_ENABLE_CUDA)
  # kokkos_enable_options is not yet called so use lower case here
  IF(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
    kokkos_internal_have_compiler_nvcc(${CMAKE_CUDA_COMPILER})
  ELSE()
    # find kokkos_launch_compiler
    FIND_PROGRAM(Kokkos_COMPILE_LAUNCHER
        NAMES           kokkos_launch_compiler
        HINTS           ${PROJECT_SOURCE_DIR}
        PATHS           ${PROJECT_SOURCE_DIR}
        PATH_SUFFIXES   bin)

    FIND_PROGRAM(Kokkos_NVCC_WRAPPER
        NAMES           nvcc_wrapper
        HINTS           ${PROJECT_SOURCE_DIR}
        PATHS           ${PROJECT_SOURCE_DIR}
        PATH_SUFFIXES   bin)

    # Check if compiler was set to nvcc_wrapper
    kokkos_internal_have_compiler_nvcc(${CMAKE_CXX_COMPILER})
    # If launcher was found and nvcc_wrapper was not specified as
    # compiler and `CMAKE_CXX_COMPILIER_LAUNCHER` is not set, set to use launcher.
    # Will ensure CMAKE_CXX_COMPILER is replaced by nvcc_wrapper
    IF(Kokkos_COMPILE_LAUNCHER AND NOT INTERNAL_HAVE_COMPILER_NVCC AND NOT KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
      IF(CMAKE_CXX_COMPILER_LAUNCHER)
       MESSAGE(FATAL_ERROR "Cannot use CMAKE_CXX_COMPILER_LAUNCHER if the CMAKE_CXX_COMPILER is not able to compile CUDA code, i.e. nvcc_wrapper or
clang++!")
      ENDIF()
      # the first argument to launcher is always the C++ compiler defined by cmake
      # if the second argument matches the C++ compiler, it forwards the rest of the
      # args to nvcc_wrapper
      kokkos_internal_have_compiler_nvcc(
        ${Kokkos_COMPILE_LAUNCHER} ${Kokkos_NVCC_WRAPPER} ${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER} -DKOKKOS_DEPENDENCE)
      SET(INTERNAL_USE_COMPILER_LAUNCHER true)
    ENDIF()
  ENDIF()
ENDIF()

IF(INTERNAL_HAVE_COMPILER_NVCC)
  # Save the host compiler id before overwriting it.
  SET(KOKKOS_CXX_HOST_COMPILER_ID ${KOKKOS_CXX_COMPILER_ID})

  # SET the compiler id to nvcc.  We use the value used by CMake 3.8.
  SET(KOKKOS_CXX_COMPILER_ID NVIDIA CACHE STRING INTERNAL FORCE)

  STRING(REGEX MATCH "V[0-9]+\\.[0-9]+\\.[0-9]+"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_COMPILER_VERSION_ONE_LINE})
  STRING(SUBSTRING ${TEMP_CXX_COMPILER_VERSION} 1 -1 TEMP_CXX_COMPILER_VERSION)
  SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  MESSAGE(STATUS "Compiler Version: ${KOKKOS_CXX_COMPILER_VERSION}")
  IF(INTERNAL_USE_COMPILER_LAUNCHER)
    MESSAGE(STATUS "kokkos_launch_compiler (${Kokkos_COMPILE_LAUNCHER}) is enabled...")
    kokkos_compilation(GLOBAL)
  ENDIF()
ENDIF()

IF(Kokkos_ENABLE_HIP)
  # get HIP version
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE INTERNAL_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  STRING(REPLACE "\n" " - " INTERNAL_COMPILER_VERSION_ONE_LINE ${INTERNAL_COMPILER_VERSION} )

  STRING(FIND ${INTERNAL_COMPILER_VERSION_ONE_LINE} "HIP version" INTERNAL_COMPILER_VERSION_CONTAINS_HIP)
  IF(INTERNAL_COMPILER_VERSION_CONTAINS_HIP GREATER -1)
    SET(KOKKOS_CXX_COMPILER_ID HIPCC CACHE STRING INTERNAL FORCE)
  ENDIF()

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_COMPILER_VERSION_ONE_LINE})
  SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  MESSAGE(STATUS "Compiler Version: ${KOKKOS_CXX_COMPILER_VERSION}")
ENDIF()

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c Cray
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    SET(KOKKOS_CLANG_IS_CRAY TRUE)
  ENDIF()
  # The clang based Intel compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c "DPC++\\|icpx"
                  OUTPUT_VARIABLE INTERNAL_HAVE_INTEL_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_INTEL_COMPILER) #not actually Clang
    SET(KOKKOS_CLANG_IS_INTEL TRUE)
    SET(KOKKOS_CXX_COMPILER_ID IntelLLVM CACHE STRING INTERNAL FORCE)
    EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
    STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
           KOKKOS_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  ENDIF()
ENDIF()

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Cray OR KOKKOS_CLANG_IS_CRAY)
  # SET Cray's compiler version.
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  IF (KOKKOS_CLANG_IS_CRAY)
    SET(KOKKOS_CLANG_CRAY_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION})
  ELSE()
    SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  ENDIF()
ENDIF()

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Fujitsu)
  # SET Fujitsus compiler version which is not detected by CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
ENDIF()

# Enforce the minimum compilers supported by Kokkos.
SET(KOKKOS_MESSAGE_TEXT "Compiler not supported by Kokkos.  Required compiler versions:")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang(CPU)         8.0.0 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang(CUDA)       10.0.0 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    GCC                8.2.0 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Intel             19.0.5 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    IntelLLVM(CPU)  2021.1.1 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    IntelLLVM(SYCL) 2022.0.0 or higher") #FIXME
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVCC              11.0.0 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    HIPCC              4.5.0 or higher") #FIXME
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVHPC/PGI           22.3 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    MSVC               19.29 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    XL/XLClang          none supported")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\nCompiler: ${KOKKOS_CXX_COMPILER_ID} ${KOKKOS_CXX_COMPILER_VERSION}\n")

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND NOT Kokkos_ENABLE_CUDA)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 8.0.0)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang AND Kokkos_ENABLE_CUDA)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 10.0.0)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL GNU)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 8.2.0)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 19.0.5)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND NOT Kokkos_ENABLE_SYCL)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 2021.1.1)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL IntelLLVM AND Kokkos_ENABLE_SYCL)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 2022.0.0) #FIXME 2022.2.0
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 11.0.0)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
  SET(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Kokkos turns off CXX extensions" FORCE)
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL HIPCC)
  # Note that ROCm 4.5 reports as version 4.4
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.4.0) #FIXME ROCm 5.2
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI OR KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 22.3)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
  # Treat PGI internally as NVHPC to simplify handling both compilers.
  # Before CMake 3.20 NVHPC was identified as PGI, nvc++ is
  # backward-compatible to pgc++.
  SET(KOKKOS_CXX_COMPILER_ID NVHPC CACHE STRING INTERNAL FORCE)
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL "MSVC")
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 19.29)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL XL OR KOKKOS_CXX_COMPILER_ID STREQUAL XLClang)
  MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
ENDIF()

IF(NOT DEFINED KOKKOS_CXX_HOST_COMPILER_ID)
  SET(KOKKOS_CXX_HOST_COMPILER_ID ${KOKKOS_CXX_COMPILER_ID})
ELSEIF(KOKKOS_CXX_HOST_COMPILER_ID STREQUAL PGI)
  SET(KOKKOS_CXX_HOST_COMPILER_ID NVHPC CACHE STRING INTERNAL FORCE)
ENDIF()

STRING(REPLACE "." ";" VERSION_LIST ${KOKKOS_CXX_COMPILER_VERSION})
LIST(GET VERSION_LIST 0 KOKKOS_COMPILER_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 KOKKOS_COMPILER_VERSION_MINOR)
LIST(GET VERSION_LIST 2 KOKKOS_COMPILER_VERSION_PATCH)
