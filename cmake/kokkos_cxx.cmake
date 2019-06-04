
FUNCTION(kokkos_set_cxx_standard_feature standard)
  SET(EXTENSION_NAME CMAKE_CXX${standard}_EXTENSION_COMPILE_OPTION)
  SET(STANDARD_NAME  CMAKE_CXX${standard}_STANDARD_COMPILE_OPTION)
  SET(FEATURE_NAME   cxx_std_${standard})
  #message("HAVE ${FEATURE_NAME} ${${EXTENSION_NAME}} ${${STANDARD_NAME}}")
  #CMake's way of telling us that the standard (or extension)
  #flags are supported is the extension/standard variables
  IF (CMAKE_CXX_EXTENSIONS AND ${EXTENSION_NAME})
    MESSAGE(STATUS "Using ${${FEATURE_NAME}} for C++${standard} extensions as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSEIF(NOT CMAKE_CXX_EXTENSIONS AND ${STANDARD_NAME})
    MESSAGE(STATUS "Using ${${STANDARD_NAME}} for C++${standard} standard as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSE()
    #nope, we can't do anything here
    MESSAGE(STATUS "C++${standard} is not supported as a compiler feature - choosing custom flags")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE "")
  ENDIF()

  IF(NOT ${FEATURE_NAME} IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    IF (KOKKOS_CXX_COMPILER_ID STREQUAL "NVIDIA")
      MESSAGE(STATUS "nvcc_wrapper does not support TARGET_COMPILE_FEATURES")
      GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE "")
    ELSE()
      MESSAGE(FATAL_ERROR "Compiler ${KOKKOS_CXX_COMPILER_ID} should support ${FEATURE_NAME}, but CMake reports feature not supported")
    ENDIF()
  ENDIF()
ENDFUNCTION(kokkos_set_cxx_standard_feature)

SET(KOKKOS_CXX_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING INTERNAL)
SET(KOKKOS_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID} CACHE STRING INTERNAL)
SET(KOKKOS_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION} CACHE STRING INTERNAL)

# Check if the compiler is nvcc (which really means nvcc_wrapper).
EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                COMMAND grep nvcc
                COMMAND wc -l
                OUTPUT_VARIABLE INTERNAL_HAVE_COMPILER_NVCC
                OUTPUT_STRIP_TRAILING_WHITESPACE)


STRING(REGEX REPLACE "^ +" ""
       INTERNAL_HAVE_COMPILER_NVCC ${INTERNAL_HAVE_COMPILER_NVCC})


IF(INTERNAL_HAVE_COMPILER_NVCC)
  # SET the compiler id to nvcc.  We use the value used by CMake 3.8.
  SET(KOKKOS_CXX_COMPILER_ID NVIDIA CACHE STRING INTERNAL FORCE)

  # SET nvcc's compiler version.
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep release
                  OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+$"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL)
ENDIF()

# Enforce the minimum compilers supported by Kokkos.
SET(KOKKOS_MESSAGE_TEXT "Compiler not supported by Kokkos.  Required compiler versions:")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang      3.5.2 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    GCC        4.8.4 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Intel     15.0.2 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVCC      7.0.28 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    PGI         17.1 or higher\n")

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 3.5.2)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL GNU)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.8.4)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 15.0.2)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 7.0.28)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 17.1)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ENDIF()

# Enforce that extensions are turned off for nvcc_wrapper.
# For compiling CUDA code using nvcc_wrapper, we will use the host compiler's
# flags for turning on C++11.  Since for compiler ID and versioning purposes
# CMake recognizes the host compiler when calling nvcc_wrapper, this just
# works.  Both NVCC and nvcc_wrapper only recognize '-std=c++11' which means
# that we can only use host compilers for CUDA builds that use those flags.
# It also means that extensions (gnu++11) can't be turned on for CUDA builds.
IF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(CMAKE_CXX_EXTENSIONS)
    MESSAGE(FATAL_ERROR "NVCC doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
  ENDIF()
ENDIF()

IF(KOKKOS_ENABLE_CUDA)
  # ENFORCE that the compiler can compile CUDA code.
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
    IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.0.0)
      MESSAGE(FATAL_ERROR "Compiling CUDA code directly with Clang requires version 4.0.0 or higher.")
    ENDIF()
  ELSEIF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    MESSAGE(FATAL_ERROR "Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang, but compiler ID was ${KOKKOS_CXX_COMPILER_ID}")
  ENDIF()
ENDIF()


# From CMake 3.10 documentation
#CMake is currently aware of the C++ standards compiler features

#AppleClang: Apple Clang for Xcode versions 4.4 though 6.2.
#Clang: Clang compiler versions 2.9 through 3.4.
#GNU: GNU compiler versions 4.4 through 5.0.
#MSVC: Microsoft Visual Studio versions 2010 through 2017.
#SunPro: Oracle SolarisStudio versions 12.4 through 12.5.
#Intel: Intel compiler versions 12.1 through 17.0.
#
#Cray: Cray Compiler Environment version 8.1 through 8.5.8.
#PGI: PGI version 12.10 through 17.5.
#XL: IBM XL version 10.1 through 13.1.5.

kokkos_option(CXX_STANDARD "" STRING "The C++ standard for Kokkos to use: c++11, c++14, or c++17")


# Set CXX standard flags
SET(KOKKOS_ENABLE_CXX11 OFF CACHE INTERNAL "Enable C++11 flags")
SET(KOKKOS_ENABLE_CXX14 OFF CACHE INTERNAL "Enable C++14 flags")
SET(KOKKOS_ENABLE_CXX17 OFF CACHE INTERNAL "Enable C++17 flags")
SET(KOKKOS_ENABLE_CXX20 OFF CACHE INTERNAL "Enable C++20 flags")
IF (KOKKOS_CXX_STANDARD)
  IF (${KOKKOS_CXX_STANDARD} STREQUAL "c++98")
    MESSAGE(FATAL_ERROR "Kokkos no longer supports C++98 - minimum C++11")
  ELSEIF (${KOKKOS_CXX_STANDARD} STREQUAL "c++11")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++11'. Use '11' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "14" STRING)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++14")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++14'. Use '14' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "14" STRING)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++17")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++17'. Use '17' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "17" STRING)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++1y")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++1y'. Use '1Y' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "1Y" STRING)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++1z")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++1z'. Use '1Z' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "1Z" STRING)
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++2a")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++2a'. Use '2A' instead.")
    GLOBAL_OVERWRITE(KOKKOS_CXX_STANDARD "2A" STRING)
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD)
  MESSAGE(STATUS "Setting default Kokkos CXX standard to 11")
  SET(KOKKOS_CXX_STANDARD "11" CACHE STRING "C++ standard" FORCE)
  SET(CMAKE_CXX_STANDARD "11" CACHE STRING "C++ standard" FORCE)
ELSEIF(NOT KOKKOS_CXX_STANDARD)
  MESSAGE(STATUS "Setting default Kokkos CXX standard to ${CMAKE_CXX_STANDARD}")
  SET(KOKKOS_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
ELSEIF(NOT CMAKE_CXX_STANDARD)
  SET(CMAKE_CXX_STANDARD ${KOKKOS_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
ENDIF()


IF (KOKKOS_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  #make sure these are consistent
  IF (NOT KOKKOS_CXX_STANDARD STREQUAL CMAKE_CXX_STANDARD)
    MESSAGE(WARNING "Specified both CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} and KOKKOS_CXX_STANDARD=${KOKKOS_CXX_STANDARD}, but they don't match")
    SET(CMAKE_CXX_STANDARD ${KOKKOS_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
  ENDIF()
ENDIF()

get_property(CMAKE_CXX_KNOWN_FEATURES GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)

IF (KOKKOS_CXX_STANDARD STREQUAL "11" )
  kokkos_set_cxx_standard_feature(11)
  GLOBAL_SET(KOKKOS_ENABLE_CXX11 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "14")
  kokkos_set_cxx_standard_feature(14)
  GLOBAL_SET(KOKKOS_ENABLE_CXX14 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "17")
  kokkos_set_cxx_standard_feature(17)
  GLOBAL_SET(KOKKOS_ENABLE_CXX17 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "98")
  MESSAGE(FATAL_ERROR "Kokkos requires C++11 or newer!")
ELSE()
  #set to empty
  GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE "")
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL "NVIDIA")
    MESSAGE(FATAL_ERROR "nvcc_wrapper does not support intermediate standards (1Y,1Z,2A) - must use 11, 14, or 17")
  ENDIF()
  #okay, this is funky - kill this variable
  #this value is not really valid as a cmake variable
  UNSET(CMAKE_CXX_STANDARD CACHE)
  IF     (KOKKOS_CXX_STANDARD STREQUAL "1Y")
    GLOBAL_SET(KOKKOS_ENABLE_CXX14 ON)
  ELSEIF (KOKKOS_CXX_STANDARD STREQUAL "1Z")
    GLOBAL_SET(KOKKOS_ENABLE_CXX17 ON)
  ELSEIF (KOKKOS_CXX_STANDARD STREQUAL "2A")
    GLOBAL_SET(KOKKOS_ENABLE_CXX20 ON)
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD_FEATURE)
  UNSET(CMAKE_CXX_STANDARD CACHE) #don't let cmake do this as a feature either
  #we need to pick the C++ flags ourselves
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Cray)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/cray.cmake)
    kokkos_set_cray_flags(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/pgi.cmake)
    kokkos_set_pgi_flags(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/intel.cmake)
    kokkos_set_intel_flags(${KOKKOS_CXX_STANDARD})
  ELSE()
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/gnu.cmake)
    kokkos_set_gnu_flags(${KOKKOS_CXX_STANDARD})
  ENDIF()
  #check that the compiler accepts the C++ standard flag
  INCLUDE(CheckCXXCompilerFlag)
  IF (DEFINED CXX_STD_FLAGS_ACCEPTED)
    UNSET(CXX_STD_FLAGS_ACCEPTED CACHE)
  ENDIF()
  CHECK_CXX_COMPILER_FLAG(${KOKKOS_CXX_STANDARD_FLAG} CXX_STD_FLAGS_ACCEPTED)
  IF (NOT CXX_STD_FLAGS_ACCEPTED)
    MESSAGE(FATAL_ERROR "${KOKKOS_CXX_COMPILER_ID} did not accept ${KOKKOS_CXX_STANDARD_FLAG}. You likely need to reduce the level of the C++ standard from ${KOKKOS_CXX_STANDARD}")
  ELSE()
    MESSAGE(STATUS "Compiler features not supported, but ${KOKKOS_CXX_COMPILER_ID} accepts ${KOKKOS_CXX_STANDARD_FLAG}")
  ENDIF()
ENDIF()




