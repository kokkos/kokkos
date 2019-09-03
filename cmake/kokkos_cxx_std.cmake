KOKKOS_CFG_DEPENDS(CXX_STD COMPILER_ID)

FUNCTION(KOKKOS_SET_CXX_STANDARD_FEATURE STANDARD)
  SET(EXTENSION_NAME CMAKE_CXX${STANDARD}_EXTENSION_COMPILE_OPTION)
  SET(STANDARD_NAME  CMAKE_CXX${STANDARD}_STANDARD_COMPILE_OPTION)
  SET(FEATURE_NAME   cxx_std_${STANDARD})
  #message("HAVE ${FEATURE_NAME} ${${EXTENSION_NAME}} ${${STANDARD_NAME}}")
  #CMake's way of telling us that the standard (or extension)
  #flags are supported is the extension/standard variables
  IF (NOT DEFINED CMAKE_CXX_EXTENSIONS)
    IF(KOKKOS_DONT_ALLOW_EXTENSIONS)
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS OFF)
    ELSE()
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    IF(KOKKOS_DONT_ALLOW_EXTENSIONS)
      MESSAGE(FATAL_ERROR "The chosen configuration does not support CXX extensions flags: ${KOKKOS_DONT_ALLOW_EXTENSIONS}. Must set CMAKE_CXX_EXTENSIONS=OFF to continue") 
    ELSE()
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSE()
    GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS OFF)
  ENDIF()

  IF (KOKKOS_USE_CXX_EXTENSIONS AND ${EXTENSION_NAME})
    MESSAGE(STATUS "Using ${${EXTENSION_NAME}} for C++${STANDARD} extensions as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSEIF(NOT KOKKOS_USE_CXX_EXTENSIONS AND ${STANDARD_NAME})
    MESSAGE(STATUS "Using ${${STANDARD_NAME}} for C++${STANDARD} standard as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSE()
    #nope, we can't do anything here
    MESSAGE(STATUS "C++${STANDARD} is not supported as a compiler feature - choosing custom flags")
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
ENDFUNCTION()




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

#This can run at any time
KOKKOS_OPTION(CXX_STANDARD "" STRING "The C++ standard for Kokkos to use: 11, 14, or 17. In some cases, intermediate standards 1Y, 1Z, or 2A are also supported.")

# Set CXX standard flags
SET(KOKKOS_ENABLE_CXX11 OFF)
SET(KOKKOS_ENABLE_CXX14 OFF)
SET(KOKKOS_ENABLE_CXX17 OFF)
SET(KOKKOS_ENABLE_CXX20 OFF)
IF (KOKKOS_CXX_STANDARD)
  IF (${KOKKOS_CXX_STANDARD} STREQUAL "c++98")
    MESSAGE(FATAL_ERROR "Kokkos no longer supports C++98 - minimum C++11")
  ELSEIF (${KOKKOS_CXX_STANDARD} STREQUAL "c++11")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++11'. Use '11' instead.")
    SET(KOKKOS_CXX_STANDARD "11")
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++14")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++14'. Use '14' instead.")
    SET(KOKKOS_CXX_STANDARD "14")
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++17")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++17'. Use '17' instead.")
    SET(KOKKOS_CXX_STANDARD "17")
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++1y")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++1y'. Use '1Y' instead.")
    SET(KOKKOS_CXX_STANDARD "1Y")
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++1z")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++1z'. Use '1Z' instead.")
    SET(KOKKOS_CXX_STANDARD "1Z")
  ELSEIF(${KOKKOS_CXX_STANDARD} STREQUAL "c++2a")
    MESSAGE(WARNING "Deprecated Kokkos C++ standard set as 'c++2a'. Use '2A' instead.")
    SET(KOKKOS_CXX_STANDARD "2A")
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD AND NOT CMAKE_CXX_STANDARD)
  MESSAGE(STATUS "Setting default Kokkos CXX standard to 11")
  SET(KOKKOS_CXX_STANDARD "11")
  SET(CMAKE_CXX_STANDARD "11")
ELSEIF(NOT KOKKOS_CXX_STANDARD)
  MESSAGE(STATUS "Setting default Kokkos CXX standard to ${CMAKE_CXX_STANDARD}")
  SET(KOKKOS_CXX_STANDARD ${CMAKE_CXX_STANDARD})
ELSEIF(NOT CMAKE_CXX_STANDARD)
  SET(CMAKE_CXX_STANDARD ${KOKKOS_CXX_STANDARD})
ENDIF()


IF (KOKKOS_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  #make sure these are consistent
  IF (NOT KOKKOS_CXX_STANDARD STREQUAL CMAKE_CXX_STANDARD)
    MESSAGE(WARNING "Specified both CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} and KOKKOS_CXX_STANDARD=${KOKKOS_CXX_STANDARD}, but they don't match")
    SET(CMAKE_CXX_STANDARD ${KOKKOS_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
  ENDIF()
ENDIF()

IF (KOKKOS_CXX_STANDARD STREQUAL "11" )
  KOKKOS_SET_CXX_STANDARD_FEATURE(11)
  GLOBAL_SET(KOKKOS_ENABLE_CXX11 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "14")
  KOKKOS_SET_CXX_STANDARD_FEATURE(14)
  GLOBAL_SET(KOKKOS_ENABLE_CXX14 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "17")
  KOKKOS_SET_CXX_STANDARD_FEATURE(17)
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



# Enforce that extensions are turned off for nvcc_wrapper.
# For compiling CUDA code using nvcc_wrapper, we will use the host compiler's
# flags for turning on C++11.  Since for compiler ID and versioning purposes
# CMake recognizes the host compiler when calling nvcc_wrapper, this just
# works.  Both NVCC and nvcc_wrapper only recognize '-std=c++11' which means
# that we can only use host compilers for CUDA builds that use those flags.
# It also means that extensions (gnu++11) can't be turned on for CUDA builds.

IF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    SET(CMAKE_CXX_EXTENSIONS OFF)
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    MESSAGE(FATAL_ERROR "NVCC doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
  ENDIF()
ENDIF()

IF(KOKKOS_ENABLE_CUDA)
  # ENFORCE that the compiler can compile CUDA code.
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
    IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.0.0)
      MESSAGE(FATAL_ERROR "Compiling CUDA code directly with Clang requires version 4.0.0 or higher.")
    ENDIF()
    IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
      SET(CMAKE_CXX_EXTENSIONS OFF)
    ELSEIF(CMAKE_CXX_EXTENSIONS)
      MESSAGE(FATAL_ERROR "Compiling CUDA code with clang doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
    ENDIF()
  ELSEIF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    MESSAGE(FATAL_ERROR "Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang, but compiler ID was ${KOKKOS_CXX_COMPILER_ID}")
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD_FEATURE)
  UNSET(CMAKE_CXX_STANDARD CACHE) #don't let cmake do this as a feature either
  #we need to pick the C++ flags ourselves
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Cray)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/cray.cmake)
    KOKKOS_SET_CRAY_FLAGS(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/pgi.cmake)
    KOKKOS_SET_PGI_FLAGS(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/intel.cmake)
    KOKKOS_SET_INTEL_FLAGS(${KOKKOS_CXX_STANDARD})
  ELSE()
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/gnu.cmake)
    KOKKOS_SET_GNU_FLAGS(${KOKKOS_CXX_STANDARD})
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




