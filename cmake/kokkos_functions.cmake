################################### FUNCTIONS ##################################
# List of functions
#   set_kokkos_cxx_compiler
#   set_kokkos_cxx_standard
#   set_kokkos_srcs

#-------------------------------------------------------------------------------
# function(set_kokkos_cxx_compiler)
# Sets the following compiler variables that are analogous to the CMAKE_*
# versions.  We add the ability to detect NVCC (really nvcc_wrapper).
#   KOKKOS_CXX_COMPILER
#   KOKKOS_CXX_COMPILER_ID
#   KOKKOS_CXX_COMPILER_VERSION
#
# Inputs:
#   KOKKOS_ENABLE_CUDA
#   CMAKE_CXX_COMPILER
#   CMAKE_CXX_COMPILER_ID
#   CMAKE_CXX_COMPILER_VERSION
#
# Also verifies the compiler version meets the minimum required by Kokkos.
function(set_kokkos_cxx_compiler)
  # Since CMake doesn't recognize the nvcc compiler until 3.8, we use our own
  # version of the CMake variables and detect nvcc ourselves.  Initially set to
  # the CMake variable values.
  set(INTERNAL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  set(INTERNAL_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
  set(INTERNAL_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})

  # Check if the compiler is nvcc (which really means nvcc_wrapper).
  execute_process(COMMAND ${INTERNAL_CXX_COMPILER} --version
                  COMMAND grep nvcc
                  COMMAND wc -l
                  OUTPUT_VARIABLE INTERNAL_HAVE_COMPILER_NVCC
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX REPLACE "^ +" ""
         INTERNAL_HAVE_COMPILER_NVCC ${INTERNAL_HAVE_COMPILER_NVCC})

  if(INTERNAL_HAVE_COMPILER_NVCC)
    # Set the compiler id to nvcc.  We use the value used by CMake 3.8.
    set(INTERNAL_CXX_COMPILER_ID NVIDIA)

    # Set nvcc's compiler version.
    execute_process(COMMAND ${INTERNAL_CXX_COMPILER} --version
                    COMMAND grep release
                    OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+$"
           INTERNAL_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  endif()

  # Enforce the minimum compilers supported by Kokkos.
  set(KOKKOS_MESSAGE_TEXT "Compiler not supported by Kokkos.  Required compiler versions:")
  set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang      3.5.2 or higher")
  set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    GCC        4.8.4 or higher")
  set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Intel     15.0.2 or higher")
  set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVCC      7.0.28 or higher")
  set(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    PGI         17.1 or higher\n")

  if(INTERNAL_CXX_COMPILER_ID STREQUAL Clang)
    if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 3.5.2)
      message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
    endif()
  elseif(INTERNAL_CXX_COMPILER_ID STREQUAL GNU)
    if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 4.8.4)
      message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
    endif()
  elseif(INTERNAL_CXX_COMPILER_ID STREQUAL Intel)
    if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 15.0.2)
      message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
    endif()
  elseif(INTERNAL_CXX_COMPILER_ID STREQUAL NVIDIA)
    if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 7.0.28)
      message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
    endif()
  elseif(INTERNAL_CXX_COMPILER_ID STREQUAL PGI)
    if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 17.1)
      message(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
    endif()
  endif()

  # Enforce that extensions are turned off for nvcc_wrapper.
  if(INTERNAL_CXX_COMPILER_ID STREQUAL NVIDIA)
    if(DEFINED CMAKE_CXX_EXTENSIONS AND CMAKE_CXX_EXTENSIONS STREQUAL ON)
      message(FATAL_ERROR "NVCC doesn't support C++ extensions.  Set CMAKE_CXX_EXTENSIONS to OFF in your CMakeLists.txt.")
    endif()
  endif()

  if(KOKKOS_ENABLE_CUDA)
    # Enforce that the compiler can compile CUDA code.
    if(INTERNAL_CXX_COMPILER_ID STREQUAL Clang)
      if(INTERNAL_CXX_COMPILER_VERSION VERSION_LESS 4.0.0)
        message(FATAL_ERROR "Compiling CUDA code directly with Clang requires version 4.0.0 or higher.")
      endif()
    elseif(NOT INTERNAL_CXX_COMPILER_ID STREQUAL NVIDIA)
      message(FATAL_ERROR "Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang, but compiler ID was ${INTERNAL_CXX_COMPILER_ID}")
    endif()
  endif()

  set(KOKKOS_CXX_COMPILER ${INTERNAL_CXX_COMPILER} PARENT_SCOPE)
  set(KOKKOS_CXX_COMPILER_ID ${INTERNAL_CXX_COMPILER_ID} PARENT_SCOPE)
  set(KOKKOS_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# function(set_kokkos_cxx_standard)
#  Transitively enforces that the appropriate CXX standard compile flags (C++11
#  or above) are added to targets that use the Kokkos library.  Compile features
#  are used if possible.  Otherwise, the appropriate flags are added to
#  KOKKOS_CXX_FLAGS.  Values set by the user to CMAKE_CXX_STANDARD and
#  CMAKE_CXX_EXTENSIONS are honored.
#
# Outputs:
#   KOKKOS_CXX11_FEATURES
#   KOKKOS_CXX_FLAGS
#
# Inputs:
#  KOKKOS_CXX_COMPILER
#  KOKKOS_CXX_COMPILER_ID
#  KOKKOS_CXX_COMPILER_VERSION
#
function(set_kokkos_cxx_standard)
  # The following table lists the versions of CMake that supports CXX_STANDARD
  # and the CXX compile features for different compilers.  The versions are
  # based on CMake documentation, looking at CMake code, and verifying by
  # testing with specific CMake versions.
  #
  #   COMPILER                      CXX_STANDARD     Compile Features
  #   ---------------------------------------------------------------
  #   Clang                             3.1                3.1
  #   GNU                               3.1                3.2
  #   AppleClang                        3.2                3.2
  #   Intel                             3.6                3.6
  #   Cray                              No                 No
  #   PGI                               No                 No
  #   XL                                No                 No
  #
  # Kokkos now requires a minimum of CMake 3.8
  # For compiling CUDA code using nvcc_wrapper, we will use the host compiler's
  # flags for turning on C++11.  Since for compiler ID and versioning purposes
  # CMake recognizes the host compiler when calling nvcc_wrapper, this just
  # works.  Both NVCC and nvcc_wrapper only recognize '-std=c++11' which means
  # that we can only use host compilers for CUDA builds that use those flags.
  # It also means that extensions (gnu++11) can't be turned on for CUDA builds.


  # Check if we can use compile features.
  SET(VALID_FOR_FEATURES Clang GNU Intel AppleClang)
  #always valid for certain compilers
  IF(${KOKKOS_CXX_COMPILER_ID} IN_LIST VALID_FOR_FEATURES)
    set(KOKKOS_CXX_STANDARD_IS_FEATURE ON CACHE INTERNAL 
      "Whether the compiler family supports target_compile_features")
    return()
  ENDIF()

  set(KOKKOS_CXX_STANDARD_IS_FEATURE OFF CACHE INTERNAL 
    "Whether the compiler family supports target_compile_features")
  if(CMAKE_CXX_COMPILER_ID STREQUAL Cray)
    # CMAKE doesn't support CXX_STANDARD or C++ compile features for the Cray
    # compiler.  Set compiler options transitively here such that they trickle
    # down to a call to target_compile_options().
    set(KOKKOS_CXX_STANDARD_FLAG "-hstd=c++11" CACHE INTERNAL 
      "The flags needed for the C++ standard, if not supported as feature")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL PGI)
    # CMAKE doesn't support CXX_STANDARD or C++ compile features for the PGI
    # compiler.  Set compiler options transitively here such that they trickle
    # down to a call to target_compile_options().
    set(KOKKOS_CXX_STANDARD_FLAG "--c++11" CACHE INTERNAL 
      "The flags needed for the C++ standard, if not supported as feature")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL XL)
    # CMAKE doesn't support CXX_STANDARD or C++ compile features for the XL
    # compiler.  Set compiler options transitively here such that they trickle
    # down to a call to target_compile_options().
    set(KOKKOS_CXX_STANDARD_FLAG "-std=c++11" CACHE INTERNAL 
      "The flags needed for the C++ standard, if not supported as feature")
  else()
    message(FATAL_ERROR "Got unknown compiler ${KOKKOS_COMPILER_ID}")
  endif()
endfunction()



