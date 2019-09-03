
FUNCTION(KOKKOS_SET_INTEL_FLAGS STANDARD)
  STRING(TOLOWER ${STANDARD} LC_STANDARD)
  # The following three blocks of code were copied from
  # /Modules/Compiler/Intel-CXX.cmake from CMake 3.7.2 and then modified.
  if(CMAKE_CXX_SIMULATE_ID STREQUAL MSVC)
    set(_std -Qstd)
    set(_ext c++)
  else()
    set(_std -std)
    set(_ext gnu++)
  endif()

  if(NOT KOKKOS_CXX_STANDARD STREQUAL 11 AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0.2)
    #There is no gnu++14 value supported; figure out what to do.
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "${_std}=c++${LC_STANDARD}")
  elseif(KOKKOS_CXX_STANDARD STREQUAL 11 AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
    IF (CMAKE_CXX_EXTENSIONS)
      GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "${_std}=${_ext}c++11")
    ELSE()
      GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "${_std}=c++11")
    ENDIF()
  else()
    message(FATAL_ERROR "Intel compiler version too low - need 13.0 for C++11 and 15.0 for C++14")
  endif()

ENDFUNCTION()

