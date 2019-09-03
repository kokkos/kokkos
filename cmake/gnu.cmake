
FUNCTION(KOKKOS_SET_GNU_FLAGS STANDARD)
  STRING(TOLOWER ${STANDARD} LC_STANDARD)
  # The following three blocks of code were copied from
  # /Modules/Compiler/Intel-CXX.cmake from CMake 3.7.2 and then modified.
  IF(CMAKE_CXX_SIMULATE_ID STREQUAL MSVC)
    SET(_std -Qstd)
    SET(_ext c++)
  ELSE()
    SET(_std -std)
    SET(_ext gnu++)
  ENDIF()

  IF (CMAKE_CXX_EXTENSIONS)
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "-std=gnu++${LC_STANDARD}")
  ELSE()
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "-std=c++${LC_STANDARD}")
  ENDIF()
ENDFUNCTION()

