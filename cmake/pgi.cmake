
FUNCTION(KOKKOS_SET_PGI_FLAGS STANDARD)
  STRING(TOLOWER ${STANDARD} LC_STANDARD)
  GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "--c++${LC_STANDARD}")
ENDFUNCTION()

