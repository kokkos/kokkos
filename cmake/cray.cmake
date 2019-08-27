

function(kokkos_set_cray_flags standard)
  STRING(TOLOWER ${standard} LC_STANDARD)
  GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "-hstd=c++${LC_STANDARD}")
endfunction()

