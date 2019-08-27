
function(kokkos_set_pgi_flags standard)
  STRING(TOLOWER ${standard} LC_STANDARD)
  GLOBAL_SET(KOKKOS_CXX_STANDARD_FLAG "--c++${LC_STANDARD}")
endfunction()

