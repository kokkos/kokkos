function(kokkos_set_cray_flags full_standard int_standard)
  string(TOLOWER ${full_standard} FULL_LC_STANDARD)
  string(TOLOWER ${int_standard} INT_LC_STANDARD)
  set(KOKKOS_CXX_STANDARD_FLAG "-hstd=c++${FULL_LC_STANDARD}", PARENT_SCOPE)
  set(KOKKOS_CXX_INTERMDIATE_STANDARD_FLAG "-hstd=c++${INT_LC_STANDARD}" PARENT_SCOPE)
endfunction()
