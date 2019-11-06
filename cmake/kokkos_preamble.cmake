# We need some tricks here to get CMake to pass feature tests
# for nvcc_wrapper
#this uses list dir for Trilinos 
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/kokkos_functions.cmake)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/kokkos_pick_cxx_std.cmake)


