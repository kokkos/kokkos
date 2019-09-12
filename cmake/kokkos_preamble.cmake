# We need some tricks here to get CMake to pass feature tests
# for nvcc_wrapper
#this uses list dir for Trilinos 
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/kokkos_functions.cmake)
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/kokkos_pick_cxx_std.cmake)

# NVCC does not accept all the -std flags
# If NVCC receives a "bad flag", this variable tells nvcc_wrapper which 
# -std flag to use. This is most often necessary during
# feature tests in CMake
SET(ENV{KOKKOS_CMAKE_CXX_STANDARD} ${KOKKOS_CXX_STANDARD})

