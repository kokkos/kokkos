#.rst:
# FindLIBNUMA
# ----------
#
# Try to find LIBNUMA, based on KOKKOS_LIBNUMA_DIR
#
# The following variables are defined:
#
#   LIBNUMA_FOUND - System has LIBNUMA
#   LIBNUMA_INCLUDE_DIR - LIBNUMA include directory
#   LIBNUMA_LIBRARIES - Libraries needed to use LIBNUMA

find_path(LIBNUMA_INCLUDE_DIR numa.h PATHS "${KOKKOS_LIBNUMA_DIR}/include" ${libnuma_ROOT}/include ${LIBNUMA_ROOT}/include)
find_library(LIBNUMA_LIBRARIES numa PATHS "${KOKKOS_LIBNUMA_DIR}/lib" ${libnuma_ROOT}/lib ${LIBNUMA_ROOT}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBNUMA DEFAULT_MSG
                                  LIBNUMA_INCLUDE_DIR LIBNUMA_LIBRARIES)

add_library(Kokkos::libnuma UNKNOWN IMPORTED)

#See note in kokkos_tribits.cmake about why they are not included
#INTERFACE_INCLUDE_DIRECTORIES "${LIBNUMA_INCLUDE_DIR}"
set_target_properties(Kokkos::libnuma PROPERTIES
  INTERFACE_COMPILE_FEATURES ""
  INTERFACE_COMPILE_OPTIONS ""
  IMPORTED_LOCATION "${LIBNUMA_LIBRARIES}"
)

mark_as_advanced(LIBNUMA_INCLUDE_DIR LIBNUMA_LIBRARIES)

