#.rst:
# FindHWLOC
# ----------
#
# Try to find HWLOC, based on KOKKOS_HWLOC_DIR
#
# The following variables are defined:
#
#   HWLOC_FOUND - System has HWLOC
#   HWLOC_INCLUDE_DIR - HWLOC include directory
#   HWLOC_LIBRARIES - Libraries needed to use HWLOC

find_path(HWLOC_INCLUDE_DIR hwloc.h PATHS "${KOKKOS_HWLOC_DIR}/include" ${hwloc_ROOT}/include ${HWLOC_ROOT}/include)
find_library(HWLOC_LIBRARIES hwloc PATHS "${KOKKOS_HWLOC_DIR}/lib" ${hwloc_ROOT}/lib ${HWLOC_ROOT}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HWLOC DEFAULT_MSG
                                  HWLOC_INCLUDE_DIR HWLOC_LIBRARIES)

add_library(Kokkos::hwloc UNKNOWN IMPORTED)

#See note in kokkos_tribits.cmake about why they are not included
#INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}"
set_target_properties(Kokkos::hwloc PROPERTIES
  INTERFACE_COMPILE_FEATURES ""
  INTERFACE_COMPILE_OPTIONS ""
  IMPORTED_LOCATION "${HWLOC_LIBRARIES}"
)

mark_as_advanced(HWLOC_INCLUDE_DIR HWLOC_LIBRARIES)

