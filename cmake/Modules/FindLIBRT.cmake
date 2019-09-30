find_path(LIBRT_INCLUDE_DIRS
  NAMES time.h
  PATHS ${LIBRT_ROOT}/include/ ${KOKKOS_LIBRT_DIR}/include
)

find_library(LIBRT_LIBRARIES rt
 PATHS ${KOKKOS_HWLOC_DIR}/lib ${LIBRT_ROOT}/lib
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBRT DEFAULT_MSG LIBRT_LIBRARIES LIBRT_INCLUDE_DIRS)
mark_as_advanced(LIBRT_INCLUDE_DIRS LIBRT_LIBRARIES)

add_library(Kokkos::librt UNKNOWN IMPORTED)
set_target_properties(Kokkos::librt PROPERTIES
  IMPORTED_LOCATION "${LIBRT_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBRT_INCLUDE_DIRS}")

