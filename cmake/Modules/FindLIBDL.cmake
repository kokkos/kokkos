find_path(LIBDL_INCLUDE_DIRS
  NAMES dlfcn.h)
find_library(LIBDL_LIBRARIES dl)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBDL DEFAULT_MSG LIBDL_LIBRARIES LIBDL_INCLUDE_DIRS)
mark_as_advanced(LIBDL_INCLUDE_DIRS LIBDL_LIBRARIES)

add_library(Kokkos::libdl UNKNOWN IMPORTED)
set_target_properties(Kokkos::libdl PROPERTIES
  IMPORTED_LOCATION "${LIBDL_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIBDL_INCLUDE_DIRS}")

