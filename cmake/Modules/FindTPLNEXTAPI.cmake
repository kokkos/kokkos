find_package(NextAPI REQUIRED COMPONENTS Core)
set_target_properties(NextAPI::Core PROPERTIES SYSTEM TRUE)

kokkos_create_imported_tpl(NEXTAPI INTERFACE LINK_LIBRARIES NextAPI::Core)
kokkos_export_cmake_tpl(NEXTAPI REQUIRED)

# $ORIGIN-relative RUNPATHs matching the NEXT_HOME layout so packages
# stay relocatable.  Mirrors the convention in nextutils/CMakeLists.txt.
if(TPLNEXTAPI_FOUND)
  set(_NS_INSTALL_RPATH "lib" "sysroot/usr/lib" "sysroot/usr/ompi/lib64")
  list(TRANSFORM _NS_INSTALL_RPATH PREPEND "$ORIGIN/../")

  list(APPEND CMAKE_INSTALL_RPATH ${_NS_INSTALL_RPATH})
  list(APPEND CMAKE_BUILD_RPATH ${_NS_INSTALL_RPATH})
endif()
