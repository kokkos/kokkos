include(FindPackageHandleStandardArgs)

find_package(hip CONFIG)

find_package_handle_standard_args(TPLROCM
  REQUIRED_VARS HIP_INCLUDE_DIR
  VERSION_VAR hip_VERSION
)

if(TARGET hip::amdhip64)
  kokkos_export_imported_tpl(hip::amdhip64)
  kokkos_create_imported_tpl(ROCM INTERFACE LINK_LIBRARIES hip::amdhip64)
endif()
