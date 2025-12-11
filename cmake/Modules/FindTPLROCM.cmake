find_package(hip REQUIRED)

set(TPL_ROCM_LIBRARIES hip::device)

kokkos_create_imported_tpl(ROCM INTERFACE LINK_LIBRARIES ${TPL_ROCM_LIBRARIES})
