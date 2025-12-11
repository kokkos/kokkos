find_package(hip 6.2 REQUIRED)

set(TPL_ROCM_LIBRARIES hip::device)

kokkos_create_imported_tpl(ROCM INTERFACE LINK_LIBRARIES ${TPL_ROCM_LIBRARIES})
