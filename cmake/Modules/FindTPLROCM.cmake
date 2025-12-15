find_package(hip REQUIRED PATHS ${ROCM_PATH} $ENV{ROCM_PATH})

kokkos_create_imported_tpl(ROCM INTERFACE LINK_LIBRARIES hip::device)
