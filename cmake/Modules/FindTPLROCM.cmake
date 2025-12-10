find_package(hip 6.2 REQUIRED)

set(TPL_ROCM_LIBRARIES "")
if(KOKKOS_ENABLE_COMPILE_AS_CMAKE_LANGUAGE)
  set(TPL_ROCM_LIBRARIES hip::device)
else()
  set(TPL_ROCM_LIBRARIES hip::device)
endif()

kokkos_create_imported_tpl(ROCM INTERFACE LINK_LIBRARIES ${TPL_ROCM_LIBRARIES})
