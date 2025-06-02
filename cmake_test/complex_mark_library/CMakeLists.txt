# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
project(KokkosCMakeTestingComplexMarkLibrary CXX Fortran)

######
# This test has the libs defer calls to set the kokkos flags downstream.
# This will work independently of other libs in the project that have Kokkos automatically setting the properties

######
# This test can be used like a library, just scroll to the section that mimics how Kokkos is used in your project.
# There are tree sections:
#   - Targets without Kokkos dependency
#   - Targets dependent on a library that has a Kokkos dependency
#   - Targets dependent on a library that depends on a library that has a Kokkos dependency
# Although the test stops at two layers, find_package(Kokkos) sets the source and target properties independent of
# depth of the dependency tree.

###### Targets that don't depend on Kokkos are not manipulated by us
add_subdirectory(lib_without_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_without ../test_source/test_without.cpp ../test_source/source_plain_cxx.cpp
                                     ../test_source/source_plain_fortran.f
)

target_link_libraries(${PROJECT_NAME}_executable_without ${PROJECT_NAME}_Lib_without_kokkos_dependency)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_without PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_without PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosWithout_Verify COMMAND ${PROJECT_NAME}_executable_without 10)

###### Libraries that have direct dependency on Kokkos
###### PUBLIC dependency
add_subdirectory(lib_with_public_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_public ../test_source/test_public.cpp ../test_source/source_plain_cxx.cpp
                                    ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_public ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_public PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_Verify COMMAND ${PROJECT_NAME}_executable_public 10)

###### INTERFACE dependency
add_subdirectory(lib_with_interface_kokkos_dependency)
add_executable(
  ${PROJECT_NAME}_executable_interface ../test_source/test_interface.cpp ../test_source/source_plain_cxx.cpp
                                       ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_interface ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_interface_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_interface PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_Verify COMMAND ${PROJECT_NAME}_executable_interface 10)

###### PRIVATE dependency
add_subdirectory(lib_with_private_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_private ../test_source/test_private.cpp ../test_source/source_plain_cxx.cpp
                                     ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_private ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_private PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_Verify COMMAND ${PROJECT_NAME}_executable_private 10)

##### Libraries that have indirect dependency on Kokkos
##### PUBLIC dependency on INTERFACE dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_public_interface ../test_source/test_public_interface.cpp
                                              ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_public_interface ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_interface_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_interface PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_public_interface PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_interface_Verify COMMAND ${PROJECT_NAME}_executable_public_interface 10)

###### PUBLIC dependency on PUBLIC dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_public_public ../test_source/test_public_public.cpp ../test_source/source_plain_cxx.cpp
                                           ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_public_public ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_public_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_public PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_public_public PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_public_Verify COMMAND ${PROJECT_NAME}_executable_public_public 10)

###### PUBLIC dependency on PRIVATE dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_public_private ../test_source/test_public_private.cpp ../test_source/source_plain_cxx.cpp
                                            ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_public_private ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_private_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_private PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_public_private PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_private_Verify COMMAND ${PROJECT_NAME}_executable_public_private 10)

###### PRIVATE dependency on INTERFACE dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_private_interface
  ../test_source/test_private_interface.cpp ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_private_interface ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_interface_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_interface PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_private_interface PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_interface_Verify COMMAND ${PROJECT_NAME}_executable_private_interface 10)

###### PRIVATE dependency on PUBLIC dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_private_public ../test_source/test_private_public.cpp ../test_source/source_plain_cxx.cpp
                                            ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_private_public ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_public PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_private_public PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_public_Verify COMMAND ${PROJECT_NAME}_executable_private_public 10)

###### PRIVATE dependency on PRIVATE dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_private_private ../test_source/test_private_private.cpp
                                             ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_private_private ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_private_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_private PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_private_private PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_private_Verify COMMAND ${PROJECT_NAME}_executable_private_private 10)

###### INTERFACE dependency on INTERFACE dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_interface_interface
  ../test_source/test_interface_interface.cpp ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_interface_interface ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_interface_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_interface PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_interface_interface PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_interface_Verify COMMAND ${PROJECT_NAME}_executable_interface_interface
                                                                       10
)

###### INTERFACE dependency on PUBLIC dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_interface_public ../test_source/test_interface_public.cpp
                                              ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_interface_public ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_public_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_public PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_interface_public PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_public_Verify COMMAND ${PROJECT_NAME}_executable_interface_public 10)

###### INTERFACE dependency on PRIVATE dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
  ${PROJECT_NAME}_executable_interface_private
  ../test_source/test_interface_private.cpp ../test_source/source_plain_cxx.cpp ../test_source/source_plain_fortran.f
)

target_link_libraries(
  ${PROJECT_NAME}_executable_interface_private ${PROJECT_NAME}_Lib_without_kokkos_dependency
  ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_private PROPERTIES LINKER_LANGUAGE Fortran)
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 19)
    target_link_options(${PROJECT_NAME}_executable_interface_private PRIVATE -fno-fortran-main)
  endif()
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_private_Verify COMMAND ${PROJECT_NAME}_executable_interface_private 10)
