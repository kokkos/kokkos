# Kokkos minimally requires 3.21 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.21)

# Projects can safely mix languages - must have C++ support
project(ExampleComplexMarkKokkosFiles CXX Fortran)

######
# This example searches for Kokkos at the top level and deactivates automatically marking source files and libs
# Then individual sources and targets are opt-in to have Kokkos dependency
find_package(Kokkos)
# Opt out of kokkos setting the source and target flags automatically by excluding the whole project.
kokkos_exclude_from_setting_build_properties(GLOBAL)

######
# This example can be used like a library, just scroll to the section that mimics how Kokkos is used in your project.
# There are tree sections:
#   - Targets without Kokkos dependency
#   - Targets dependent on a library that has a Kokkos dependency
#   - Targets dependent on a library that depends on a library that has a Kokkos dependency
# Although the example stops at two layers, find_package(Kokkos) sets the source and target properties independent of
# depth of the dependency tree.

###### Targets that don't depend on Kokkos are not manipulated by us
add_subdirectory(lib_without_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_without
        ../example_source/example_without.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_without
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_without PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosWithout_Verify COMMAND ${PROJECT_NAME}_executable_without 10)

###### Libraries that have direct dependency on Kokkos
###### PUBLIC dependency
add_subdirectory(lib_with_public_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_public
        ../example_source/example_public.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_public
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
)
# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_public.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_public
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_Verify COMMAND ${PROJECT_NAME}_executable_public 10)

###### INTERFACE dependency
add_subdirectory(lib_with_interface_kokkos_dependency)
add_executable(
        ${PROJECT_NAME}_executable_interface
        ../example_source/example_interface.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_interface
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_interface_kokkos_dependency
)
# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_interface.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_interface
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_Verify COMMAND ${PROJECT_NAME}_executable_interface 10)

###### PRIVATE dependency
add_subdirectory(lib_with_private_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_private
        ../example_source/example_private.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_private
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_private_kokkos_dependency
)
# since Kokkos is a private dependency for the used libs we do not need to mark source files
# but as the target need to know which runtime to link, it needs to be marked
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_private
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_Verify COMMAND ${PROJECT_NAME}_executable_private 10)


##### Libraries that have indirect dependency on Kokkos
##### PUBLIC dependency on INTERFACE dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_public_interface
        ../example_source/example_public_interface.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_public_interface
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_interface_kokkos_dependency
)
# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_public_interface.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_public_interface
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_interface PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_interface_Verify COMMAND ${PROJECT_NAME}_executable_public_interface 10)

###### PUBLIC dependency on PUBLIC dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_public_public
        ../example_source/example_public_public.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_public_public
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_public_kokkos_dependency
)
# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_public_public.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_public_public
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_public PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_public_Verify COMMAND ${PROJECT_NAME}_executable_public_public 10)

###### PUBLIC dependency on PRIVATE dependency
add_subdirectory(lib_with_public_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_public_private
        ../example_source/example_public_private.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_public_private
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_public_dependency_on_lib_with_private_kokkos_dependency
)
# the target needs kokkos linker settings
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_public_private
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_public_private PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPublic_private_Verify COMMAND ${PROJECT_NAME}_executable_public_private 10)

###### PRIVATE dependency on INTERFACE dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_private_interface
        ../example_source/example_private_interface.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_private_interface
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_interface_kokkos_dependency
)
# the target needs kokkos linker settings
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_private_interface
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_interface PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_interface_Verify COMMAND ${PROJECT_NAME}_executable_private_interface 10)

###### PRIVATE dependency on PUBLIC dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_private_public
        ../example_source/example_private_public.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_private_public
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_public_kokkos_dependency
)
# the target needs kokkos linker settings
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_private_public
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_public PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_public_Verify COMMAND ${PROJECT_NAME}_executable_private_public 10)

###### PRIVATE dependency on PRIVATE dependency
add_subdirectory(lib_with_private_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_private_private
        ../example_source/example_private_private.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_private_private
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_private_dependency_on_lib_with_private_kokkos_dependency
)
# the target needs kokkos linker settings
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_private_private
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_private_private PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosPrivate_private_Verify COMMAND ${PROJECT_NAME}_executable_private_private 10)

###### INTERFACE dependency on INTERFACE dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_interface_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_interface_interface
        ../example_source/example_interface_interface.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_interface_interface
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_interface_kokkos_dependency
)

# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_interface_interface.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_interface_interface
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_interface PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_interface_Verify COMMAND ${PROJECT_NAME}_executable_interface_interface 10)

###### INTERFACE dependency on PUBLIC dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_public_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_interface_public
        ../example_source/example_interface_public.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_interface_public
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_public_kokkos_dependency
)
# as the executable uses Kokkos in its implementation, the source files have to be marked accordingly
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/example_interface_public.cpp
)
# FIXME: for language builds Kokkos still sets CXX flags which infect non Kokkos files, thus we will need to mark them here
kokkos_set_source_and_target_properties(
        SOURCE
        ${CMAKE_CURRENT_SOURCE_DIR}/../example_source/source_plain_cxx.cpp
)
# same for the target
kokkos_set_source_and_target_properties(
        TARGET ${PROJECT_NAME}_executable_interface_public
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_public PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_public_Verify COMMAND ${PROJECT_NAME}_executable_interface_public 10)

###### INTERFACE dependency on PRIVATE dependency
add_subdirectory(lib_with_interface_dependency_on_lib_with_private_kokkos_dependency)

add_executable(
        ${PROJECT_NAME}_executable_interface_private
        ../example_source/example_interface_private.cpp ../example_source/source_plain_cxx.cpp
        ../example_source/source_plain_fortran.f
)

target_link_libraries(
        ${PROJECT_NAME}_executable_interface_private
        ${PROJECT_NAME}_Lib_without_kokkos_dependency
        ${PROJECT_NAME}_Lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
)

# the target needs to link with kokkos settings
kokkos_set_source_and_target_properties(
        LINK_ONLY_TARGET ${PROJECT_NAME}_executable_interface_private
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL LLVMFlang)
  set_target_properties(${PROJECT_NAME}_executable_interface_private PROPERTIES LINKER_LANGUAGE Fortran)
endif()

add_test(NAME ${PROJECT_NAME}_KokkosInterface_private_Verify COMMAND ${PROJECT_NAME}_executable_interface_private 10)