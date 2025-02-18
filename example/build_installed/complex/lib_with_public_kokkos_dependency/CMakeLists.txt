find_package(Kokkos)

add_library(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency
  ../../example_source/lib_with_public_kokkos_dependency/source_with_public_kokkos_dependency.cpp
)

if((CUDA IN_LIST Kokkos_DEVICES) OR (HIP IN_LIST Kokkos_DEVICES))
    target_sources(
            ${PROJECT_NAME}_Lib_with_public_kokkos_dependency PRIVATE
            ../../example_source/lib_with_public_kokkos_dependency/cuda_hip_source_without_kokkos_dependency.cu
    )
    #for builds without CUDA/HIP language support we need to mark the cuda file as CXX. For language support it marks the file in the appropriate language
    set_property(SOURCE ../../example_source/lib_with_public_kokkos_dependency/cuda_hip_source_without_kokkos_dependency.cu TARGET_DIRECTORY ${PROJECT_NAME}_Lib_with_public_kokkos_dependency PROPERTY LANGUAGE ${Kokkos_COMPILE_LANGUAGE})
endif()

target_include_directories(
  ${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC ../../example_source/lib_with_public_kokkos_dependency
)
target_link_libraries(${PROJECT_NAME}_Lib_with_public_kokkos_dependency PUBLIC Kokkos::kokkos)