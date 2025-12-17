cmake_minimum_required(VERSION 3.12...3.19)

project(example LANGUAGES CXX)

if(NOT TARGET mdspan::mdspan)
  find_package(mdspan REQUIRED)
endif()

function(mdspan_add_example EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
  target_link_libraries(${EXENAME} mdspan::mdspan)
endfunction()

add_subdirectory(godbolt_starter)
add_subdirectory(dot_product)
add_subdirectory(tiled_layout)
add_subdirectory(restrict_accessor)
add_subdirectory(aligned_accessor)
