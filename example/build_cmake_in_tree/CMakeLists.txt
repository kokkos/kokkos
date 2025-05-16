# Kokkos minimally requires 3.16 right now,
# but your project can set it higher
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)

# Project can mix languages - must have C++ support
# Kokkos flags are only applied to C++ files
project(Example CXX)

# We build kokkos as a subdirectory of our project
add_subdirectory(${Example_SOURCE_DIR}/../.. ${Example_BINARY_DIR}/kokkos)

add_executable(example cmake_example.cpp)

# This is the only line required to set up all compiler/linker flags
#target_link_libraries(example kokkoscore_module)
target_link_libraries(example Kokkos::kokkos)

# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

print_target_properties(kokkoscore_module)
print_target_properties(kokkoscore)

if(NOT TARGET kokkoscore)
  message(FATAL_ERROR "BLA")
endif()

# Adds a test for the executable
enable_testing()
add_test(NAME KokkosInTree_Verify COMMAND example 10)
