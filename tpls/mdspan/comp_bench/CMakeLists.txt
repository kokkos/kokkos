
include(metabench)

# C++ standards that our benchmark will actually support
set(_cbench_supported_standards 17 20 23)

# Obtain the subset that are actually supported by the compiler
set(_cbench_standards "")
foreach(_std IN LISTS _cbench_supported_standards)
  if ("cxx_std_${_std}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    list(APPEND _cbench_standards "${_std}")
  endif()
endforeach()

message(STATUS "Using the following C++ standards for compiler benchmarking: ${_cbench_standards}")

function(add_cxx_comparison name template range)

    set(all_datasets)
    foreach(std IN LISTS _cbench_standards)
      metabench_add_dataset(
          ${name}_${std}
          ${template}
          ${range}
          MEDIAN_OF 3
      )
      target_link_libraries(${name}_${std} mdspan)
      # Set parenthesis operator regardless of which standard we are using
      # so we don't have to ifdef
      target_compile_definitions(${name}_${std} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
      set_property(TARGET ${name}_${std} PROPERTY CXX_STANDARD ${std})
      set(all_datasets ${all_datasets} ${name}_${std})
    endforeach()

    metabench_add_chart(${name} DATASETS ${all_datasets})
endfunction()

add_cxx_comparison(submdspan_chart "cbench_submdspan.cpp.erb" "[2, 4, 8, 16, 32]")
