
function(mdspan_add_benchmark EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
  target_link_libraries(${EXENAME} mdspan benchmark::benchmark)
  target_include_directories(${EXENAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
  )
  # Set flag to build with parenthesis enabled
  target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
endfunction()

if(MDSPAN_USE_SYSTEM_BENCHMARK)
  find_package(benchmark REQUIRED)
else()
  set(BENCHMARK_ENABLE_TESTING OFF)
  include(FetchContent)
  
  fetchcontent_declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.1
  )

  # TODO CMake 3.28, we can pass EXCLUDE_FROM_ALL directly to fetchcontent_makeavailable(benchmark)
  fetchcontent_getproperties(benchmarkgoogletest)
  if (NOT benchmark_POPULATED)
    fetchcontent_populate(benchmark)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

function(mdspan_add_cuda_benchmark EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cu)
  target_link_libraries(${EXENAME} PUBLIC mdspan)
  target_include_directories(${EXENAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
  )
  # This is gross, but it's the best I can do in this version of CMake
  get_target_property(_benchmark_include benchmark::benchmark INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(_benchmark_libs_old benchmark::benchmark INTERFACE_LINK_LIBRARIES)
  get_target_property(_benchmark_libs_imported benchmark::benchmark IMPORTED_LOCATION_RELEASE)
  if(NOT _benchmark_libs_imported)
    get_target_property(_benchmark_libs_imported benchmark::benchmark IMPORTED_LOCATION_DEBUG)
    if(NOT _benchmark_libs_imported)
      get_target_property(_benchmark_libs_imported benchmark::benchmark IMPORTED_LOCATION_RELWITHDEBINFO)
      if(NOT _benchmark_libs_imported)
        get_target_property(_benchmark_libs_imported benchmark::benchmark IMPORTED_LOCATION_MINSIZEREL)
        if(NOT _benchmark_libs_imported)
          message(FATAL_ERROR
              "Could not figure out how to import google benchmark to hack around a cmake Cuda compatibility issue.  Later versions of CMake will make this unnecessary"
          )
        else()
          message(WARNING "Importing a Google Benchmark installation that was compiled in MinSizeRel mode.  Times might not be reliable")
        endif()
      else()
        message(WARNING "Importing a Google Benchmark installation that was compiled in RelWithDebInfo mode.  Times might not be reliable")
      endif()
    else()
      message(WARNING "Importing a Google Benchmark installation that was compiled in Debug mode.  Times might not be reliable")
    endif()
  endif()
  string(REPLACE "-pthread" "" _benchmark_libs "${_benchmark_libs_old}")
  target_include_directories(${EXENAME} PUBLIC "${_benchmark_include}")
  target_link_libraries(${EXENAME} PUBLIC "${_benchmark_libs};${_benchmark_libs_imported}")
  target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
  if(_benchmark_libs_old MATCHES "-pthread")
    target_compile_options(${EXENAME} PUBLIC "-Xcompiler=-pthread")
  endif()
endfunction()

if(MDSPAN_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

function(mdspan_add_openmp_benchmark EXENAME)
  if(MDSPAN_ENABLE_OPENMP)
    if(OpenMP_CXX_FOUND)
      add_executable(${EXENAME} ${EXENAME}.cpp)
      target_link_libraries(${EXENAME} mdspan benchmark::benchmark OpenMP::OpenMP_CXX)
      target_include_directories(${EXENAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
      )
      target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
    else()
      message(WARNING "Not adding target ${EXENAME} because OpenMP was not found")
    endif()
  endif()
endfunction()

add_subdirectory(sum)
add_subdirectory(matvec)
add_subdirectory(copy)
add_subdirectory(stencil)
add_subdirectory(tiny_matrix_add)
