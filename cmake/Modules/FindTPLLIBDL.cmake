# Finding dlfcn.h includes a problematic path on MacOS X using homebrew's clang
# Mac OS X doesn't require extra libraries for using dlopen
if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
  kokkos_find_imported(LIBDL HEADER dlfcn.h INTERFACE LIBRARIES ${CMAKE_DL_LIBS})
endif()
