cmake_minimum_required (VERSION 3.5.1)

foreach(p
    CMP0048 # OK to clear PROJECT_VERSION on project()
    CMP0054 # CMake 3.1
    CMP0056 # export EXE_LINKER_FLAGS to try_run
    CMP0057 # Support no if() IN_LIST operator
    CMP0063 # Honor visibility properties for all targets
    CMP0077 # Allow option() overrides in importing projects
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

project (benchmark VERSION 1.6.1 LANGUAGES CXX)

option(BENCHMARK_ENABLE_EXCEPTIONS "Enable the use of exceptions in the benchmark library." ON)
option(BENCHMARK_USE_LIBCXX "Build and test using libc++ as the standard library." OFF)

if(NOT MSVC)
  option(BENCHMARK_BUILD_32_BITS "Build a 32 bit version of the library." OFF)
else()
  set(BENCHMARK_BUILD_32_BITS OFF CACHE BOOL "Build a 32 bit version of the library - unsupported when using MSVC)" FORCE)
endif()

# Export only public symbols
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(MSVC)
    # As of CMake 3.18, CMAKE_SYSTEM_PROCESSOR is not set properly for MSVC and
    # cross-compilation (e.g. Host=x86_64, target=aarch64) requires using the
    # undocumented, but working variable.
    # See https://gitlab.kitware.com/cmake/cmake/-/issues/15170
    set(CMAKE_SYSTEM_PROCESSOR ${MSVC_CXX_ARCHITECTURE_ID})
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ARM")
      set(CMAKE_CROSSCOMPILING TRUE)
    endif()
endif()

# Make sure we can import out CMake functions
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Read the git tags to determine the project version
include(GetGitVersion)
get_git_version(GIT_VERSION)

# If no git version can be determined, use the version
# from the project() command
if ("${GIT_VERSION}" STREQUAL "0.0.0")
  set(VERSION "${benchmark_VERSION}")
else()
  set(VERSION "${GIT_VERSION}")
endif()
# Tell the user what versions we are using
message(STATUS "Version: ${VERSION}")

# The version of the libraries
set(GENERIC_LIB_VERSION ${VERSION})
string(SUBSTRING ${VERSION} 0 1 GENERIC_LIB_SOVERSION)

# Import our CMake modules
include(AddCXXCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(CXXFeatureCheck)
include(GenerateExportHeader)

check_library_exists(rt shm_open "" HAVE_LIB_RT)

if (BENCHMARK_BUILD_32_BITS)
  add_required_cxx_compiler_flag(-m32)
endif()

if (MSVC)
  # Turn compiler warnings up to 11
  string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)

  if (NOT BENCHMARK_ENABLE_EXCEPTIONS)
    add_cxx_compiler_flag(-EHs-)
    add_cxx_compiler_flag(-EHa-)
    add_definitions(-D_HAS_EXCEPTIONS=0)
  endif()
else()
  # Try and enable C++11. Don't use C++14 because it doesn't work in some
  # configurations.
  add_cxx_compiler_flag(-std=c++11)
  if (NOT HAVE_CXX_FLAG_STD_CXX11)
    add_cxx_compiler_flag(-std=c++0x)
  endif()

  # Turn compiler warnings up to 11
  add_cxx_compiler_flag(-Wall)
  add_cxx_compiler_flag(-Wextra)
  add_cxx_compiler_flag(-Wshadow)
  add_cxx_compiler_flag(-Wfloat-equal)
  add_cxx_compiler_flag(-Wsuggest-override)
  add_cxx_compiler_flag(-pedantic)
  add_cxx_compiler_flag(-pedantic-errors)
  add_cxx_compiler_flag(-Wshorten-64-to-32)
  add_cxx_compiler_flag(-fstrict-aliasing)
  # Disable warnings regarding deprecated parts of the library while building
  # and testing those parts of the library.
  add_cxx_compiler_flag(-Wno-deprecated-declarations)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # Intel silently ignores '-Wno-deprecated-declarations',
    # warning no. 1786 must be explicitly disabled.
    # See #631 for rationale.
    add_cxx_compiler_flag(-wd1786)
  endif()
  if (NOT BENCHMARK_ENABLE_EXCEPTIONS)
    add_cxx_compiler_flag(-fno-exceptions)
  endif()

  if (HAVE_CXX_FLAG_FSTRICT_ALIASING)
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel") #ICC17u2: Many false positives for Wstrict-aliasing
      add_cxx_compiler_flag(-Wstrict-aliasing)
    endif()
  endif()
  # ICC17u2: overloaded virtual function "benchmark::Fixture::SetUp" is only partially overridden
  # (because of deprecated overload)
  add_cxx_compiler_flag(-wd654)
  add_cxx_compiler_flag(-Wthread-safety)
  if (HAVE_CXX_FLAG_WTHREAD_SAFETY)
    cxx_feature_check(THREAD_SAFETY_ATTRIBUTES)
  endif()

  # On most UNIX like platforms g++ and clang++ define _GNU_SOURCE as a
  # predefined macro, which turns on all of the wonderful libc extensions.
  # However g++ doesn't do this in Cygwin so we have to define it ourselfs
  # since we depend on GNU/POSIX/BSD extensions.
  if (CYGWIN)
    add_definitions(-D_GNU_SOURCE=1)
  endif()

  if (QNXNTO)
    add_definitions(-D_QNX_SOURCE)
  endif()

  # Coverage build type
  set(BENCHMARK_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG}"
    CACHE STRING "Flags used by the C++ compiler during coverage builds."
    FORCE)
  set(BENCHMARK_EXE_LINKER_FLAGS_COVERAGE "${CMAKE_EXE_LINKER_FLAGS_DEBUG}"
    CACHE STRING "Flags used for linking binaries during coverage builds."
    FORCE)
  set(BENCHMARK_SHARED_LINKER_FLAGS_COVERAGE "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}"
    CACHE STRING "Flags used by the shared libraries linker during coverage builds."
    FORCE)
  mark_as_advanced(
    BENCHMARK_CXX_FLAGS_COVERAGE
    BENCHMARK_EXE_LINKER_FLAGS_COVERAGE
    BENCHMARK_SHARED_LINKER_FLAGS_COVERAGE)
  set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage.")
  add_cxx_compiler_flag(--coverage COVERAGE)
endif()

if (BENCHMARK_USE_LIBCXX)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_cxx_compiler_flag(-stdlib=libc++)
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
          "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    add_cxx_compiler_flag(-nostdinc++)
    message(WARNING "libc++ header path must be manually specified using CMAKE_CXX_FLAGS")
    # Adding -nodefaultlibs directly to CMAKE_<TYPE>_LINKER_FLAGS will break
    # configuration checks such as 'find_package(Threads)'
    list(APPEND BENCHMARK_CXX_LINKER_FLAGS -nodefaultlibs)
    # -lc++ cannot be added directly to CMAKE_<TYPE>_LINKER_FLAGS because
    # linker flags appear before all linker inputs and -lc++ must appear after.
    list(APPEND BENCHMARK_CXX_LIBRARIES c++)
  else()
    message(FATAL_ERROR "-DBENCHMARK_USE_LIBCXX:BOOL=ON is not supported for compiler")
  endif()
endif(BENCHMARK_USE_LIBCXX)

set(EXTRA_CXX_FLAGS "")
if (WIN32 AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  # Clang on Windows fails to compile the regex feature check under C++11
  set(EXTRA_CXX_FLAGS "-DCMAKE_CXX_STANDARD=14")
endif()

# C++ feature checks
# Determine the correct regular expression engine to use
cxx_feature_check(STD_REGEX ${EXTRA_CXX_FLAGS})
cxx_feature_check(GNU_POSIX_REGEX ${EXTRA_CXX_FLAGS})
cxx_feature_check(POSIX_REGEX ${EXTRA_CXX_FLAGS})
if(NOT HAVE_STD_REGEX AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
  message(FATAL_ERROR "Failed to determine the source files for the regular expression backend")
endif()
if (NOT BENCHMARK_ENABLE_EXCEPTIONS AND HAVE_STD_REGEX
        AND NOT HAVE_GNU_POSIX_REGEX AND NOT HAVE_POSIX_REGEX)
  message(WARNING "Using std::regex with exceptions disabled is not fully supported")
endif()

cxx_feature_check(STEADY_CLOCK)
# Ensure we have pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set up directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Build the targets
add_subdirectory(src)
