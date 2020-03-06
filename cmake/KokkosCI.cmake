cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

#----------------------------------------------------------------------------------------#
#
#   Macros and variables
#
#----------------------------------------------------------------------------------------#

macro(CHECK_REQUIRED VAR)
    if(NOT DEFINED ${VAR})
        message(FATAL_ERROR "Error! Variable '${VAR}' must be defined")
    endif()
endmacro()

macro(SET_DEFAULT VAR)
    if(NOT DEFINED ${VAR})
        set(${VAR} ${ARGN})
    endif()
endmacro()

# determine the default working directory
if(NOT "$ENV{WORKSPACE}" STREQUAL "")
    set(WORKING_DIR "$ENV{WORKSPACE}")
else()
    get_filename_component(WORKING_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
endif()

# determine the hostname
execute_process(COMMAND hostname
    OUTPUT_VARIABLE HOSTNAME
    OUTPUT_STRIP_TRAILING_WHITESPACE)

SET_DEFAULT(HOSTNAME "$ENV{HOSTNAME}")

# get the number of processors
include(ProcessorCount)
ProcessorCount(NUM_PROCESSORS)

# find git
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    unset(GIT_EXECUTABLE CACHE)
    unset(GIT_EXECUTABLE)
endif()

# function for finding variables matching a regex expression
function(GET_CMAKE_VARIABLES VAR REGEXPR)
    # get local and cache variables
    get_cmake_property(_VAR_NAMES VARIABLES)
    get_cmake_property(_TMP_NAMES CACHE_VARIABLES)
    list(APPEND _VAR_NAMES ${_TMP_NAMES})
    list(SORT _VAR_NAMES)
    set(_TMP)
    # loop over variables
    foreach(_VAR ${_VAR_NAMES})
        # loop over regex arguments
        foreach(_ARG ${REGEXPR} ${ARGN})
            # apply regex
            string(REGEX MATCH ${_ARG} MATCHED ${_VAR})
            # append if matched
            if(MATCHED)
                list(APPEND _TMP ${MATCHED})
            endif()
            unset(MATCHED)
        endforeach()
    endforeach()
    if(_TMP)
        list(REMOVE_DUPLICATES _TMP)
        list(SORT _TMP)
    endif()
    # message(STATUS "MATCHES: ${_TMP}")
    set(${VAR} ${_TMP} PARENT_SCOPE)
endfunction()


#----------------------------------------------------------------------------------------#
#
#   Set the configurations
#
#----------------------------------------------------------------------------------------#

set(VALID_CONFIGS
    "HIP-3.1-HCC"
    "CUDA-9.2-Clang"
    "CUDA-9.2-NVCC"
    "CUDA-10.1-NVCC-RDC"
    "CUDA-10.1-NVCC-DEBUG"
    "GCC-4.8.4"
    "CUSTOM"
)

set(COMMON_CONFIG_ARGS
    "-DCMAKE_CXX_FLAGS=-Werror"
    "-DKokkos_ENABLE_COMPILER_WARNINGS=ON"
    )

set(COMMON_DEBUG_ARGS
    "-DKokkos_ENABLE_DEBUG=ON"
    "-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON"
    )

set(COMMON_CUDA_ARGS
    "-DKokkos_ENABLE_CUDA=ON"
    "-DKokkos_ENABLE_CUDA_LAMBDA=ON"
    "-DKokkos_ARCH_VOLTA70=ON"
    )

#----------------------------------------------------------------------------------------#
#
#   Check
#
#----------------------------------------------------------------------------------------#

SET_DEFAULT(CONFIG "CUSTOM")

if(NOT "${CONFIG}" IN_LIST VALID_CONFIGS)
    message(DEVELOPER_WARNING "Error! Configuration '${CONFIG}' not in: '${VALID_CONFIGS}'")
    set(CONFIG CUSTOM)
endif()

string(REGEX REPLACE "(\\.|-)" "_" CONFIG_VAR "${CONFIG}")

#----------------------------------------------------------------------------------------#
#
#   Build types
#
#----------------------------------------------------------------------------------------#

set(HIP_3_1_HCC_BUILD_TYPE          "Debug")
set(CUDA_9_2_CLANG_BUILD_TYPE       "Release")
set(CUDA_9_2_NVCC_BUILD_TYPE        "Release")
set(CUDA_10_1_NVCC_RDC_BUILD_TYPE   "Release")
set(CUDA_10_1_NVCC_DEBUG_BUILD_TYPE "Debug")
set(GCC_4_8_4_BUILD_TYPE            "Release")
set(CUSTOM_BUILD_TYPE               "${CMAKE_BUILD_TYPE}")

#----------------------------------------------------------------------------------------#
#
#   Configure arguments
#
#----------------------------------------------------------------------------------------#

set(HIP_3_1_HCC_CONFIG_ARGS
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER=hipcc"
    "-DKokkos_ENABLE_HIP=ON"
    "-DKokkos_ENABLE_LIBDL=OFF"
    "-DKokkos_ENABLE_PROFILING=OFF"
    )

set(CUDA_9_2_CLANG_CONFIG_ARGS
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER=clang++"
    ${COMMON_CUDA_ARGS}
    ${COMMON_CONFIG_ARGS}
    )

set(CUDA_9_2_NVCC_CONFIG_ARGS
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper"
    ${COMMON_CUDA_ARGS}
    ${COMMON_CONFIG_ARGS}
    )

set(CUDA_10_1_NVCC_RDC_CONFIG_ARGS
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper"
    "-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON"
    "-DKokkos_ENABLE_CUDA_UVM=ON"
    "-DKokkos_ENABLE_OPENMP=ON"
    ${COMMON_CUDA_ARGS}
    ${COMMON_CONFIG_ARGS}
    )

set(CUDA_10_1_NVCC_DEBUG_CONFIG_ARGS
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER=$WORKSPACE/bin/nvcc_wrapper"
    ${COMMON_CUDA_ARGS}
    ${COMMON_CONFIG_ARGS}
    ${COMMON_DEBUG_ARGS}
    )

set(GCC_4_8_4_CONFIG_ARGS
    "-DKokkos_ENABLE_OPENMP=ON"
    ${COMMON_CONFIG_ARGS}
    )

#----------------------------------------------------------------------------------------#
#
#   Set default values if not provided on command-line
#
#----------------------------------------------------------------------------------------#

SET_DEFAULT(SOURCE_DIR      "${WORKING_DIR}")       # source directory
SET_DEFAULT(BINARY_DIR      "${WORKING_DIR}/build")  # build directory
SET_DEFAULT(BUILD_TYPE      "${${CONFIG_VAR}_BUILD_TYPE}")  # Release, Debug, etc.
SET_DEFAULT(BUILD_NAME      "${CONFIG}")            # Build id
SET_DEFAULT(MODEL           "Continuous")           # Continuous, Nightly, or Experimental
SET_DEFAULT(SITE            "${HOSTNAME}")          # update site
SET_DEFAULT(JOBS            1)                      # number of parallel ctests
SET_DEFAULT(CTEST_COMMAND   "ctest")                # just in case
SET_DEFAULT(CTEST_ARGS      "")                     # extra arguments when ctest is called
SET_DEFAULT(GIT_EXECUTABLE  "git")                  # ctest_update
SET_DEFAULT(NUM_PROC        "${NUM_PROCESSORS}")    # number of parallel compile jobs
SET_DEFAULT(TARGET          "all")                  # build target
#
#   The variable below correspond to ctest arguments, i.e. START,END,STRIDE are
#   '-I START,END,STRIDE'
#
SET_DEFAULT(START           "")
SET_DEFAULT(END             "")
SET_DEFAULT(STRIDE          "")
SET_DEFAULT(INCLUDE         "")
SET_DEFAULT(EXCLUDE         "")
SET_DEFAULT(INCLUDE_LABEL   "")
SET_DEFAULT(EXCLUDE_LABEL   "")
SET_DEFAULT(PARALLEL_LEVEL  "")
SET_DEFAULT(STOP_TIME       "")
SET_DEFAULT(LABELS          "")
SET_DEFAULT(NOTES           "")

if("${CONFIG}" STREQUAL "CUSTOM" AND "${BUILD_NAME}" STREQUAL "${CONFIG}")
    set(BUILD_NAME "${HOSTNAME}")
endif()

# check binary directory
if(EXISTS ${BINARY_DIR})
    if(NOT IS_DIRECTORY "${BINARY_DIR}")
        message(FATAL_ERROR "Error! '${BINARY_DIR}' already exists and is not a directory!")
    endif()
    file(GLOB BINARY_DIR_FILES "${BINARY_DIR}/*")
    if(NOT "${BINARY_DIR_FILES}" STREQUAL "")
        message(FATAL_ERROR "Error! '${BINARY_DIR}' already exists and is not empty!")
    endif()
endif()

get_filename_component(SOURCE_REALDIR ${SOURCE_DIR} REALPATH)
get_filename_component(BINARY_REALDIR ${BINARY_DIR} REALPATH)

#----------------------------------------------------------------------------------------#
#
#   Generate the CTestConfig.cmake
#
#----------------------------------------------------------------------------------------#

get_cmake_variables(KOKKOS_CMAKE_ARGS "Kokkos_.*" "CMAKE_BUILD_.*" "CMAKE_INSTALL_.*"
    "CMAKE_.*_FLAGS")

foreach(_ARG ${KOKKOS_CMAKE_ARGS})
    if(NOT "${${_ARG}}" STREQUAL "")
        list(APPEND ${CONFIG_VAR}_CONFIG_ARGS "-D${_ARG}=${${_ARG}}")
    endif()
endforeach()

string(REPLACE ";" " " CONFIG_ARGS  "${${CONFIG_VAR}_CONFIG_ARGS}")

# message(STATUS "BUILD_TYPE: ${BUILD_TYPE}; CONFIG: ${CONFIG}; CONFIG_VAR: ${CONFIG_VAR}")
# message(STATUS "CONFIG_ARGS: ${CONFIG_ARGS}; BUILD_TYPE: ${BUILD_TYPE}")

# generate the CTestConfig.cmake
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/CTestConfig.cmake.in
    ${BINARY_REALDIR}/CTestConfig.cmake
    @ONLY)

# copy/generate the dashboard script
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/KokkosCTest.cmake.in
    ${BINARY_REALDIR}/KokkosCTest.cmake
    @ONLY)

# custom CTest settings go in ${BINARY_DIR}/CTestCustom.cmake
execute_process(
    COMMAND             ${CMAKE_COMMAND} -E touch CTestCustom.cmake
    WORKING_DIRECTORY   ${BINARY_REALDIR}
    )

#----------------------------------------------------------------------------------------#
#
#   Execute CTest
#
#----------------------------------------------------------------------------------------#

execute_process(
    COMMAND             ${CTEST_COMMAND} -S KokkosCTest.cmake ${CTEST_ARGS}
    RESULT_VARIABLE     RET
    WORKING_DIRECTORY   ${BINARY_REALDIR}
    )

if(RET GREATER 0)
    message(FATAL_ERROR "CTest return non-zero exit code: ${RET}")
endif()
