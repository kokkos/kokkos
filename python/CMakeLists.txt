
IF(NOT Kokkos_ENABLE_PYBIND11)
    return()
ENDIF()


# don't build a static python module
SET(BUILD_SHARED_LIBS ON)

# pybind11 has not migrated to CMAKE_CXX_STANDARD and neither has kokkos
if(NOT Kokkos_CXX_STANDARD)
    set(Kokkos_CXX_STANDARD 11)
endif()
IF(MSVC)
    set(PYBIND11_CPP_STANDARD /std:c++${Kokkos_CXX_STANDARD} CACHE STRING "C++ Language standard used by Pybind11")
ELSE()
    set(PYBIND11_CPP_STANDARD -std=c++${Kokkos_CXX_STANDARD} CACHE STRING "C++ Language standard used by Pybind11")
ENDIF()

# basically just used to get Python3_SITEARCH for installation
FIND_PACKAGE(Python3 REQUIRED COMPONENTS Interpreter Development)
# python binding library
FIND_PACKAGE(pybind11 REQUIRED)

pybind11_add_module(libpykokkos MODULE THIN_LTO
    ${CMAKE_CURRENT_LIST_DIR}/libpykokkos.cpp)

# link to kokkos
TARGET_LINK_LIBRARIES(libpykokkos PRIVATE kokkoscore)

# set the output path to <BINARY_DIR>/kokkos so one
# can test the python import from the build directory
# Really, only LIBRARY_* is needed for Unix but Windows
# builds are weird so just setting all of them
SET_TARGET_PROPERTIES(libpykokkos PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos
    PDB_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/kokkos)

# put version, python interpreter, etc. in the __init__.py file for reference
CONFIGURE_FILE(
    ${CMAKE_CURRENT_LIST_DIR}/kokkos/__init__.py.in
    ${PROJECT_BINARY_DIR}/kokkos/__init__.py @ONLY)

SET(Kokkos_INSTALL_PYTHONDIR ${Python3_SITEARCH}/kokkos)

# figure out if we can install to Python3_SITEARCH
EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E touch ${Python3_SITEARCH}/kokkos/__init__.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    ERROR_VARIABLE ERR_MSG
    RESULT_VARIABLE ERR_CODE)

IF(ERR_CODE)
    # get the python directory name, e.g. 'python3.6' from
    # '/opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6'
    get_filename_component(PYDIR "${Python3_STDLIB}" NAME)
    # Should not be CMAKE_INSTALL_LIBDIR! Python won't look in a lib64 folder
    set(Kokkos_INSTALL_PYTHONDIR lib/${PYDIR}/site-packages/kokkos)
ENDIF()

INSTALL(TARGETS libpykokkos
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})

INSTALL(FILES ${PROJECT_BINARY_DIR}/kokkos/__init__.py
    DESTINATION ${Kokkos_INSTALL_PYTHONDIR})
