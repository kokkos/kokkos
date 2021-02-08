# include guard
INCLUDE_GUARD(DIRECTORY)

##########################################################################################
#
#        Creates a 'format' target that runs clang-format
#
##########################################################################################

SET(_FMT ON)
# Visual Studio GUI reports "errors" occasionally
IF(WIN32)
    SET(_FMT OFF)
ENDIF()

OPTION(Kokkos_FORMAT_TARGET "Enable a clang-format target" ${_FMT})
MARK_AS_ADVANCED(Kokkos_FORMAT_TARGET)

IF(NOT Kokkos_FORMAT_TARGET)
    RETURN()
ENDIF()

# prefer clang-format 6.0
FIND_PROGRAM(KOKKOS_CLANG_FORMATTER
    NAMES
        clang-format-8
        clang-format-8.0
        clang-format-mp-8.0 # macports
        $ENV{CLANG_FORMAT_EXECUTABLE}
        clang-format)

FIND_PACKAGE(Git QUIET)

IF(NOT GIT_EXECUTABLE OR NOT KOKKOS_CLANG_FORMATTER)
    RETURN()
ENDIF()

EXECUTE_PROCESS(
    COMMAND             ${GIT_EXECUTABLE} ls-files
    WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE     Kokkos_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

STRING(REPLACE "\n" ";" Kokkos_FILES "${Kokkos_FILES}")
STRING(REPLACE " " ";" Kokkos_FILES "${Kokkos_FILES}")
FOREACH(_SRC ${Kokkos_FILES})
    IF(EXISTS "${PROJECT_SOURCE_DIR}/${_SRC}")
        SET(_SRC "${PROJECT_SOURCE_DIR}/${_SRC}")
    ENDIF()
    LIST(APPEND Kokkos_GIT_FILES ${_SRC})
ENDFOREACH()

FILE(GLOB_RECURSE Kokkos_SOURCES
    ${PROJECT_SOURCE_DIR}/*.hpp
    ${PROJECT_SOURCE_DIR}/*.cpp
    ${PROJECT_SOURCE_DIR}/*.h
    ${PROJECT_SOURCE_DIR}/*.cc)

# name of the format target
SET(Kokkos_FORMAT_TARGET format)
IF(TARGET ${Kokkos_FORMAT_TARGET})
    SET(${Kokkos_FORMAT_TARGET} format-kokkos)
ENDIF()

FOREACH(_SRC ${Kokkos_SOURCES})
    IF(NOT "${_SRC}" IN_LIST Kokkos_GIT_FILES OR "${_SRC}" MATCHES ".*/tpls/.*")
        LIST(REMOVE_ITEM Kokkos_SOURCES ${_SRC})
    ENDIF()
ENDFOREACH()

# simplify file names
STRING(REPLACE "${PROJECT_SOURCE_DIR}/" "" Kokkos_SOURCES "${Kokkos_SOURCES}")

ADD_CUSTOM_TARGET(${Kokkos_FORMAT_TARGET}
    COMMAND             ${KOKKOS_CLANG_FORMATTER} -i ${Kokkos_SOURCES}
    WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR}
    COMMENT             "[${PROJECT_NAME}] Running '${KOKKOS_CLANG_FORMATTER}'..."
    SOURCES             ${Kokkos_SOURCES})

UNSET(Kokkos_FILES)
UNSET(Kokkos_GIT_FILES)
UNSET(Kokkos_SOURCES)
