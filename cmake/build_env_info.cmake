
find_package(Git QUIET)

if(Git_FOUND)
    # Get the current working branch
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Get the latest commit hash
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Get the latest commit description
    execute_process(
        COMMAND ${GIT_EXECUTABLE} show -s --format=%s
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_DESCRIPTION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Get the latest commit date
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --format=%cI
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    # Check if repo is dirty / clean
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff-index --quiet HEAD --
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE IS_DIRTY
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IS_DIRTY EQUAL 0)
        set(GIT_CLEAN_STATUS "CLEAN")
    else()
        set(GIT_CLEAN_STATUS "DIRTY")
    endif()

    configure_file(cmake/Environment_Info.h.in Environment_Info.hpp @ONLY)
endif()


