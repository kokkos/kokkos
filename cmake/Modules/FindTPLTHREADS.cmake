INCLUDE(FindPackageHandleStandardArgs)
INCLUDE("${CMAKE_SOURCE_DIR}/cmake/tpls/FindTPLPthread.cmake")

IF (TARGET Threads::Threads)
  SET(FOUND_THREADS TRUE)
ELSE()
  SET(FOUND_THREADS FALSE)
ENDIF()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLTHREADS DEFAULT_MSG FOUND_THREADS)
#Only create the TPL if we succeed
IF (FOUND_THREADS)
  IF(USE_THREADS)
    KOKKOS_CREATE_IMPORTED_TPL(THREADS INTERFACE LINK_OPTIONS
            ${TPL_Pthread_LIBRARIES})
  ELSE()
    KOKKOS_CREATE_IMPORTED_TPL(THREADS
            INTERFACE   #this is not a real library with a real location
            COMPILE_OPTIONS -pthread
            LINK_OPTIONS    -pthread)
  ENDIF()
ENDIF()
