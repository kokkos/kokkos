

TRIBITS_SUBPACKAGE(Profiler)

TRIBITS_ADD_OPTION_AND_DEFINE(
  KokkosProfiler_ENABLE_Profiling_Aggregate_MPI
  KOKKOSP_ENABLE_PROFILING_AGGREGATE_MPI
  "Enable MPI aggregation of profiling data in Kokkos."
  ${Kokkos_ENABLE_MPI}
  )

IF( KokkosProfiler_ENABLE_RTlib )
  set (KOKKOSP_ENABLE_RTLIB ON)
ENDIF()

ADD_SUBDIRECTORY(src)

#TRIBITS_ADD_TEST_DIRECTORIES(unit_tests)

TRIBITS_SUBPACKAGE_POSTPROCESS()
