kokkos_add_executable(stream SOURCES stream-kokkos.cpp)

if(KOKKOS_ENABLE_HWLOC)
  kokkos_add_executable(stream_cpu SOURCES stream-kokkos-cpu.cpp)

  set(Kokkos_Benchmark_StreamCPU_NoNUMA_ENVIRONMENT          "NUMA_AWARE=0")
  set(Kokkos_Benchmark_StreamCPU_Numa_Firsttouch_ENVIRONMENT "NUMA_AWARE=1 NUMA_POLICY=FIRSTTOUCH")
  set(Kokkos_Benchmark_StreamCPU_Numa_Interleave_ENVIRONMENT "NUMA_AWARE=1 NUMA_POLICY=INTERLEAVE")
  kokkos_add_test(NAME Benchmark_StreamCPU_NoNUMA_           EXE stream_cpu)
  kokkos_add_test(NAME Benchmark_StreamCPU_Numa_Firsttouch   EXE stream_cpu)
  kokkos_add_test(NAME Benchmark_StreamCPU_Numa_Interleave   EXE stream_cpu)
endif()
