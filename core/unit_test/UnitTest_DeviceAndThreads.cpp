#include <Kokkos_Core.hpp>
#include <filesystem>
#include <iostream>
#include <string>

int get_device_count() {
#if defined(KOKKOS_ENABLE_CUDA)
  int count;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
  return count;
#elif defined(KOKKOS_ENABLE_HIP)
  int count;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDevice(&count));
  return count;
#elif defined(KOKKOS_ENABLE_OPENACC)
  return acc_get_num_devices(acc_get_device_type());
#else
  return 0;
#endif
}

int get_device_id() {
#if defined(KOKKOS_ENABLE_CUDA)
  int device;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDevice(&device));
  return device;
#elif defined(KOKKOS_ENABLE_HIP)
  int device_id;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDevice(&device_id));
  return device_id;
#elif defined(KOKKOS_ENABLE_OPENACC)
  return acc_get_device_num(acc_get_device_type());
#else
  return -1;
#endif
}

int get_max_threads() {
#if defined(KOKKOS_ENABLE_OPENMP)
  return omp_get_max_threads();
#elif defined(KOKKOS_ENABLE_THREADS)
  return std::thread::hardware_concurrency();
#else
  return 1;
#endif
}

int get_num_threads() {
  return Kokkos::DefaultHostExecutionSpace().concurrency();
}

int print_flag(std::string const& flag) {
  std::vector<std::string> valid_flags;
#define KOKKOS_TEST_PRINT_FLAG(NAME)   \
  if (flag == #NAME) {                 \
    std::cout << get_##NAME() << '\n'; \
    return EXIT_SUCCESS;               \
  }                                    \
  valid_flags.push_back(#NAME)

  KOKKOS_TEST_PRINT_FLAG(num_threads);
  KOKKOS_TEST_PRINT_FLAG(max_threads);
  KOKKOS_TEST_PRINT_FLAG(device_id);
  KOKKOS_TEST_PRINT_FLAG(device_count);

#undef KOKKOS_TEST_PRINT_FLAG

  std::cerr << "Invalid flag name " << flag << ".  Valid names are ";
  for (int i = 0; i < (int)valid_flags.size() - 1; ++i) {
    std::cerr << valid_flags[i] << ", ";
  }
  std::cerr << "and " << valid_flags.back() << ".\n";
  return EXIT_FAILURE;
}

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  if (argc != 2) {
    auto const filename = std::filesystem::path(argv[0]).filename().string();
    std::cerr << "Usage: " << filename << " NAME_OF_FLAG\n";
    return EXIT_FAILURE;
  }
  int exit_code = print_flag(argv[1]);
  return exit_code;
}
