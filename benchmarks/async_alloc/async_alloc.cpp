#include <iostream>
#include <sstream>
#include <utility>

// Kokkos Headers
#include "Kokkos_Core.hpp"

using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

// Store the individual timings
std::vector<std::pair<size_t, double>> inner_loop_times;

// The pair returned are the first_alloc_time and the loop time
//
std::pair<double, double> test(bool up) {
  int iters      = 50;
  size_t minimum = 8 / sizeof(float);  // 64K

  size_t gb = 1024 * 1024 * 1024 / sizeof(float);  // number of floats per GiB
  size_t maximum = gb;  // on 32 bit, we make 1GiB the max

  // On 64-bit we make 16 GiB the max
  if constexpr (sizeof(size_t) == 8) maximum *= 16;

  std::vector<size_t> sizes;
  if (up) {
    for (size_t num = minimum; num <= maximum; num *= 2) sizes.push_back(num);
  } else {
    for (size_t num = maximum; num >= minimum; num /= 2) sizes.push_back(num);
  }

  Kokkos::Timer first_alloc_timer;
  {  // Prime the pump - first long alloc -- Time it.
    Kokkos::View<float *, MemorySpace> dummy("unlabeled", 64);
  }
  double first_alloc_time = first_alloc_timer.seconds();

  // FIXME: Exponential stepping here
  Kokkos::Timer inner_loop_timer;
  Kokkos::Timer alloc_loop_timer;

  // Measured
  for (size_t num : sizes) {
    inner_loop_timer.reset();
    for (int i = 0; i < iters; i++) {
      Kokkos::View<float *, MemorySpace> a("unlabeled", num);
    }
    double inner_loop_time = inner_loop_timer.seconds();

    // Store in vector
    inner_loop_times.push_back(std::make_pair<>(
        num * sizeof(float), inner_loop_time / static_cast<double>(iters)));
  }
  double alloc_loop_time = alloc_loop_timer.seconds();

  return std::make_pair(first_alloc_time, alloc_loop_time);
}

int main(int argc, char *argv[]) {
  bool up = true;

  for (int j = 0; j < argc; j++) {
    if (std::string(argv[j]) == "-d") {
      up = false;
    }
  }

  // Check the env var for reporting
  char *env_string = getenv("KOKKOS_CUDA_MEMPOOL_SIZE");
  std::cout << "Async Malloc Benchmark: KOKKOS_CUDA_MEMPOOL_SIZE is ";

  if (env_string == nullptr)
    std::cout << "not set,";
  else
    std::cout << " " << env_string << ",";

  if (up)
    std::cout << " memory cycling upwards \n";
  else
    std::cout << " memory_cycling downwards \n";
  std::cout << std::flush;

  Kokkos::initialize(argc, argv);

  inner_loop_times.reserve(34);

  // Love structured bindings?
  const auto [first_alloc_time, alloc_loop_time] = test(up);

  std::cout << "First Alloc: " << 64 << " bytes, " << first_alloc_time
            << " sec\n";
  std::cout << "Test Alloc Loop Total: " << alloc_loop_time << " sec\n";
  std::cout << "Alloc Loop Timings:\n";
  std::cout << "===================\n";

  std::cout << "# size (B) \t time (sec) \n";
  std::cout << "# -----------------------\n";
  std::sort(inner_loop_times.begin(), inner_loop_times.end(),
            [=](const auto &a, const auto &b) { return a.first < b.first; });
  for (auto pair : inner_loop_times) {
    std::cout << pair.first << ", " << pair.second << "\n";
  }
  Kokkos::finalize();
}
