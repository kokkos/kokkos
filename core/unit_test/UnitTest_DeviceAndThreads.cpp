/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <thread>

int get_device_count() {
#if defined(KOKKOS_ENABLE_CUDA)
  int count;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
  return count;
#elif defined(KOKKOS_ENABLE_HIP)
  int count;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDevice(&count));
  return count;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
  return omp_get_num_devices();
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
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
  return omp_get_device_num();
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

int get_disable_warnings() { return !Kokkos::show_warnings(); }

int get_tune_internals() { return Kokkos::tune_internals(); }

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
  KOKKOS_TEST_PRINT_FLAG(disable_warnings);
  KOKKOS_TEST_PRINT_FLAG(tune_internals);

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
    std::cerr << "Usage: <executable> NAME_OF_FLAG\n";
    return EXIT_FAILURE;
  }
  int exit_code = print_flag(argv[1]);
  return exit_code;
}
