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
#include <impl/Kokkos_Error.hpp>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <stack>
#include <functional>
#include <list>
#include <cerrno>
#ifndef _WIN32
#include <unistd.h>
#endif

//----------------------------------------------------------------------------
namespace {
bool g_is_initialized = false;
bool g_show_warnings  = true;
// When compiling with clang/LLVM and using the GNU (GCC) C++ Standard Library
// (any recent version between GCC 7.3 and GCC 9.2), std::deque SEGV's during
// the unwinding of the atexit(3C) handlers at program termination.  However,
// this bug is not observable when building with GCC.
// As an added bonus, std::list<T> provides constant insertion and
// deletion time complexity, which translates to better run-time performance. As
// opposed to std::deque<T> which does not provide the same constant time
// complexity for inserts/removals, since std::deque<T> is implemented as a
// segmented array.
using hook_function_type = std::function<void()>;
std::stack<hook_function_type, std::list<hook_function_type>> finalize_hooks;
}  // namespace

namespace Kokkos {
namespace Impl {

int get_ctest_gpu(const char* local_rank_str) {
  auto const* ctest_kokkos_device_type =
      std::getenv("CTEST_KOKKOS_DEVICE_TYPE");
  if (!ctest_kokkos_device_type) {
    return 0;
  }

  auto const* ctest_resource_group_count_str =
      std::getenv("CTEST_RESOURCE_GROUP_COUNT");
  if (!ctest_resource_group_count_str) {
    return 0;
  }

  // Make sure rank is within bounds of resource groups specified by CTest
  auto resource_group_count = std::stoi(ctest_resource_group_count_str);
  auto local_rank           = std::stoi(local_rank_str);
  if (local_rank >= resource_group_count) {
    std::ostringstream ss;
    ss << "Error: local rank " << local_rank
       << " is outside the bounds of resource groups provided by CTest. Raised"
       << " by Kokkos::Impl::get_ctest_gpu().";
    throw_runtime_exception(ss.str());
  }

  // Get the resource types allocated to this resource group
  std::ostringstream ctest_resource_group;
  ctest_resource_group << "CTEST_RESOURCE_GROUP_" << local_rank;
  std::string ctest_resource_group_name = ctest_resource_group.str();
  auto const* ctest_resource_group_str =
      std::getenv(ctest_resource_group_name.c_str());
  if (!ctest_resource_group_str) {
    std::ostringstream ss;
    ss << "Error: " << ctest_resource_group_name << " is not specified. Raised"
       << " by Kokkos::Impl::get_ctest_gpu().";
    throw_runtime_exception(ss.str());
  }

  // Look for the device type specified in CTEST_KOKKOS_DEVICE_TYPE
  bool found_device                        = false;
  std::string ctest_resource_group_cxx_str = ctest_resource_group_str;
  std::istringstream instream(ctest_resource_group_cxx_str);
  while (true) {
    std::string devName;
    std::getline(instream, devName, ',');
    if (devName == ctest_kokkos_device_type) {
      found_device = true;
      break;
    }
    if (instream.eof() || devName.length() == 0) {
      break;
    }
  }

  if (!found_device) {
    std::ostringstream ss;
    ss << "Error: device type '" << ctest_kokkos_device_type
       << "' not included in " << ctest_resource_group_name
       << ". Raised by Kokkos::Impl::get_ctest_gpu().";
    throw_runtime_exception(ss.str());
  }

  // Get the device ID
  std::string ctest_device_type_upper = ctest_kokkos_device_type;
  for (auto& c : ctest_device_type_upper) {
    c = std::toupper(c);
  }
  ctest_resource_group << "_" << ctest_device_type_upper;

  std::string ctest_resource_group_id_name = ctest_resource_group.str();
  auto resource_str = std::getenv(ctest_resource_group_id_name.c_str());
  if (!resource_str) {
    std::ostringstream ss;
    ss << "Error: " << ctest_resource_group_id_name
       << " is not specified. Raised by Kokkos::Impl::get_ctest_gpu().";
    throw_runtime_exception(ss.str());
  }

  auto const* comma = std::strchr(resource_str, ',');
  if (!comma || strncmp(resource_str, "id:", 3)) {
    std::ostringstream ss;
    ss << "Error: invalid value of " << ctest_resource_group_id_name << ": '"
       << resource_str << "'. Raised by Kokkos::Impl::get_ctest_gpu().";
    throw_runtime_exception(ss.str());
  }

  std::string id(resource_str + 3, comma - resource_str - 3);
  return std::stoi(id.c_str());
}

namespace {

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
int get_gpu(const InitArguments& args) {
  int use_gpu           = args.device_id;
  const int ndevices    = args.ndevices;
  const int skip_device = args.skip_device;

  // if the exact device is not set, but ndevices was given, assign round-robin
  // using on-node MPI rank
  if (use_gpu < 0) {
    auto const* local_rank_str =
        std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");  // OpenMPI
    if (!local_rank_str)
      local_rank_str = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");  // MVAPICH2
    if (!local_rank_str)
      local_rank_str = std::getenv("SLURM_LOCALID");  // SLURM

    auto const* ctest_kokkos_device_type =
        std::getenv("CTEST_KOKKOS_DEVICE_TYPE");  // CTest
    auto const* ctest_resource_group_count_str =
        std::getenv("CTEST_RESOURCE_GROUP_COUNT");  // CTest
    if (ctest_kokkos_device_type && ctest_resource_group_count_str &&
        local_rank_str) {
      // Use the device assigned by CTest
      use_gpu = get_ctest_gpu(local_rank_str);
    } else if (ndevices >= 0) {
      // Use the device assigned by the rank
      if (local_rank_str) {
        auto local_rank = std::stoi(local_rank_str);
        use_gpu         = local_rank % ndevices;
      } else {
        // user only gave use ndevices, but the MPI environment variable wasn't
        // set. start with GPU 0 at this point
        use_gpu = 0;
      }
    }
    // shift assignments over by one so no one is assigned to "skip_device"
    if (use_gpu >= skip_device) ++use_gpu;
  }
  return use_gpu;
}
#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP

bool is_unsigned_int(const char* str) {
  const size_t len = strlen(str);
  for (size_t i = 0; i < len; ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

void initialize_backends(const InitArguments& args) {
// This is an experimental setting
// For KNL in Flat mode this variable should be set, so that
// memkind allocates high bandwidth memory correctly.
#ifdef KOKKOS_ENABLE_HBWSPACE
  setenv("MEMKIND_HBW_NODES", "1", 0);
#endif

  // Protect declarations, to prevent "unused variable" warnings.
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS) || \
    defined(KOKKOS_ENABLE_HPX)
  const int num_threads = args.num_threads;
#endif
#if defined(KOKKOS_ENABLE_THREADS)
  const int use_numa = args.num_numa;
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  int use_gpu = get_gpu(args);
#endif  // defined( KOKKOS_ENABLE_CUDA )

#if defined(KOKKOS_ENABLE_OPENMP)
  if (std::is_same<Kokkos::OpenMP, Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::OpenMP, Kokkos::HostSpace::execution_space>::value) {
    Kokkos::OpenMP::impl_initialize(num_threads);
  } else {
    // std::cout << "Kokkos::initialize() fyi: OpenMP enabled but not
    // initialized" << std::endl ;
  }
#endif

#if defined(KOKKOS_ENABLE_THREADS)
  if (std::is_same<Kokkos::Threads, Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Threads,
                   Kokkos::HostSpace::execution_space>::value) {
    if (num_threads > 0) {
      if (use_numa > 0) {
        Kokkos::Threads::impl_initialize(num_threads, use_numa);
      } else {
        Kokkos::Threads::impl_initialize(num_threads);
      }
    } else {
      Kokkos::Threads::impl_initialize();
    }
    // std::cout << "Kokkos::initialize() fyi: Pthread enabled and initialized"
    // << std::endl ;
  } else {
    // std::cout << "Kokkos::initialize() fyi: Pthread enabled but not
    // initialized" << std::endl ;
  }
#endif

#if defined(KOKKOS_ENABLE_HPX)
  if (std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::HostSpace::execution_space>::value) {
    if (num_threads > 0) {
      Kokkos::Experimental::HPX::impl_initialize(num_threads);
    } else {
      Kokkos::Experimental::HPX::impl_initialize();
    }
    // std::cout << "Kokkos::initialize() fyi: HPX enabled and initialized" <<
    // std::endl ;
  } else {
    // std::cout << "Kokkos::initialize() fyi: HPX enabled but not initialized"
    // << std::endl ;
  }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  // Prevent "unused variable" warning for 'args' input struct.  If
  // Serial::initialize() ever needs to take arguments from the input
  // struct, you may remove this line of code.
  (void)args;

  // Always initialize Serial if it is configure time enabled
  Kokkos::Serial::impl_initialize();
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  if (std::is_same<Kokkos::Experimental::OpenMPTarget,
                   Kokkos::DefaultExecutionSpace>::value) {
    Kokkos::Experimental::OpenMPTarget().impl_initialize();
    // std::cout << "Kokkos::initialize() fyi: OpenMP enabled and initialized"
    // << std::endl ;
  } else {
    // std::cout << "Kokkos::initialize() fyi: OpenMP enabled but not
    // initialized" << std::endl ;
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  if (std::is_same<Kokkos::Cuda, Kokkos::DefaultExecutionSpace>::value ||
      0 < use_gpu) {
    if (use_gpu > -1) {
      Kokkos::Cuda::impl_initialize(Kokkos::Cuda::SelectDevice(use_gpu));
    } else {
      Kokkos::Cuda::impl_initialize();
    }
    // std::cout << "Kokkos::initialize() fyi: Cuda enabled and initialized" <<
    // std::endl ;
  }
#endif

#if defined(KOKKOS_ENABLE_HIP)
  if (std::is_same<Kokkos::Experimental::HIP,
                   Kokkos::DefaultExecutionSpace>::value ||
      0 < use_gpu) {
    if (use_gpu > -1) {
      Kokkos::Experimental::HIP::impl_initialize(
          Kokkos::Experimental::HIP::SelectDevice(use_gpu));
    } else {
      Kokkos::Experimental::HIP::impl_initialize();
    }
  }
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  if (std::is_same<Kokkos::Experimental::SYCL,
                   Kokkos::DefaultExecutionSpace>::value ||
      0 < use_gpu) {
    // FIXME_SYCL choose a specific device
    Kokkos::Experimental::SYCL::impl_initialize(
        Kokkos::Experimental::SYCL::SYCLDevice(cl::sycl::default_selector()));
  }
#endif
}

void initialize_profiling(const InitArguments&) {
  Kokkos::Profiling::initialize();
}

void pre_initialize_internal(const InitArguments& args) {
  if (args.disable_warnings) g_show_warnings = false;
}

void post_initialize_internal(const InitArguments& args) {
  initialize_profiling(args);
  g_is_initialized = true;
}

void initialize_internal(const InitArguments& args) {
  pre_initialize_internal(args);
  initialize_backends(args);
  post_initialize_internal(args);
}

void finalize_internal(const bool all_spaces = false) {
  typename decltype(finalize_hooks)::size_type numSuccessfulCalls = 0;
  while (!finalize_hooks.empty()) {
    auto f = finalize_hooks.top();
    try {
      f();
    } catch (...) {
      std::cerr << "Kokkos::finalize: A finalize hook (set via "
                   "Kokkos::push_finalize_hook) threw an exception that it did "
                   "not catch."
                   "  Per std::atexit rules, this results in std::terminate.  "
                   "This is "
                   "finalize hook number "
                << numSuccessfulCalls
                << " (1-based indexing) "
                   "out of "
                << finalize_hooks.size()
                << " to call.  Remember that "
                   "Kokkos::finalize calls finalize hooks in reverse order "
                   "from how they "
                   "were pushed."
                << std::endl;
      std::terminate();
    }
    finalize_hooks.pop();
    ++numSuccessfulCalls;
  }

  Kokkos::Profiling::finalize();

#if defined(KOKKOS_ENABLE_CUDA)
  if (std::is_same<Kokkos::Cuda, Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::Cuda::impl_is_initialized()) Kokkos::Cuda::impl_finalize();
  }
#else
  (void)all_spaces;
#endif

#if defined(KOKKOS_ENABLE_HIP)
  if (std::is_same<Kokkos::Experimental::HIP,
                   Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::Experimental::HIP::impl_is_initialized())
      Kokkos::Experimental::HIP::impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  if (std::is_same<Kokkos::Experimental::SYCL,
                   Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::Experimental::SYCL::impl_is_initialized())
      Kokkos::Experimental::SYCL::impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
  if (std::is_same<Kokkos::Experimental::OpenMPTarget,
                   Kokkos::DefaultExecutionSpace>::value ||
      all_spaces) {
    if (Kokkos::Experimental::OpenMPTarget().impl_is_initialized())
      Kokkos::Experimental::OpenMPTarget().impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  if (std::is_same<Kokkos::OpenMP, Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::OpenMP, Kokkos::HostSpace::execution_space>::value ||
      all_spaces) {
    if (Kokkos::OpenMP::impl_is_initialized()) Kokkos::OpenMP::impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_HPX)
  if (std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Experimental::HPX,
                   Kokkos::HostSpace::execution_space>::value ||
      all_spaces) {
    if (Kokkos::Experimental::HPX::impl_is_initialized())
      Kokkos::Experimental::HPX::impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_THREADS)
  if (std::is_same<Kokkos::Threads, Kokkos::DefaultExecutionSpace>::value ||
      std::is_same<Kokkos::Threads,
                   Kokkos::HostSpace::execution_space>::value ||
      all_spaces) {
    if (Kokkos::Threads::impl_is_initialized())
      Kokkos::Threads::impl_finalize();
  }
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  if (Kokkos::Serial::impl_is_initialized()) Kokkos::Serial::impl_finalize();
#endif

  g_is_initialized = false;
  g_show_warnings  = true;
}

void fence_internal() {
#if defined(KOKKOS_ENABLE_CUDA)
  Kokkos::Cuda::impl_static_fence();
#endif

#if defined(KOKKOS_ENABLE_HIP)
  Kokkos::Experimental::HIP::impl_static_fence();
#endif

#if defined(KOKKOS_ENABLE_SYCL)
  Kokkos::Experimental::SYCL().fence();
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  Kokkos::OpenMP::impl_static_fence();
#endif

#if defined(KOKKOS_ENABLE_HPX)
  Kokkos::Experimental::HPX().fence();
#endif

#if defined(KOKKOS_ENABLE_THREADS)
  Kokkos::Threads::impl_static_fence();
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  Kokkos::Serial::impl_static_fence();
#endif
}

bool check_arg(char const* arg, char const* expected) {
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  if (arg_len < exp_len) return false;
  if (std::strncmp(arg, expected, exp_len) != 0) return false;
  if (arg_len == exp_len) return true;
  /* if expected is "--threads", ignore "--threads-for-application"
     by checking this character          ---------^
     to see if it continues to make a longer name */
  if (std::isalnum(arg[exp_len]) || arg[exp_len] == '-' ||
      arg[exp_len] == '_') {
    return false;
  }
  return true;
}

bool check_int_arg(char const* arg, char const* expected, int* value) {
  if (!check_arg(arg, expected)) return false;
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  bool okay           = true;
  if (arg_len == exp_len || arg[exp_len] != '=') okay = false;
  char const* number = arg + exp_len + 1;
  if (!Impl::is_unsigned_int(number) || strlen(number) == 0) okay = false;
  *value = std::stoi(number);
  if (!okay) {
    std::ostringstream ss;
    ss << "Error: expecting an '=INT' after command line argument '" << expected
       << "'";
    ss << ". Raised by Kokkos::initialize(int narg, char* argc[]).";
    Impl::throw_runtime_exception(ss.str());
  }
  return true;
}

void warn_deprecated_command_line_argument(std::string deprecated,
                                           std::string valid) {
  std::cerr
      << "Warning: command line argument '" << deprecated
      << "' is deprecated. Use '" << valid
      << "' instead. Raised by Kokkos::initialize(int narg, char* argc[])."
      << std::endl;
}

unsigned get_process_id() {
#ifdef _WIN32
  return unsigned(GetCurrentProcessId());
#else
  return unsigned(getpid());
#endif
}

void parse_command_line_arguments(int& narg, char* arg[],
                                  InitArguments& arguments) {
  auto& num_threads      = arguments.num_threads;
  auto& numa             = arguments.num_numa;
  auto& device           = arguments.device_id;
  auto& ndevices         = arguments.ndevices;
  auto& skip_device      = arguments.skip_device;
  auto& disable_warnings = arguments.disable_warnings;

  bool kokkos_threads_found  = false;
  bool kokkos_numa_found     = false;
  bool kokkos_device_found   = false;
  bool kokkos_ndevices_found = false;

  int iarg = 0;

  while (iarg < narg) {
    if (check_int_arg(arg[iarg], "--kokkos-threads", &num_threads)) {
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      kokkos_threads_found = true;
      narg--;
    } else if (!kokkos_threads_found &&
               check_int_arg(arg[iarg], "--threads", &num_threads)) {
      iarg++;
    } else if (check_int_arg(arg[iarg], "--kokkos-numa", &numa)) {
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      kokkos_numa_found = true;
      narg--;
    } else if (!kokkos_numa_found &&
               check_int_arg(arg[iarg], "--numa", &numa)) {
      iarg++;
    } else if (check_int_arg(arg[iarg], "--kokkos-device-id", &device) ||
               check_int_arg(arg[iarg], "--kokkos-device", &device)) {
      if (check_arg(arg[iarg], "--kokkos-device")) {
        warn_deprecated_command_line_argument("--kokkos-device",
                                              "--kokkos-device-id");
      }
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      kokkos_device_found = true;
      narg--;
    } else if (!kokkos_device_found &&
               (check_int_arg(arg[iarg], "--device-id", &device) ||
                check_int_arg(arg[iarg], "--device", &device))) {
      if (check_arg(arg[iarg], "--device")) {
        warn_deprecated_command_line_argument("--device", "--device-id");
      }
      iarg++;
    } else if (check_arg(arg[iarg], "--kokkos-num-devices") ||
               check_arg(arg[iarg], "--num-devices") ||
               check_arg(arg[iarg], "--kokkos-ndevices") ||
               check_arg(arg[iarg], "--ndevices")) {
      if (check_arg(arg[iarg], "--ndevices")) {
        warn_deprecated_command_line_argument("--ndevices", "--num-devices");
      }
      if (check_arg(arg[iarg], "--kokkos-ndevices")) {
        warn_deprecated_command_line_argument("--kokkos-ndevices",
                                              "--kokkos-num-devices");
      }
      // Find the number of device (expecting --device=XX)
      if (!((strncmp(arg[iarg], "--kokkos-num-devices=", 21) == 0) ||
            (strncmp(arg[iarg], "--num-ndevices=", 14) == 0) ||
            (strncmp(arg[iarg], "--kokkos-ndevices=", 18) == 0) ||
            (strncmp(arg[iarg], "--ndevices=", 11) == 0)))
        throw_runtime_exception(
            "Error: expecting an '=INT[,INT]' after command line argument "
            "'--num-devices/--kokkos-num-devices'. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");

      char* num1      = strchr(arg[iarg], '=') + 1;
      char* num2      = strpbrk(num1, ",");
      int num1_len    = num2 == nullptr ? strlen(num1) : num2 - num1;
      char* num1_only = new char[num1_len + 1];
      strncpy(num1_only, num1, num1_len);
      num1_only[num1_len] = 0;

      if (!is_unsigned_int(num1_only) || (strlen(num1_only) == 0)) {
        throw_runtime_exception(
            "Error: expecting an integer number after command line argument "
            "'--kokkos-numdevices'. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");
      }
      if (check_arg(arg[iarg], "--kokkos-num-devices") ||
          check_arg(arg[iarg], "--kokkos-ndevices") || !kokkos_ndevices_found)
        ndevices = std::stoi(num1_only);
      delete[] num1_only;

      if (num2 != nullptr) {
        if ((!is_unsigned_int(num2 + 1)) || (strlen(num2) == 1))
          throw_runtime_exception(
              "Error: expecting an integer number after command line argument "
              "'--kokkos-num-devices=XX,'. Raised by "
              "Kokkos::initialize(int narg, char* argc[]).");

        if (check_arg(arg[iarg], "--kokkos-num-devices") ||
            check_arg(arg[iarg], "--kokkos-ndevices") || !kokkos_ndevices_found)
          skip_device = std::stoi(num2 + 1);
      }

      // Remove the --kokkos-num-devices argument from the list but leave
      // --num-devices
      if (check_arg(arg[iarg], "--kokkos-num-devices") ||
          check_arg(arg[iarg], "--kokkos-ndevices")) {
        for (int k = iarg; k < narg - 1; k++) {
          arg[k] = arg[k + 1];
        }
        kokkos_ndevices_found = true;
        narg--;
      } else {
        iarg++;
      }
    } else if (check_arg(arg[iarg], "--kokkos-disable-warnings")) {
      disable_warnings = true;
      for (int k = iarg; k < narg - 1; k++) {
        arg[k] = arg[k + 1];
      }
      narg--;
    } else if (check_arg(arg[iarg], "--kokkos-help") ||
               check_arg(arg[iarg], "--help")) {
      auto const help_message = R"(
      --------------------------------------------------------------------------------
      -------------Kokkos command line arguments--------------------------------------
      --------------------------------------------------------------------------------
      The following arguments exist also without prefix 'kokkos' (e.g. --help).
      The prefixed arguments will be removed from the list by Kokkos::initialize(),
      the non-prefixed ones are not removed. Prefixed versions take precedence over
      non prefixed ones, and the last occurrence of an argument overwrites prior
      settings.

      --kokkos-help                  : print this message
      --kokkos-disable-warnings      : disable kokkos warning messages
      --kokkos-threads=INT           : specify total number of threads or
                                       number of threads per NUMA region if
                                       used in conjunction with '--numa' option.
      --kokkos-numa=INT              : specify number of NUMA regions used by process.
      --kokkos-device-id=INT         : specify device id to be used by Kokkos.
      --kokkos-num-devices=INT[,INT] : used when running MPI jobs. Specify number of
                                       devices per node to be used. Process to device
                                       mapping happens by obtaining the local MPI rank
                                       and assigning devices round-robin. The optional
                                       second argument allows for an existing device
                                       to be ignored. This is most useful on workstations
                                       with multiple GPUs of which one is used to drive
                                       screen output.
      --------------------------------------------------------------------------------
)";
      std::cout << help_message << std::endl;

      // Remove the --kokkos-help argument from the list but leave --help
      if (check_arg(arg[iarg], "--kokkos-help")) {
        for (int k = iarg; k < narg - 1; k++) {
          arg[k] = arg[k + 1];
        }
        narg--;
      } else {
        iarg++;
      }
    } else
      iarg++;
  }
}

void parse_environment_variables(InitArguments& arguments) {
  auto& num_threads      = arguments.num_threads;
  auto& numa             = arguments.num_numa;
  auto& device           = arguments.device_id;
  auto& ndevices         = arguments.ndevices;
  auto& skip_device      = arguments.skip_device;
  auto& disable_warnings = arguments.disable_warnings;

  char* endptr;
  auto env_num_threads_str = std::getenv("KOKKOS_NUM_THREADS");
  if (env_num_threads_str != nullptr) {
    errno                = 0;
    auto env_num_threads = std::strtol(env_num_threads_str, &endptr, 10);
    if (endptr == env_num_threads_str)
      Impl::throw_runtime_exception(
          "Error: cannot convert KOKKOS_NUM_THREADS to an integer. Raised by "
          "Kokkos::initialize(int narg, char* argc[]).");
    if (errno == ERANGE)
      Impl::throw_runtime_exception(
          "Error: KOKKOS_NUM_THREADS out of range of representable values by "
          "an integer. Raised by Kokkos::initialize(int narg, char* argc[]).");
    if ((num_threads != -1) && (env_num_threads != num_threads))
      Impl::throw_runtime_exception(
          "Error: expecting a match between --kokkos-threads and "
          "KOKKOS_NUM_THREADS if both are set. Raised by "
          "Kokkos::initialize(int narg, char* argc[]).");
    else
      num_threads = env_num_threads;
  }
  auto env_numa_str = std::getenv("KOKKOS_NUMA");
  if (env_numa_str != nullptr) {
    errno         = 0;
    auto env_numa = std::strtol(env_numa_str, &endptr, 10);
    if (endptr == env_numa_str)
      Impl::throw_runtime_exception(
          "Error: cannot convert KOKKOS_NUMA to an integer. Raised by "
          "Kokkos::initialize(int narg, char* argc[]).");
    if (errno == ERANGE)
      Impl::throw_runtime_exception(
          "Error: KOKKOS_NUMA out of range of representable values by an "
          "integer. Raised by Kokkos::initialize(int narg, char* argc[]).");
    if ((numa != -1) && (env_numa != numa))
      Impl::throw_runtime_exception(
          "Error: expecting a match between --kokkos-numa and KOKKOS_NUMA if "
          "both are set. Raised by Kokkos::initialize(int narg, char* "
          "argc[]).");
    else
      numa = env_numa;
  }
  auto env_device_str = std::getenv("KOKKOS_DEVICE_ID");
  if (env_device_str != nullptr) {
    errno           = 0;
    auto env_device = std::strtol(env_device_str, &endptr, 10);
    if (endptr == env_device_str)
      Impl::throw_runtime_exception(
          "Error: cannot convert KOKKOS_DEVICE_ID to an integer. Raised by "
          "Kokkos::initialize(int narg, char* argc[]).");
    if (errno == ERANGE)
      Impl::throw_runtime_exception(
          "Error: KOKKOS_DEVICE_ID out of range of representable values by an "
          "integer. Raised by Kokkos::initialize(int narg, char* argc[]).");
    if ((device != -1) && (env_device != device))
      Impl::throw_runtime_exception(
          "Error: expecting a match between --kokkos-device and "
          "KOKKOS_DEVICE_ID if both are set. Raised by Kokkos::initialize(int "
          "narg, char* argc[]).");
    else
      device = env_device;
  }
  auto env_rdevices_str = std::getenv("KOKKOS_RAND_DEVICES");
  auto env_ndevices_str = std::getenv("KOKKOS_NUM_DEVICES");
  if (env_ndevices_str != nullptr || env_rdevices_str != nullptr) {
    errno = 0;
    if (env_ndevices_str != nullptr && env_rdevices_str != nullptr) {
      Impl::throw_runtime_exception(
          "Error: cannot specify both KOKKOS_NUM_DEVICES and "
          "KOKKOS_RAND_DEVICES. "
          "Raised by Kokkos::initialize(int narg, char* argc[]).");
    }
    int rdevices = -1;
    if (env_ndevices_str != nullptr) {
      auto env_ndevices = std::strtol(env_ndevices_str, &endptr, 10);
      if (endptr == env_ndevices_str)
        Impl::throw_runtime_exception(
            "Error: cannot convert KOKKOS_NUM_DEVICES to an integer. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");
      if (errno == ERANGE)
        Impl::throw_runtime_exception(
            "Error: KOKKOS_NUM_DEVICES out of range of representable values by "
            "an integer. Raised by Kokkos::initialize(int narg, char* "
            "argc[]).");
      if ((ndevices != -1) && (env_ndevices != ndevices))
        Impl::throw_runtime_exception(
            "Error: expecting a match between --kokkos-ndevices and "
            "KOKKOS_NUM_DEVICES if both are set. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");
      else
        ndevices = env_ndevices;
    } else {  // you set KOKKOS_RAND_DEVICES
      auto env_rdevices = std::strtol(env_rdevices_str, &endptr, 10);
      if (endptr == env_ndevices_str)
        Impl::throw_runtime_exception(
            "Error: cannot convert KOKKOS_RAND_DEVICES to an integer. Raised "
            "by Kokkos::initialize(int narg, char* argc[]).");
      if (errno == ERANGE)
        Impl::throw_runtime_exception(
            "Error: KOKKOS_RAND_DEVICES out of range of representable values "
            "by an integer. Raised by Kokkos::initialize(int narg, char* "
            "argc[]).");
      else
        rdevices = env_rdevices;
    }
    // Skip device
    auto env_skip_device_str = std::getenv("KOKKOS_SKIP_DEVICE");
    if (env_skip_device_str != nullptr) {
      errno                = 0;
      auto env_skip_device = std::strtol(env_skip_device_str, &endptr, 10);
      if (endptr == env_skip_device_str)
        Impl::throw_runtime_exception(
            "Error: cannot convert KOKKOS_SKIP_DEVICE to an integer. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");
      if (errno == ERANGE)
        Impl::throw_runtime_exception(
            "Error: KOKKOS_SKIP_DEVICE out of range of representable values by "
            "an integer. Raised by Kokkos::initialize(int narg, char* "
            "argc[]).");
      if ((skip_device != 9999) && (env_skip_device != skip_device))
        Impl::throw_runtime_exception(
            "Error: expecting a match between --kokkos-ndevices and "
            "KOKKOS_SKIP_DEVICE if both are set. Raised by "
            "Kokkos::initialize(int narg, char* argc[]).");
      else
        skip_device = env_skip_device;
    }
    if (rdevices > 0) {
      if (skip_device > 0 && rdevices == 1)
        Impl::throw_runtime_exception(
            "Error: cannot KOKKOS_SKIP_DEVICE the only KOKKOS_RAND_DEVICE. "
            "Raised by Kokkos::initialize(int narg, char* argc[]).");

      std::srand(get_process_id());
      while (device < 0) {
        int test_device = std::rand() % rdevices;
        if (test_device != skip_device) device = test_device;
      }
    }
  }
  char* env_disablewarnings_str = std::getenv("KOKKOS_DISABLE_WARNINGS");
  if (env_disablewarnings_str != nullptr) {
    std::string env_str(env_disablewarnings_str);  // deep-copies string
    for (char& c : env_str) {
      c = toupper(c);
    }
    if ((env_str == "TRUE") || (env_str == "ON") || (env_str == "1"))
      disable_warnings = true;
    else if (disable_warnings)
      Impl::throw_runtime_exception(
          "Error: expecting a match between --kokkos-disable-warnings and "
          "KOKKOS_DISABLE_WARNINGS if both are set. Raised by "
          "Kokkos::initialize(int narg, char* argc[]).");
  }
}

}  // namespace

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

void initialize(int& narg, char* arg[]) {
  InitArguments arguments;
  Impl::parse_command_line_arguments(narg, arg, arguments);
  Impl::parse_environment_variables(arguments);
  Impl::initialize_internal(arguments);
}

void initialize(InitArguments arguments) {
  Impl::parse_environment_variables(arguments);
  Impl::initialize_internal(arguments);
}

namespace Impl {

void pre_initialize(const InitArguments& args) {
  pre_initialize_internal(args);
}

void post_initialize(const InitArguments& args) {
  post_initialize_internal(args);
}
}  // namespace Impl

void push_finalize_hook(std::function<void()> f) { finalize_hooks.push(f); }

void finalize() { Impl::finalize_internal(); }

void finalize_all() {
  enum : bool { all_spaces = true };
  Impl::finalize_internal(all_spaces);
}

void fence() { Impl::fence_internal(); }

void print_configuration(std::ostream& out, const bool detail) {
  std::ostringstream msg;

  msg << "Kokkos Version:" << std::endl;
  msg << "  " << KOKKOS_VERSION / 10000 << "." << (KOKKOS_VERSION % 10000) / 100
      << "." << KOKKOS_VERSION % 100 << std::endl;

  msg << "Compiler:" << std::endl;
#ifdef KOKKOS_COMPILER_APPLECC
  msg << "  KOKKOS_COMPILER_APPLECC: " << KOKKOS_COMPILER_APPLECC << std::endl;
#endif
#ifdef KOKKOS_COMPILER_CLANG
  msg << "  KOKKOS_COMPILER_CLANG: " << KOKKOS_COMPILER_CLANG << std::endl;
#endif
#ifdef KOKKOS_COMPILER_CRAYC
  msg << "  KOKKOS_COMPILER_CRAYC: " << KOKKOS_COMPILER_CRAYC << std::endl;
#endif
#ifdef KOKKOS_COMPILER_GNU
  msg << "  KOKKOS_COMPILER_GNU: " << KOKKOS_COMPILER_GNU << std::endl;
#endif
#ifdef KOKKOS_COMPILER_IBM
  msg << "  KOKKOS_COMPILER_IBM: " << KOKKOS_COMPILER_IBM << std::endl;
#endif
#ifdef KOKKOS_COMPILER_INTEL
  msg << "  KOKKOS_COMPILER_INTEL: " << KOKKOS_COMPILER_INTEL << std::endl;
#endif
#ifdef KOKKOS_COMPILER_NVCC
  msg << "  KOKKOS_COMPILER_NVCC: " << KOKKOS_COMPILER_NVCC << std::endl;
#endif
#ifdef KOKKOS_COMPILER_PGI
  msg << "  KOKKOS_COMPILER_PGI: " << KOKKOS_COMPILER_PGI << std::endl;
#endif

  msg << "Architecture:" << std::endl;
#ifdef KOKKOS_ENABLE_ISA_KNC
  msg << "  KOKKOS_ENABLE_ISA_KNC: yes" << std::endl;
#else
  msg << "  KOKKOS_ENABLE_ISA_KNC: no" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_ISA_POWERPCLE
  msg << "  KOKKOS_ENABLE_ISA_POWERPCLE: yes" << std::endl;
#else
  msg << "  KOKKOS_ENABLE_ISA_POWERPCLE: no" << std::endl;
#endif
#ifdef KOKKOS_ENABLE_ISA_X86_64
  msg << "  KOKKOS_ENABLE_ISA_X86_64: yes" << std::endl;
#else
  msg << "  KOKKOS_ENABLE_ISA_X86_64: no" << std::endl;
#endif

  msg << "Devices:" << std::endl;
  msg << "  KOKKOS_ENABLE_CUDA: ";
#ifdef KOKKOS_ENABLE_CUDA
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_HIP: ";
#ifdef KOKKOS_ENABLE_HIP
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_SYCL: ";
#ifdef KOKKOS_ENABLE_SYCL
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_OPENMP: ";
#ifdef KOKKOS_ENABLE_OPENMP
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_HPX: ";
#ifdef KOKKOS_ENABLE_HPX
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_THREADS: ";
#ifdef KOKKOS_ENABLE_THREADS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_SERIAL: ";
#ifdef KOKKOS_ENABLE_SERIAL
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "Default Device:" << std::endl;
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_CUDA: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_CUDA
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HIP: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HIP
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SYCL: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SYCL
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL: ";
#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "Atomics:" << std::endl;
  msg << "  KOKKOS_ENABLE_CUDA_ATOMICS: ";
#ifdef KOKKOS_ENABLE_CUDA_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_GNU_ATOMICS: ";
#ifdef KOKKOS_ENABLE_GNU_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_INTEL_ATOMICS: ";
#ifdef KOKKOS_ENABLE_INTEL_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_OPENMP_ATOMICS: ";
#ifdef KOKKOS_ENABLE_OPENMP_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_WINDOWS_ATOMICS: ";
#ifdef KOKKOS_ENABLE_WINDOWS_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_SERIAL_ATOMICS: ";
#ifdef KOKKOS_ENABLE_SERIAL_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "Vectorization:" << std::endl;
  msg << "  KOKKOS_ENABLE_PRAGMA_IVDEP: ";
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_PRAGMA_LOOPCOUNT: ";
#ifdef KOKKOS_ENABLE_PRAGMA_LOOPCOUNT
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_PRAGMA_SIMD: ";
#ifdef KOKKOS_ENABLE_PRAGMA_SIMD
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_PRAGMA_UNROLL: ";
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_PRAGMA_VECTOR: ";
#ifdef KOKKOS_ENABLE_PRAGMA_VECTOR
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "Memory:" << std::endl;
  msg << "  KOKKOS_ENABLE_HBWSPACE: ";
#ifdef KOKKOS_ENABLE_HBWSPACE
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_INTEL_MM_ALLOC: ";
#ifdef KOKKOS_ENABLE_INTEL_MM_ALLOC
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_POSIX_MEMALIGN: ";
#ifdef KOKKOS_ENABLE_POSIX_MEMALIGN
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "Options:" << std::endl;
  msg << "  KOKKOS_ENABLE_ASM: ";
#ifdef KOKKOS_ENABLE_ASM
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CXX14: ";
#ifdef KOKKOS_ENABLE_CXX14
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CXX17: ";
#ifdef KOKKOS_ENABLE_CXX17
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CXX20: ";
#ifdef KOKKOS_ENABLE_CXX20
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: ";
#ifdef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_HWLOC: ";
#ifdef KOKKOS_ENABLE_HWLOC
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_LIBRT: ";
#ifdef KOKKOS_ENABLE_LIBRT
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_MPI: ";
#ifdef KOKKOS_ENABLE_MPI
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

#ifdef KOKKOS_ENABLE_CUDA
  msg << "Cuda Options:" << std::endl;
  msg << "  KOKKOS_ENABLE_CUDA_LAMBDA: ";
#ifdef KOKKOS_ENABLE_CUDA_LAMBDA
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CUDA_LDG_INTRINSIC: ";
#ifdef KOKKOS_ENABLE_CUDA_LDG_INTRINSIC
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE: ";
#ifdef KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CUDA_UVM: ";
#ifdef KOKKOS_ENABLE_CUDA_UVM
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CUSPARSE: ";
#ifdef KOKKOS_ENABLE_CUSPARSE
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
  msg << "  KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA: ";
#ifdef KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

#endif

#ifdef KOKKOS_ENABLE_HIP
  msg << "HIP Options:" << std::endl;
  msg << "  KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE: ";
#ifdef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif
#endif

  msg << "\nRuntime Configuration:" << std::endl;
#ifdef KOKKOS_ENABLE_CUDA
  Cuda::print_configuration(msg, detail);
#endif
#ifdef KOKKOS_ENABLE_HIP
  Experimental::HIP::print_configuration(msg, detail);
#endif
#ifdef KOKKOS_ENABLE_SYCL
  // FIXME_SYCL
  std::abort();
  // Experimental::SYCL::print_configuration(msg, detail);
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  OpenMP::print_configuration(msg, detail);
#endif
#ifdef KOKKOS_ENABLE_HPX
  Experimental::HPX::print_configuration(msg, detail);
#endif
#if defined(KOKKOS_ENABLE_THREADS)
  Threads::print_configuration(msg, detail);
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Serial::print_configuration(msg, detail);
#endif

  out << msg.str() << std::endl;
}

bool is_initialized() noexcept { return g_is_initialized; }

bool show_warnings() noexcept { return g_show_warnings; }

#ifdef KOKKOS_COMPILER_PGI
namespace Impl {
// Bizzarely, an extra jump instruction forces the PGI compiler to not have a
// bug related to (probably?) empty base optimization and/or aggregate
// construction.
void _kokkos_pgi_compiler_bug_workaround() {}
}  // end namespace Impl
#endif

}  // namespace Kokkos
