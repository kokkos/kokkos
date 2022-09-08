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

#if defined(_WIN32)  // windows system
#include <windows.h>
unsigned getBytesPerPage() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
}
#else  // unix/posix system
#include <unistd.h>
unsigned getBytesPerPage() { return sysconf(_SC_PAGESIZE); }
#endif

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <numeric>
#include <iostream>

namespace {
void printTimings(std::ostream& out, std::vector<uint64_t> const& tr,
                  uint64_t threshold = std::numeric_limits<uint64_t>::max()) {
  out << "TimingResult contains " << tr.size() << " results:\n";
  for (auto it = tr.begin(); it != tr.end(); ++it) {
    out << "Duration of loop " << it - tr.begin() << " is " << *it
        << " clock cycles. ";
    if ((*it) > threshold) out << "Migration assumed.";

    out << "\n";
  }
}

template <typename T>
T computeMean(std::vector<T> const& results) {
  return std::accumulate(results.begin(), results.end(), T{}) / results.size();
}

template <typename ViewType>
class IncrementFunctor {
 private:
  using index_type = decltype(std::declval<ViewType>().size());
  ViewType view_;

 public:
  IncrementFunctor() = delete;

  explicit IncrementFunctor(ViewType view) : view_(view) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type idx, uint64_t& clockTics) const {
    uint64_t start = Kokkos::Impl::clock_tic();
    ++view_(idx);
    clockTics += Kokkos::Impl::clock_tic() - start;
  }
};

// TIMING CAPTURED KERNEL
// PREMISE: This kernel should always be memory bound, as we are measuring
// memory access times. The compute load of an increment is small enough on
// current hardware but this could be different for new hardware. As we count
// the clocks in the kernel, the core frequency of the device has to be fast
// enough to guarante that the kernel stays memory bound.
template <typename ExecSpace, typename ViewType>
std::vector<uint64_t> incrementInLoop(ViewType& view,
                                      unsigned int numRepetitions) {
  using index_type = decltype(view.size());
  std::vector<uint64_t> results;

  Kokkos::fence();
  for (unsigned i = 0; i < numRepetitions; ++i) {
    uint64_t sum_clockTics;
    IncrementFunctor<ViewType> func(view);
    Kokkos::parallel_reduce(
        "increment",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<index_type>>{
            0, view.size()},
        func, sum_clockTics);
    Kokkos::fence();
    results.push_back(sum_clockTics / view.size());
  }
  return results;
}

TEST(defaultdevicetype, shared_space) {
  ASSERT_TRUE(KOKKOS_HAS_SHARED_SPACE);
  ASSERT_TRUE(Kokkos::has_shared_space);

  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>)
    GTEST_SKIP() << "Skipping as host and device are the same space";

#if defined(KOKKOS_ARCH_VEGA900) || defined(KOKKOS_ARCH_VEGA906) || \
    defined(KOKKOS_ARCH_VEGA908)
  GTEST_SKIP()
      << "skipping because specified arch does not support page migration";
#endif
#if defined(KOKKOS_ENABLE_SYCL) && !defined(KOKKOS_ARCH_INTEL_GPU)
  GTEST_SKIP()
      << "skipping because clock_tic is only defined for sycl+intel gpu";
#endif

  const unsigned int numRepetitions      = 10;
  const unsigned int numDeviceHostCycles = 3;
  double threshold                       = 1.5;
  unsigned int numPages                  = 100;
  size_t numBytes                        = numPages * getBytesPerPage();

  // we rely on this to allocate the right amount of memory in the ALLOCATION
  ASSERT_TRUE(numBytes % sizeof(int) == 0);

  // ALLOCATION
  Kokkos::View<int*, Kokkos::SharedSpace> sharedData("sharedData",
                                                     numBytes / sizeof(int));
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace::memory_space>
      defaultExecData("defaultExecData", numBytes / sizeof(int));
  Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace::memory_space>
      defaultHostExecData("defaultHostExecData", numBytes / sizeof(int));
  Kokkos::fence();

  // GET DEFAULT EXECSPACE LOCAL TIMINGS
  auto defaultExecLocalTimings = incrementInLoop<Kokkos::DefaultExecutionSpace>(
      defaultExecData, numRepetitions);

  // GET DEFAULT HOSTEXECSPACE LOCAL TIMINGS
  auto defaultHostExecLocalTimings =
      incrementInLoop<Kokkos::DefaultHostExecutionSpace>(defaultHostExecData,
                                                         numRepetitions);

  // GET PAGE MIGRATING TIMINGS DATA
  std::vector<decltype(defaultExecLocalTimings)> defaultExecSharedTimings{};
  std::vector<decltype(defaultExecLocalTimings)> defaultHostExecSharedTimings{};
  for (unsigned i = 0; i < numDeviceHostCycles; ++i) {
    // GET RESULTS DEVICE
    defaultExecSharedTimings.push_back(
        incrementInLoop<Kokkos::DefaultExecutionSpace>(sharedData,
                                                       numRepetitions));

    // GET RESULTS HOST
    defaultHostExecSharedTimings.push_back(
        incrementInLoop<Kokkos::DefaultHostExecutionSpace>(sharedData,
                                                           numRepetitions));
  }

  // COMPUTE STATISTICS OF HOST AND DEVICE LOCAL KERNELS
  auto defaultExecLocalMean     = computeMean(defaultExecLocalTimings);
  auto defaultHostExecLocalMean = computeMean(defaultHostExecLocalTimings);

  // ASSESS RESULTS
  bool fastAsLocalOnRepeatedAccess = true;

  for (unsigned cycle = 0; cycle < numDeviceHostCycles; ++cycle) {
    std::for_each(std::next(defaultExecSharedTimings[cycle].begin()),
                  defaultExecSharedTimings[cycle].end(),
                  [&](const double timing) {
                    (timing < threshold * defaultExecLocalMean)
                        ? fastAsLocalOnRepeatedAccess &= true
                        : fastAsLocalOnRepeatedAccess &= false;
                  });

    std::for_each(std::next(defaultHostExecSharedTimings[cycle].begin()),
                  defaultHostExecSharedTimings[cycle].end(),
                  [&](const double timing) {
                    (timing < threshold * defaultExecLocalMean)
                        ? fastAsLocalOnRepeatedAccess &= true
                        : fastAsLocalOnRepeatedAccess &= false;
                  });
  }

  // CHECK IF PASSED
  bool passed = (fastAsLocalOnRepeatedAccess);

  // PRINT IF NOT PASSED
  if (!passed) {
    std::cout << "Page size as reported by os: " << getBytesPerPage()
              << " bytes \n";
    std::cout << "Allocating " << numPages
              << " pages of memory in SharedSpace.\n";

    std::cout << "Behavior found: \n";
    std::cout << "SharedSpace is as fast as local space on repeated access: "
              << fastAsLocalOnRepeatedAccess << ", we expect true \n\n";

    std::cout
        << "Please look at the following timings. The first access in a "
           "different ExecutionSpace is not evaluated for the test. As we "
           "expect the memory to migrate during the first access it might have "
           "a higher cycle count than subsequent accesses, depending on your "
           "hardware. If the cycles are more than "
        << threshold
        << " times the cycles for pure local memory access, we assume a page "
           "migration happened.\n\n";

    std::cout << "################SHARED SPACE####################\n";
    for (unsigned cycle = 0; cycle < numDeviceHostCycles; ++cycle) {
      std::cout << "DefaultExectionSpace timings of run " << cycle << ":\n";
      printTimings(std::cout, defaultExecSharedTimings[cycle],
                   threshold * defaultExecLocalMean);
      std::cout << "DefaultHostExecutionSpace timings of run " << cycle
                << ":\n";
      printTimings(std::cout, defaultHostExecSharedTimings[cycle],
                   threshold * defaultHostExecLocalMean);
    }
    std::cout << "################LOCAL SPACE####################\n";
    printTimings(std::cout, defaultExecLocalTimings);
    printTimings(std::cout, defaultHostExecLocalTimings);
  }
  ASSERT_TRUE(passed);
}
}  // namespace
