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
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace {

template <typename ViewType>
struct Increment {
  ViewType view_;

  template <typename ExecutionSpace>
  explicit Increment(ExecutionSpace, ViewType view) : view_(view) {
    Kokkos::parallel_for(
        "increment",
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_t>>{
            0, view_.size()},
        *this);
  }

  KOKKOS_FUNCTION
  void operator()(const size_t idx) const { ++view_(idx); }
};

template <typename ViewType>
struct CheckResult {
  ViewType view_;
  int targetVal_;
  unsigned numErrors = 0;

  template <typename ExecutionSpace>
  CheckResult(ExecutionSpace, ViewType view, int targetVal)
      : view_(view), targetVal_(targetVal) {
    Kokkos::parallel_reduce(
        "check",
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<size_t>>{
            0, view_.size()},
        *this, Kokkos::Sum<unsigned>(numErrors));
  }

  KOKKOS_FUNCTION
  void operator()(const size_t idx, unsigned& errors) const {
    if (view_(idx) != targetVal_) ++errors;
  }
};

TEST(defaultdevicetype, shared_host_pinned_space) {
  ASSERT_TRUE(Kokkos::has_shared_host_pinned_space);

  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>)
    GTEST_SKIP() << "Skipping as host and device are the same space";

  const unsigned int numDeviceHostCycles = 3;
  size_t numInts                         = 1024;

  using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
  using HostExecutionSpace   = Kokkos::DefaultHostExecutionSpace;

  // ALLOCATION
  Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> sharedData("sharedData",
                                                               numInts);
  // MAIN LOOP
  unsigned incrementCount = 0;

  for (unsigned i = 0; i < numDeviceHostCycles; ++i) {
    // INCREMENT DEVICE
    Increment incrementOnDevice(DeviceExecutionSpace{}, sharedData);
    ++incrementCount;
    Kokkos::fence();
    // CHECK RESULTS HOST
    ASSERT_EQ(
        CheckResult(HostExecutionSpace{}, sharedData, incrementCount).numErrors,
        0u)
        << "Changes to SharedHostPinnedSpace made on device not visible to "
           "host. Iteration "
        << i << " of " << numDeviceHostCycles;

    // INCREMENT HOST
    Increment incrementOnHost(HostExecutionSpace{}, sharedData);
    ++incrementCount;
    Kokkos::fence();
    // CHECK RESULTS Device
    ASSERT_EQ(CheckResult(DeviceExecutionSpace{}, sharedData, incrementCount)
                  .numErrors,
              0u)
        << "Changes to SharedHostPinnedSpace made on host not visible to "
           "device. Iteration "
        << i << " of " << numDeviceHostCycles;
  }
}
}  // namespace
