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

#include <hip/TestHIP_Category.hpp>
#include <Test_InterOp_Streams.hpp>

namespace Test {
// Test Interoperability with HIP Streams
// The difference with the CUDA tests are: raw HIP vs raw CUDA and no launch
// bound in HIP due to an error when computing the block size.
TEST(hip, raw_hip_streams) {
  hipStream_t stream;
  hipStreamCreate(&stream);
  Kokkos::InitArguments arguments{-1, -1, -1, false};
  Kokkos::initialize(arguments);
  int* p;
  hipMalloc(&p, sizeof(int) * 100);
  using MemorySpace = typename TEST_EXECSPACE::memory_space;

  {
    TEST_EXECSPACE space0(stream);
    Kokkos::View<int*, TEST_EXECSPACE> v(p, 100);
    Kokkos::deep_copy(space0, v, 5);
    int sum;

    Kokkos::parallel_for("Test::hip::raw_hip_stream::Range",
                         Kokkos::RangePolicy<TEST_EXECSPACE>(space0, 0, 100),
                         FunctorRange<MemorySpace>(v));
    Kokkos::parallel_reduce("Test::hip::raw_hip_stream::RangeReduce",
                            Kokkos::RangePolicy<TEST_EXECSPACE>(space0, 0, 100),
                            FunctorRangeReduce<MemorySpace>(v), sum);
    space0.fence();
    ASSERT_EQ(600, sum);

    Kokkos::parallel_for("Test::hip::raw_hip_stream::MDRange",
                         Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(
                             space0, {0, 0}, {10, 10}),
                         FunctorMDRange<MemorySpace>(v));
    Kokkos::parallel_reduce(
        "Test::hip::raw_hip_stream::MDRangeReduce",
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>(space0, {0, 0},
                                                               {10, 10}),
        FunctorMDRangeReduce<MemorySpace>(v), sum);
    space0.fence();
    ASSERT_EQ(700, sum);

    Kokkos::parallel_for("Test::hip::raw_hip_stream::Team",
                         Kokkos::TeamPolicy<TEST_EXECSPACE>(space0, 10, 10),
                         FunctorTeam<MemorySpace, TEST_EXECSPACE>(v));
    Kokkos::parallel_reduce("Test::hip::raw_hip_stream::Team",
                            Kokkos::TeamPolicy<TEST_EXECSPACE>(space0, 10, 10),
                            FunctorTeamReduce<MemorySpace, TEST_EXECSPACE>(v),
                            sum);
    space0.fence();
    ASSERT_EQ(800, sum);
  }
  Kokkos::finalize();
  offset_streams<<<100, 64, 0, stream>>>(p);
  HIP_SAFE_CALL(hipDeviceSynchronize());
  hipStreamDestroy(stream);

  int h_p[100];
  hipMemcpy(h_p, p, sizeof(int) * 100, hipMemcpyDefault);
  HIP_SAFE_CALL(hipDeviceSynchronize());
  int64_t sum        = 0;
  int64_t sum_expect = 0;
  for (int i = 0; i < 100; i++) {
    sum += h_p[i];
    sum_expect += 8 + i;
  }

  ASSERT_EQ(sum, sum_expect);
}
}  // namespace Test
