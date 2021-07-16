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
#include <gtest/gtest.h>
#include <cstdio>
#include <PerfTest_Category.hpp>

namespace Test {

template <class IndexT, class DataT>
void run_parallel_for_range_policy_test() {
  std::vector<int64_t> lengths{0, 1, 7032, 100390, 9181911};
  int repeats = 20;

  for (int64_t N : lengths) {
    // Create Data
    Kokkos::View<DataT*> a("A", N), b("B", N), c("C", N);
    Kokkos::deep_copy(a, DataT(1));
    Kokkos::deep_copy(b, DataT(2));

    // Precreate Lambda for using the same in warmup and benchmark
    auto lambda = KOKKOS_LAMBDA(IndexT i) { c[i] = a[i] + b[i]; };

    Kokkos::RangePolicy<Kokkos::IndexType<IndexT>> policy(0, N);

    // Warmup loop
    Kokkos::parallel_for("PerfTest::ParallelFor::WarmUp", policy, lambda);

    Kokkos::fence();

    // Benchmark run
    Kokkos::Timer timer;
    for (int r = 0; r < repeats; r++) {
      Kokkos::parallel_for("PerfTest::ParallelFor::Benchmark", policy, lambda);
    }
    Kokkos::fence();

    // Performance output
    double time = timer.seconds();
    double gigabytes_moved =
        1. * repeats * N * 3 * sizeof(DataT) / 1024. / 1024. / 1024.;

    double time_per_iter = time / repeats;
    double GBs           = gigabytes_moved / time;

    printf("ParallelFor %s %s Length: %li TimePerIter: %e GB/s: %lf\n",
           typeid(IndexT).name(), typeid(DataT).name(), N, time_per_iter, GBs);
  }
}

TEST(default_exec, ParallelFor_RangePolicy) {
  run_parallel_for_range_policy_test<int, int>();
  run_parallel_for_range_policy_test<int, double>();
  run_parallel_for_range_policy_test<int64_t, double>();
}

}  // namespace Test
