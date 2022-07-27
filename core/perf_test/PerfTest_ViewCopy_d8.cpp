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

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <PerfTest_ViewCopy.hpp>  // Test::deepcopy_view

#include <cmath>

namespace Test {

void report_results(benchmark::State& state, double time) {
  state.SetIterationTime(time);

  const auto N8        = std::pow(state.range(0), 8);
  const auto size      = N8 * 8 / 1024 / 1024;
  state.counters["MB"] = benchmark::Counter(size, benchmark::Counter::kDefaults,
                                            benchmark::Counter::OneK::kIs1024);
  state.counters["GB/s"] =
      benchmark::Counter(2 * size / 1024 / time, benchmark::Counter::kDefaults,
                         benchmark::Counter::OneK::kIs1024);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8(benchmark::State& state) {
  const auto N = state.range(0);
  const auto R = state.range(1);

  const int N1 = N;
  Kokkos::View<double********, LayoutA> a("A8", N1, N1, N1, N1, N1, N1, N1, N1);
  Kokkos::View<double********, LayoutB> b("B8", N1, N1, N1, N1, N1, N1, N1, N1);

  for (auto _ : state) {
    const auto elapsed_seconds = Test::deepcopy_view(a, b, R);

    report_results(state, elapsed_seconds);
  }
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8_Raw(benchmark::State& state) {
  const auto N = state.range(0);
  const auto R = state.range(1);

  const int N1 = N;
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;
  const int N8 = N4 * N4;
  Kokkos::View<double*, LayoutA> a("A1", N8);
  Kokkos::View<double*, LayoutB> b("B1", N8);
  double* const a_ptr       = a.data();
  const double* const b_ptr = b.data();

  for (auto _ : state) {
    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for(
          N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = b_ptr[i]; });
    }
    Kokkos::fence();

    report_results(state, timer.seconds());
  }
}

BENCHMARK(ViewDeepCopy_Rank8<Kokkos::LayoutRight, Kokkos::LayoutLeft>)
    ->ArgNames({"N", "R"})
    ->Args({10, 1})
    ->UseManualTime();

#if defined(KOKKOS_ENABLE_CUDA_LAMBDA) || !defined(KOKKOS_ENABLE_CUDA)
BENCHMARK(ViewDeepCopy_Rank8_Raw<Kokkos::LayoutRight, Kokkos::LayoutLeft>)
    ->ArgNames({"N", "R"})
    ->Args({10, 1})
    ->UseManualTime();
#endif

}  // namespace Test
