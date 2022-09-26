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

#ifndef KOKKOS_CORE_PERFTEST_BENCHMARK_VIEW_COPY_HPP
#define KOKKOS_CORE_PERFTEST_BENCHMARK_VIEW_COPY_HPP

#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>
#include <cmath>

namespace Test {

/**
 * \brief Mark the label as a figure of merit.
 */
inline std::string benchmark_fom(const std::string& label) {
  return "FOM: " + label;
}

inline void report_results(benchmark::State& state, std::size_t num_elems,
                           double time) {
  state.SetIterationTime(time);

  // data size in megabytes
  const auto size = 1.0 * num_elems * sizeof(double) / 1024 / 1024;
  // data processed in gigabytes
  const auto data_processed = 2 * size / 1024;

  state.counters["MB"] = benchmark::Counter(size, benchmark::Counter::kDefaults,
                                            benchmark::Counter::OneK::kIs1024);
  state.counters[benchmark_fom("GB/s")] = benchmark::Counter(
      data_processed, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::OneK::kIs1024);
}

template <class ViewTypeA, class ViewTypeB>
void deepcopy_view(ViewTypeA& a, ViewTypeB& b, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::deep_copy(a, b);
    report_results(state, a.size(), timer.seconds());
  }
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank1(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);

  Kokkos::View<double*, LayoutA> a("A1", N8);
  Kokkos::View<double*, LayoutB> b("B1", N8);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank2(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;
  const int N4 = N2 * N2;

  Kokkos::View<double**, LayoutA> a("A2", N4, N4);
  Kokkos::View<double**, LayoutB> b("B2", N4, N4);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank3(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;
  const int N3 = N2 * N1;

  Kokkos::View<double***, LayoutA> a("A3", N3, N3, N2);
  Kokkos::View<double***, LayoutB> b("B3", N3, N3, N2);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank4(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double****, LayoutA> a("A4", N2, N2, N2, N2);
  Kokkos::View<double****, LayoutB> b("B4", N2, N2, N2, N2);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank5(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double*****, LayoutA> a("A5", N2, N2, N1, N1, N2);
  Kokkos::View<double*****, LayoutB> b("B5", N2, N2, N1, N1, N2);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank6(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double******, LayoutA> a("A6", N2, N1, N1, N1, N1, N2);
  Kokkos::View<double******, LayoutB> b("B6", N2, N1, N1, N1, N1, N2);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank7(benchmark::State& state) {
  const int N1 = state.range(0);
  const int N2 = N1 * N1;

  Kokkos::View<double*******, LayoutA> a("A7", N2, N1, N1, N1, N1, N1, N1);
  Kokkos::View<double*******, LayoutB> b("B7", N2, N1, N1, N1, N1, N1, N1);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Rank8(benchmark::State& state) {
  const int N1 = state.range(0);

  Kokkos::View<double********, LayoutA> a("A8", N1, N1, N1, N1, N1, N1, N1, N1);
  Kokkos::View<double********, LayoutB> b("B8", N1, N1, N1, N1, N1, N1, N1, N1);

  deepcopy_view(a, b, state);
}

template <class LayoutA, class LayoutB>
static void ViewDeepCopy_Raw(benchmark::State& state) {
  const int N8 = std::pow(state.range(0), 8);

  Kokkos::View<double*, LayoutA> a("A1", N8);
  Kokkos::View<double*, LayoutB> b("B1", N8);
  double* const a_ptr       = a.data();
  const double* const b_ptr = b.data();

  for (auto _ : state) {
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        N8, KOKKOS_LAMBDA(const int& i) { a_ptr[i] = b_ptr[i]; });
    Kokkos::fence();

    report_results(state, a.size(), timer.seconds());
  }
}

}  // namespace Test

#endif
