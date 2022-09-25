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

#ifndef KOKKOS_TEST_MultiThreadsMDRange_HPP
#define KOKKOS_TEST_MultiThreadsMDRange_HPP

#include <Kokkos_Core.hpp>

#include <thread>

namespace Test {
namespace {
template <typename ExecSpace>
struct TestMultiThreadsMDRange {
  using value_type = int;

  using view_type = Kokkos::View<value_type **, ExecSpace>;

  struct VerifyInitTag {};
  struct ResetTag {};
  struct VerifyResetTag {};
  struct OffsetTag {};
  struct VerifyOffsetTag {};

  int M;
  int N;
  view_type results;
  view_type results_scan;

  TestMultiThreadsMDRange(size_t const M_, size_t const N_)
      : M(M_),
        N(N_),
        results(Kokkos::view_alloc(Kokkos::WithoutInitializing, "results"), M,
                N),
        results_scan(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "results_scan"), M,
            N) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i, int j) const { results(i, j) = i + j; }

  void check_result_for() {
    auto results_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        ASSERT_EQ(results_host(i, j), i + j);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i, int const j, value_type &lsum) const {
    lsum += results(i, j);
  }
};
}  // namespace

TEST(TEST_CATEGORY, multiple_mdrange_for) {
  size_t const M_0 = 10000;
  size_t const M_1 = 2 * M_0;
  size_t const N   = 2;
  TestMultiThreadsMDRange<TEST_EXECSPACE> mdrange_0(M_0, N);
  TestMultiThreadsMDRange<TEST_EXECSPACE> mdrange_1(M_1, N);
  using range_type = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<N>>;
  auto f_0         = [&]() {
    range_type range({0, 0}, {M_0, N});
    Kokkos::parallel_for(range, mdrange_0);
  };
  auto f_1 = [&]() {
    range_type range({0, 0}, {M_1, N});
    Kokkos::parallel_for(range, mdrange_1);
  };

  std::thread t_0(f_0);
  std::thread t_1(f_1);
  t_0.join();
  t_1.join();

  mdrange_0.check_result_for();
  mdrange_1.check_result_for();
}

TEST(TEST_CATEGORY, multiple_mdrange_reduce) {
  size_t constexpr M_0 = 10000;
  size_t constexpr M_1 = 2 * M_0;
  size_t constexpr N   = 2;
  int total_0          = 0;
  int total_1          = 0;
  int ref_0            = ((M_0 - 1) * M_0 + M_0 * (M_0 + 1)) / 2;
  int ref_1            = ((M_1 - 1) * M_1 + M_1 * (M_1 + 1)) / 2;
  TestMultiThreadsMDRange<TEST_EXECSPACE> mdrange_0(M_0, N);
  TestMultiThreadsMDRange<TEST_EXECSPACE> mdrange_1(M_1, N);
  using range_type = Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<N>>;
  range_type range_0({0, 0}, {M_0, N});
  range_type range_1({0, 0}, {M_1, N});
  Kokkos::parallel_for(range_0, mdrange_0);
  Kokkos::parallel_for(range_1, mdrange_1);
  auto f_0 = [&]() { Kokkos::parallel_reduce(range_0, mdrange_0, total_0); };
  auto f_1 = [&]() { Kokkos::parallel_reduce(range_1, mdrange_1, total_1); };

  std::thread t_0(f_0);
  std::thread t_1(f_1);
  t_0.join();
  t_1.join();

  ASSERT_EQ(total_0, ref_0);
  ASSERT_EQ(total_1, ref_1);
}
}  // namespace Test

#endif
