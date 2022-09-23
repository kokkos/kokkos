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

#ifndef KOKKOS_UNITTEST_MDSPAN_HPP
#define KOKKOS_UNITTEST_MDSPAN_HPP

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN
// Look for the right mdspan
#if __has_include(<mdspan>)
#include <mdspan>
namespace mdspan_ns = std;
#else
#include <experimental/mdspan>
namespace mdspan_ns = std::experimental;
#endif

namespace {
void test_mdspan_minimal_functional() {
  int N = 100;
  Kokkos::View<int*, TEST_EXECSPACE> a("A", N);
  Kokkos::parallel_for(
      "FillSequence", Kokkos::RangePolicy<TEST_EXECSPACE>(0, N),
      KOKKOS_LAMBDA(int i) { a(i) = i; });

  mdspan_ns::mdspan<int, mdspan_ns::dextents<int, 1>> a_mds(a.data(), N);
  int errors;
  Kokkos::parallel_reduce(
      "CheckMinimalMDSpan", Kokkos::RangePolicy<TEST_EXECSPACE>(0, N),
      KOKKOS_LAMBDA(int i, int& err) {
        mdspan_ns::mdspan<int, mdspan_ns::dextents<int, 1>> b_mds(a.data(), N);
#ifdef KOKKOS_ENABLE_CXX23
        if (a_mds[i] != i) err++;
        if (b_mds[i] != i) err++;
#else
        if (a_mds(i) != i) err++;
        if (b_mds(i) != i) err++;
#endif
      },
      errors);
  ASSERT_EQ(errors, 0);
}
}  // namespace
#endif

namespace {

TEST(TEST_CATEGORY, mdspan_minimal_functional) {
#ifndef KOKKOS_ENABLE_IMPL_MDSPAN
  GTEST_SKIP() << "mdspan not enabled";
#else
  test_mdspan_minimal_functional();
#endif
}

}  // namespace Test

#endif
