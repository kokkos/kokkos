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

#include <TestStdAlgorithmsCommon.hpp>
#include <std_algorithms/Kokkos_MinMaxOperations.hpp>
#include <std_algorithms/Kokkos_MinMaxElementOperations.hpp>

namespace Test {
namespace MinMaxOps {

namespace KE = Kokkos::Experimental;

// ----------------------------------------------------------
// test max()
// ----------------------------------------------------------
TEST(std_algorithms_min_max, max) {
  int a = 1;
  int b = 2;
  EXPECT_TRUE(KE::max(a, b) == 2);

  a = 3;
  b = 1;
  EXPECT_TRUE(KE::max(a, b) == 3);
}

template <class ViewType>
struct StdAlgoMinMaxOpsTestMax {
  ViewType m_view;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& ind) const {
    auto v1 = 10.;
    if (KE::max(v1, m_view(ind)) == 10.) {
      m_view(ind) = 6.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdAlgoMinMaxOpsTestMax(ViewType aIn) : m_view(aIn) {}
};

TEST(std_algorithms_min_max, max_within_parfor) {
  using view_t = Kokkos::View<double*>;
  view_t a("a", 10);

  StdAlgoMinMaxOpsTestMax<view_t> fnc(a);
  Kokkos::parallel_for(a.extent(0), fnc);
  auto a_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(a_h(0), 6.);
  }
}

// ----------------------------------------------------------
// test min()
// ----------------------------------------------------------
TEST(std_algorithms_min_max, min) {
  int a = 1;
  int b = 2;
  EXPECT_TRUE(KE::min(a, b) == 1);

  a = 3;
  b = 2;
  EXPECT_TRUE(KE::min(a, b) == 2);
}

template <class ViewType>
struct StdAlgoMinMaxOpsTestMin {
  ViewType m_view;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& ind) const {
    auto v1 = 10.;
    if (KE::min(v1, m_view(ind)) == 0.) {
      m_view(ind) = 8.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdAlgoMinMaxOpsTestMin(ViewType aIn) : m_view(aIn) {}
};

TEST(std_algorithms_min_max, min_within_parfor) {
  using view_t = Kokkos::View<double*>;
  view_t a("a", 10);

  StdAlgoMinMaxOpsTestMin<view_t> fnc(a);
  Kokkos::parallel_for(a.extent(0), fnc);
  auto a_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(a_h(0), 8.);
  }
}

// ----------------------------------------------------------
// test minmax()
// ----------------------------------------------------------
TEST(std_algorithms_min_max, minmax) {
  int a         = 1;
  int b         = 2;
  const auto& r = KE::minmax(a, b);
  EXPECT_TRUE(r.first == 1);
  EXPECT_TRUE(r.second == 2);

  a              = 3;
  b              = 2;
  const auto& r2 = KE::minmax(a, b);
  EXPECT_TRUE(r2.first == 2);
  EXPECT_TRUE(r2.second == 3);
}

template <class ViewType>
struct StdAlgoMinMaxOpsTestMinMax {
  ViewType m_view;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& ind) const {
    auto v1       = 7.;
    const auto& r = KE::minmax(v1, m_view(ind));
    m_view(ind)   = (double)(r.first - r.second);
  }

  KOKKOS_INLINE_FUNCTION
  StdAlgoMinMaxOpsTestMinMax(ViewType aIn) : m_view(aIn) {}
};

TEST(std_algorithms_min_max, minmax_within_parfor) {
  using view_t = Kokkos::View<double*>;
  view_t a("a", 10);

  StdAlgoMinMaxOpsTestMinMax<view_t> fnc(a);
  Kokkos::parallel_for(a.extent(0), fnc);
  auto a_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  for (int i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(a_h(0), -7.);
  }
}

// ----------------------------------------------------------
// test clamp()
// ----------------------------------------------------------
TEST(std_algorithms_min_max, clamp) {
  int a         = 1;
  int b         = 2;
  int c         = 19;
  const auto& r = KE::clamp(a, b, c);
  EXPECT_TRUE(&r == &b);
  EXPECT_TRUE(r == b);

  a              = 5;
  b              = -2;
  c              = 3;
  const auto& r2 = KE::clamp(a, b, c);
  EXPECT_TRUE(&r2 == &c);
  EXPECT_TRUE(r2 == c);

  a              = 5;
  b              = -2;
  c              = 7;
  const auto& r3 = KE::clamp(a, b, c);
  EXPECT_TRUE(&r3 == &a);
  EXPECT_TRUE(r3 == a);
}

template <class ViewType>
struct StdAlgoMinMaxOpsTestClamp {
  ViewType m_view;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& ind) const {
    m_view(ind)   = 10.;
    const auto b  = -2.;
    const auto c  = 3.;
    const auto& r = KE::clamp(m_view(ind), b, c);
    m_view(ind)   = (double)(r);
  }

  KOKKOS_INLINE_FUNCTION
  StdAlgoMinMaxOpsTestClamp(ViewType aIn) : m_view(aIn) {}
};

TEST(std_algorithms_min_max, clamp_within_parfor) {
  using view_t = Kokkos::View<double*>;
  view_t a("a", 10);

  StdAlgoMinMaxOpsTestClamp<view_t> fnc(a);
  Kokkos::parallel_for(a.extent(0), fnc);
  auto a_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
  for (std::size_t i = 0; i < a.extent(0); ++i) {
    EXPECT_DOUBLE_EQ(a_h(0), 3.);
  }
}

}  // namespace MinMaxOps
}  // namespace Test
