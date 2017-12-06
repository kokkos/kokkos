/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_TEST_REDUCTIONVIEW_HPP
#define KOKKOS_TEST_REDUCTIONVIEW_HPP

#include <Kokkos_ReductionView.hpp>

namespace Test {

template <typename ExecSpace, typename Layout, int duplication, int contribution>
void test_reduction_view_config(int n)
{
  Kokkos::View<double *[3], Layout, ExecSpace> original_view("original_view", n);
  {
    auto reduction_view = Kokkos::Experimental::create_reduction_view
      < Kokkos::Experimental::ReductionSum
      , duplication
      , contribution
      > (original_view);
    auto policy = Kokkos::RangePolicy<ExecSpace, int>(0, n);
#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
    auto f = KOKKOS_LAMBDA(int i) {
      auto reduction_access = reduction_view.access();
      auto reduction_access_atomic = reduction_view.template access<Kokkos::Experimental::ReductionAtomic>();
      for (int j = 0; j < 10; ++j) {
        auto k = (i + j) % n;
        reduction_access(k, 0) += 4.2;
        reduction_access_atomic(k, 1) += 2.0;
        reduction_access(k, 2) += 1.0;
      }
    };
    Kokkos::parallel_for(policy, f, "reduction_view_test");
#endif
    Kokkos::Experimental::contribute(original_view, reduction_view);
    reduction_view.reset_except(original_view);
#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
    Kokkos::parallel_for(policy, f, "reduction_view_test");
#endif
    Kokkos::Experimental::contribute(original_view, reduction_view);
  }
  auto host_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), original_view);
  for (typename decltype(host_view)::size_type i = 0; i < host_view.dimension_0(); ++i) {
    auto val0 = host_view(i, 0);
    auto val1 = host_view(i, 1);
    auto val2 = host_view(i, 2);
#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
    EXPECT_TRUE(std::fabs((val0 - 84.0) / 84.0) < 1e-15);
    EXPECT_TRUE(std::fabs((val1 - 40.0) / 40.0) < 1e-15);
    EXPECT_TRUE(std::fabs((val2 - 20.0) / 20.0) < 1e-15);
#endif
  }
  {
    Kokkos::Experimental::ReductionView
      < double*[3]
      , Layout
      , ExecSpace
      , Kokkos::Experimental::ReductionSum
      , duplication
      , contribution
      >
      persistent_view("persistent", n);
    auto result_view = persistent_view.subview();
    contribute(result_view, persistent_view);
  }
}

template <typename ExecSpace>
struct TestDuplicatedReductionView {
  TestDuplicatedReductionView(int n) {
    test_reduction_view_config<ExecSpace, Kokkos::LayoutRight,
      Kokkos::Experimental::ReductionDuplicated,
      Kokkos::Experimental::ReductionNonAtomic>(n);
    test_reduction_view_config<ExecSpace, Kokkos::LayoutRight,
      Kokkos::Experimental::ReductionDuplicated,
      Kokkos::Experimental::ReductionAtomic>(n);
  }
};

#ifdef KOKKOS_ENABLE_CUDA
// disable duplicated instantiation with CUDA until
// UniqueToken can support it
template <>
struct TestDuplicatedReductionView<Kokkos::Cuda> {
  TestDuplicatedReductionView(int) {
  }
};
#endif

template <typename ExecSpace>
void test_reduction_view(int n)
{
  // all of these configurations should compile okay, but only some of them are
  // correct and/or sensible in terms of memory use
  Kokkos::Experimental::UniqueToken<ExecSpace> unique_token{ExecSpace()};

  // no atomics or duplication is only sensible if the execution space
  // is running essentially in serial (doesn't have to be Serial though,
  // we also test OpenMP with one thread: LAMMPS cares about that)
  if (unique_token.size() == 1) {
    test_reduction_view_config<ExecSpace, Kokkos::LayoutRight,
      Kokkos::Experimental::ReductionNonDuplicated,
      Kokkos::Experimental::ReductionNonAtomic>(n);
  }
  test_reduction_view_config<ExecSpace, Kokkos::LayoutRight,
    Kokkos::Experimental::ReductionNonDuplicated,
    Kokkos::Experimental::ReductionAtomic>(n);

  TestDuplicatedReductionView<ExecSpace> duptest(n);
}

} // namespace Test

#endif //KOKKOS_TEST_UNORDERED_MAP_HPP


