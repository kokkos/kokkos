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
#include <TestStdAlgorithmsCommon.hpp>

namespace Test {

template <class IndexType>
struct MyFunctor {
  using index_type = IndexType;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, int& redValue) const {
    (void)i;
    redValue += 1;
  }

  KOKKOS_INLINE_FUNCTION
  MyFunctor() {}
};

template <class ViewType>
struct FillViewFunctor {
  ViewType m_view;

  FillViewFunctor(ViewType view) : m_view(view) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { m_view(i) = i; }
};

TEST(red_issue, is_ok) {
  using view_t = Kokkos::View<int[10]>;
  view_t myView("VIEW");

  FillViewFunctor<view_t> F(myView);
  Kokkos::parallel_for(myView.extent(0), F);

  using index_type   = int;
  using reducer_type = Kokkos::Sum<index_type, Kokkos::DefaultExecutionSpace>;
  int sum            = 0;
  reducer_type reducer(sum);
  Kokkos::parallel_reduce("LABEL", myView.extent(10), MyFunctor<index_type>(),
                          reducer);
  std::cout << "ISOK\n";
}

TEST(red_issue, not_ok) {
  using view_t = Kokkos::View<int[10]>;
  view_t myView("VIEW");

  FillViewFunctor<view_t> F(myView);
  Kokkos::parallel_for(myView.extent(0), F);

  using index_type   = int;
  using reducer_type = Kokkos::Sum<index_type, Kokkos::DefaultExecutionSpace>;
  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("find_if_impl_result_view");
  reducer_type reducer(result);
  Kokkos::parallel_reduce("LABEL", myView.extent(0), MyFunctor<index_type>(),
                          reducer);
  std::cout << "NOT_OK\n";
}

}  // namespace Test
