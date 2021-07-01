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
#include <Kokkos_StdAlgorithms.hpp>

namespace Test {

class std_algorithms : public ::testing::Test {
 protected:
  void verify_values(int expected) {
    for (int i = 0; i < m_view.extent(0); i++) {
      EXPECT_EQ(expected, m_view[i]);
    }
  }

  template <class FunctorType>
  void test_with(FunctorType functor) {
    Kokkos::Experimental::for_each(m_view, functor);
    verify_values(1);

    Kokkos::Experimental::for_each(Kokkos::Experimental::begin(m_view),
                                   Kokkos::Experimental::end(m_view), functor);
    verify_values(2);

    Kokkos::Experimental::for_each(m_view.data(), m_view.data() + m_view.size(),
                                   functor);
    verify_values(3);

    Kokkos::Experimental::for_each_n(m_view.data(), 10, functor);
    verify_values(4);

    Kokkos::Experimental::for_each_n(m_view, 10, functor);
    verify_values(5);
  }

  Kokkos::View<int[10]> m_view{"1-D-contiguous-view"};
};

struct Functor {
  KOKKOS_INLINE_FUNCTION
  void operator()(int& i) const { i++; }
};

TEST_F(std_algorithms, for_each_functor) {
  const auto fun = Functor();
  test_with(fun);
}

TEST_F(std_algorithms, for_each_lambda) {
  const auto fun = KOKKOS_LAMBDA(int& i) { i++; };
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  test_with(fun);
#endif
}

}  // namespace Test
