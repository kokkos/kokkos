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
#include <std_algorithms/Kokkos_ModifyingSequenceOperations.hpp>

namespace KE = Kokkos::Experimental;

namespace Test {

template <class ViewType>
struct AssignIndexFunctor {
  ViewType m_view;

  AssignIndexFunctor(ViewType view) : m_view(view) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { m_view(i) = i; }
};

struct std_algorithms_mod_seq_ops : public ::testing::Test {
 public:
  using dyn_view_t    = Kokkos::View<int*>;
  using static_view_t = Kokkos::View<int[10]>;

  virtual void SetUp() {
    Kokkos::parallel_for(m_static_view.extent(0),
                         AssignIndexFunctor<static_view_t>(m_static_view));
  }
  virtual void TearDown() {}

  static_view_t m_static_view{"std-algo-test-1D-contiguous-view-static"};
  dyn_view_t m_dynamic_view{"std-algo-test-1D-contiguous-view-dynamic", 10};
};
// ------------------------------------------------------------

TEST_F(std_algorithms_mod_seq_ops, copy) {
  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::begin(m_dynamic_view);
  EXPECT_EQ(KE::end(m_dynamic_view), KE::copy(first, last, dest));

  for (std::size_t i = 0; i < m_static_view.extent(0); i++) {
    EXPECT_EQ(i, m_static_view(i));
    EXPECT_EQ(i, m_dynamic_view(i));
  }
}

TEST_F(std_algorithms_mod_seq_ops, copy_view) {
  EXPECT_EQ(KE::end(m_dynamic_view), KE::copy(m_static_view, m_dynamic_view));

  for (std::size_t i = 0; i < m_static_view.extent(0); i++) {
    EXPECT_EQ(i, m_static_view(i));
    EXPECT_EQ(i, m_dynamic_view(i));
  }
}

TEST_F(std_algorithms_mod_seq_ops, copy_n) {
  constexpr std::size_t n = 5;
  auto first              = KE::begin(m_static_view);
  auto dest               = KE::begin(m_dynamic_view);

  EXPECT_EQ(dest + n, KE::copy_n(first, n, dest));

  for (std::size_t i = 0; i < m_static_view.extent(0); i++) {
    EXPECT_EQ(i, m_static_view(i));
    if (i < n)
      EXPECT_EQ(i, m_dynamic_view(i));
    else
      EXPECT_EQ(0, m_dynamic_view(i));
  }
}

TEST_F(std_algorithms_mod_seq_ops, copy_backward) {
  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::end(m_dynamic_view);

  EXPECT_EQ(KE::begin(m_dynamic_view), KE::copy_backward(first, last, dest));

  for (std::size_t i = 0; i < m_static_view.extent(0); i++) {
    EXPECT_EQ(i, m_static_view(i));
    EXPECT_EQ(i, m_dynamic_view(i));
  }
}

TEST_F(std_algorithms_mod_seq_ops, copy_if_lambda) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  const auto is_odd = KOKKOS_LAMBDA(const int i) { return (i % 2); };

  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::begin(m_dynamic_view);

  // should only copy five values (1, 3, 5, 7, 9)
  EXPECT_EQ(dest + 5, KE::copy_if(first, last, dest, is_odd));

  EXPECT_EQ(1, m_dynamic_view(0));
  EXPECT_EQ(3, m_dynamic_view(1));
  EXPECT_EQ(5, m_dynamic_view(2));
  EXPECT_EQ(7, m_dynamic_view(3));
  EXPECT_EQ(9, m_dynamic_view(4));
  EXPECT_EQ(0, m_dynamic_view(5));
#endif
}

TEST_F(std_algorithms_mod_seq_ops, reverse_copy) {
  auto first = KE::begin(m_static_view);
  auto last  = KE::end(m_static_view);
  auto dest  = KE::begin(m_dynamic_view);

  EXPECT_EQ(KE::end(m_dynamic_view), KE::reverse_copy(first, last, dest));

  const auto size = m_static_view.extent(0);
  for (std::size_t i = 0; i < size; i++) {
    EXPECT_EQ(i, m_static_view(i));
    EXPECT_EQ(size - 1 - i, m_dynamic_view(i));
  }
}

}  // namespace Test
