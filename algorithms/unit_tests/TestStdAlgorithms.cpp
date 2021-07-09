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

struct std_algorithms : public ::testing::Test {
 public:
  virtual void SetUp() {}
  virtual void TearDown() {}

  using static_view_t = Kokkos::View<int[10]>;
  static_view_t m_static_view{"std-algo-test-1D-contiguous-view-static"};

  using dyn_view_t = Kokkos::View<int*>;
  dyn_view_t m_dynamic_view{"std-algo-test-1D-contiguous-view-dynamic", 10};

  using strided_view_t = Kokkos::View<int*, Kokkos::LayoutStride>;
  Kokkos::LayoutStride layout{10, 2};
  strided_view_t m_strided_view{"std-algo-test-1D-strided-view", layout};
};
// ------------------------------------------------------------

struct IncrementElementWiseFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(int& i) const { ++i; }
};

struct NoOpNonMutableFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const { (void)i; }
};
// ------------------------------------------------------------

template <class ViewTypeFrom, class ViewTypeTo>
struct CopyFunctor {
  ViewTypeFrom m_viewFrom;
  ViewTypeTo m_viewTo;

  CopyFunctor() = delete;

  CopyFunctor(const ViewTypeFrom viewFromIn, const ViewTypeTo viewToIn)
      : m_viewFrom(viewFromIn), m_viewTo(viewToIn) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { m_viewTo(i) = m_viewFrom(i); }
};

template <class ValueType, class ViewType>
std::enable_if_t<!std::is_same<typename ViewType::traits::array_layout,
                               Kokkos::LayoutStride>::value>
verify_values(ValueType expected, const ViewType viewIn) {
  static_assert(std::is_same<ValueType, typename ViewType::value_type>::value,
                "Non-matching value types of view and reference value");
  auto view_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), viewIn);
  for (std::size_t i = 0; i < view_h.extent(0); i++) {
    EXPECT_EQ(expected, view_h(i));
  }
}

template <class ValueType, class ViewType>
std::enable_if_t<std::is_same<typename ViewType::traits::array_layout,
                              Kokkos::LayoutStride>::value>
verify_values(ValueType expected, const ViewType viewIn) {
  static_assert(std::is_same<ValueType, typename ViewType::value_type>::value,
                "Non-matching value types of view and reference value");

  using non_strided_view_t = Kokkos::View<typename ViewType::value_type*>;
  non_strided_view_t tmpView("tmpView", viewIn.extent(0));

  Kokkos::parallel_for(
      "_std_algo_copy", viewIn.extent(0),
      CopyFunctor<ViewType, non_strided_view_t>(viewIn, tmpView));
  auto view_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tmpView);
  for (std::size_t i = 0; i < view_h.extent(0); i++) {
    EXPECT_EQ(expected, view_h(i));
  }
}
// ------------------------------------------------------------

template <class FunctorType, class ViewType>
void test_for_each_v1(FunctorType functor, const ViewType viewIn) {
  Kokkos::Experimental::for_each(viewIn, functor);
  verify_values(1, viewIn);

  Kokkos::Experimental::for_each(Kokkos::Experimental::begin(viewIn),
                                 Kokkos::Experimental::end(viewIn), functor);
  verify_values(2, viewIn);
}

template <class FunctorType, class ViewType>
void test_for_each_v2(FunctorType functor, const ViewType viewIn) {
  Kokkos::Experimental::for_each(viewIn, functor);
  verify_values(0, viewIn);

  Kokkos::Experimental::for_each(Kokkos::Experimental::cbegin(viewIn),
                                 Kokkos::Experimental::cend(viewIn), functor);
  verify_values(0, viewIn);
}

template <class FunctorType, class ViewType>
void test_for_each_n_v1(FunctorType functor, const ViewType viewIn) {
  Kokkos::Experimental::for_each_n(Kokkos::Experimental::begin(viewIn), 10,
                                   functor);
  verify_values(3, viewIn);

  Kokkos::Experimental::for_each_n(viewIn, 10, functor);
  verify_values(4, viewIn);
}

template <class FunctorType, class ViewType>
void test_for_each_n_v2(FunctorType functor, const ViewType viewIn) {
  Kokkos::Experimental::for_each_n(Kokkos::Experimental::cbegin(viewIn), 10,
                                   functor);
  verify_values(0, viewIn);

  Kokkos::Experimental::for_each_n(viewIn, 10, functor);
  verify_values(0, viewIn);
}
// ------------------------------------------------------------
// ------------------------------------------------------------

// for_each, functor with non-const arg
TEST_F(std_algorithms, for_each_modifying_functor_static_view) {
  const auto fun = IncrementElementWiseFunctor();
  test_for_each_v1(fun, m_static_view);
}
TEST_F(std_algorithms, for_each_modifying_functor_dynamic_view) {
  const auto fun = IncrementElementWiseFunctor();
  test_for_each_v1(fun, m_dynamic_view);
}
TEST_F(std_algorithms, for_each_modifying_functor_strided_view) {
  const auto fun = IncrementElementWiseFunctor();
  test_for_each_v1(fun, m_strided_view);
}

// for_each, functor with const arg
TEST_F(std_algorithms, for_each_non_modifying_functor_static_view) {
  const auto fun = NoOpNonMutableFunctor();
  test_for_each_v2(fun, m_static_view);
}
TEST_F(std_algorithms, for_each_non_modifying_functor_dynamic_view) {
  const auto fun = NoOpNonMutableFunctor();
  test_for_each_v2(fun, m_dynamic_view);
}
TEST_F(std_algorithms, for_each_non_modifying_functor_strided_view) {
  const auto fun = NoOpNonMutableFunctor();
  test_for_each_v2(fun, m_strided_view);
}

#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
// for_each, lambda with non-const arg
TEST_F(std_algorithms, for_each_modifying_lambda_static_view) {
  const auto fun = KOKKOS_LAMBDA(int& i) { ++i; };
  test_for_each_v1(fun, m_static_view);
}
TEST_F(std_algorithms, for_each_modifying_lambda_dynamic_view) {
  const auto fun = KOKKOS_LAMBDA(int& i) { ++i; };
  test_for_each_v1(fun, m_dynamic_view);
}
TEST_F(std_algorithms, for_each_modifying_lambda_strided_view) {
  const auto fun = KOKKOS_LAMBDA(int& i) { ++i; };
  test_for_each_v1(fun, m_strided_view);
}

// for_each, lambda with const arg
TEST_F(std_algorithms, for_each_non_modifying_lambda_static_view) {
  const auto fun = KOKKOS_LAMBDA(const int& i) { (void)i; };
  test_for_each_v2(fun, m_static_view);
}
TEST_F(std_algorithms, for_each_non_modifying_lambda_dynamic_view) {
  const auto fun = KOKKOS_LAMBDA(const int& i) { (void)i; };
  test_for_each_v2(fun, m_dynamic_view);
}
TEST_F(std_algorithms, for_each_non_modifying_lambda_strided_view) {
  const auto fun = KOKKOS_LAMBDA(const int& i) { (void)i; };
  test_for_each_v2(fun, m_strided_view);
}
#endif

TEST_F(std_algorithms, count_if_lambda) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  namespace KE = Kokkos::Experimental;
  EXPECT_EQ(m_static_view.extent(0),
            KE::count(KE::begin(m_static_view), KE::end(m_static_view), 0));

  m_static_view(0) = 6;
  m_static_view(1) = 7;
  m_static_view(9) = 7;

  const auto is_negative = KOKKOS_LAMBDA(const int i) { return (i < 0); };
  EXPECT_EQ(0, KE::count_if(KE::begin(m_static_view), KE::end(m_static_view),
                            is_negative));
  EXPECT_EQ(0, KE::count_if(m_static_view, is_negative));

  const auto is_odd = KOKKOS_LAMBDA(const int i) { return (i % 2); };
  EXPECT_EQ(2, KE::count_if(KE::begin(m_static_view), KE::end(m_static_view),
                            is_odd));
  EXPECT_EQ(2, KE::count_if(m_static_view, is_odd));

  EXPECT_EQ(1, KE::count(KE::begin(m_static_view), KE::end(m_static_view), 6));
  EXPECT_EQ(1, KE::count(m_static_view, 6));
#endif
}

TEST_F(std_algorithms, mismatch) {
  namespace KE      = Kokkos::Experimental;
  m_static_view(5)  = 1;
  m_dynamic_view(5) = 2;

  auto p = KE::mismatch(KE::begin(m_static_view), KE::end(m_static_view),
                        KE::begin(m_dynamic_view), KE::end(m_dynamic_view));

  EXPECT_EQ(KE::begin(m_static_view) + 5, p.first);
  EXPECT_EQ(1, *p.first);
  EXPECT_EQ(KE::begin(m_dynamic_view) + 5, p.second);
  EXPECT_EQ(2, *p.second);
}

TEST_F(std_algorithms, find_if_lambda) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  namespace KE = Kokkos::Experimental;
  EXPECT_EQ(KE::end(m_static_view),
            KE::find_if(
                KE::begin(m_static_view), KE::end(m_static_view),
                KOKKOS_LAMBDA(int i) { return i != 0; }));

  m_static_view(5) = 1;
  EXPECT_EQ(KE::begin(m_static_view) + 5,
            KE::find_if(
                KE::begin(m_static_view), KE::end(m_static_view),
                KOKKOS_LAMBDA(int i) { return i != 0; }));
#endif
}

TEST_F(std_algorithms, any_of_lambda) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  namespace KE = Kokkos::Experimental;
  EXPECT_EQ(false, KE::any_of(
                       KE::begin(m_static_view), KE::end(m_static_view),
                       KOKKOS_LAMBDA(int i) { return i != 0; }));

  m_static_view(5) = 1;
  EXPECT_EQ(true, KE::any_of(
                      KE::begin(m_static_view), KE::end(m_static_view),
                      KOKKOS_LAMBDA(int i) { return i != 0; }));
#endif
}

TEST_F(std_algorithms, none_of_lambda) {
#if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  namespace KE = Kokkos::Experimental;
  EXPECT_EQ(true, KE::none_of(
                      KE::begin(m_static_view), KE::end(m_static_view),
                      KOKKOS_LAMBDA(int i) { return i != 0; }));

  m_static_view(5) = 1;
  EXPECT_EQ(false, KE::none_of(
                       KE::begin(m_static_view), KE::end(m_static_view),
                       KOKKOS_LAMBDA(int i) { return i != 0; }));
#endif
}

}  // namespace Test
