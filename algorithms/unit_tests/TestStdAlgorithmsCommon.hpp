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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TEST_STD_ALGOS_COMMON_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace Test {
namespace stdalgos {

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

template <class ItTypeFrom, class ViewTypeTo>
struct CopyFromIteratorFunctor {
  ItTypeFrom m_itFrom;
  ViewTypeTo m_viewTo;

  CopyFromIteratorFunctor(const ItTypeFrom itFromIn, const ViewTypeTo viewToIn)
      : m_itFrom(itFromIn), m_viewTo(viewToIn) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const { m_viewTo() = *m_itFrom; }
};

struct IncrementElementWiseFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(int& i) const { ++i; }
};

struct NoOpNonMutableFunctor {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const { (void)i; }
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
      stdalgos::CopyFunctor<ViewType, non_strided_view_t>(viewIn, tmpView));
  auto view_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tmpView);
  for (std::size_t i = 0; i < view_h.extent(0); i++) {
    EXPECT_EQ(expected, view_h(i));
  }
}

}  // namespace stdalgos
}  // namespace Test

#endif
