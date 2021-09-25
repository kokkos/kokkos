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

//#include <TestStdAlgorithmsCommon.hpp>
#include <gtest/gtest.h>
#include <TestStdAlgorithmsHelperFunctors.hpp>
#include <std_algorithms/Kokkos_BeginEnd.hpp>
#include <std_algorithms/Kokkos_MinMaxOperations.hpp>
#include <std_algorithms/Kokkos_MinMaxElementOperations.hpp>

namespace KE = Kokkos::Experimental;

namespace Test {
namespace stdalgos {

template <class ViewType>
void fill_view(ViewType dest_view)
{
  using value_type      = typename ViewType::value_type;
  using exe_space       = typename ViewType::execution_space;
  using aux_view_t      = Kokkos::View<value_type*, exe_space>;

  const std::size_t ext = dest_view.extent(0);
  aux_view_t aux_view("aux_view", ext);
  auto v_h = create_mirror_view(Kokkos::HostSpace(), aux_view);

  for (std::size_t i = 0; i < ext; ++i) {
    v_h(i) = (value_type) i;
  }
  v_h(ext/2) = (value_type) -101;

  Kokkos::deep_copy(aux_view, v_h);
  CopyFunctor<aux_view_t, ViewType> F1(aux_view, dest_view);
  Kokkos::parallel_for("copy", dest_view.extent(0), F1);
}

template <class ViewType, class IndexType, class ReducerType>
struct MyFunctor
{
  using red_value_type = typename ReducerType::value_type;

  ViewType m_view;
  ReducerType m_reducer;

  KOKKOS_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const
  {
    m_reducer.join(red_value, red_value_type{m_view(i), i});
  }

  KOKKOS_FUNCTION
  MyFunctor(ViewType view, ReducerType reducer)
      : m_view(view), m_reducer(std::move(reducer)) {}
};

TEST(scalar_vs_view_red, use_scalar)
{
  using exe_space = Kokkos::DefaultExecutionSpace;
  using index_type = int;
  using scalar_type = int;
  using view_type = Kokkos::View<scalar_type*, exe_space>;

  const auto ext = 10001;
  view_type view("myview", ext);
  fill_view(view);

  using reducer_type = ::Kokkos::MinLoc<scalar_type, index_type>;
  using red_result_type = typename reducer_type::value_type;
  using func_type = MyFunctor<view_type, index_type, reducer_type>;
  red_result_type result;
  reducer_type reducer(result);
  Kokkos::parallel_reduce( "MinLocReduce",
			   Kokkos::RangePolicy<exe_space>(exe_space(), 0, ext),
			   func_type(view, reducer),
			   reducer);
  std::cout << " use_scalar = " << result.val << '\n';
}

}  // namespace stdalgos
}  // namespace Test
