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

#ifndef KOKKOS_STD_MIN_MAX_OPERATIONS_HPP
#define KOKKOS_STD_MIN_MAX_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_StdAlgorithmsConstraints.hpp"

/// \file Kokkos_MinMaxOperations.hpp
/// \brief Kokkos min/max operations

namespace Kokkos {
namespace Experimental {

// see https://github.com/kokkos/kokkos/issues/4075
//
// max_element
// min_element
// minmax_element

/*********************
  max_element
*********************/
template <class IteratorType, class RedValueType>
struct _StdAlgoMaxElemFunctor {
  IteratorType m_first;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, RedValueType & value) const
  {
    auto myValue = m_first + i;
    if (*myValue > value.val) {
      value.val = *myValue;
      value.loc = i;
    }
  }

  _StdAlgoMaxElemFunctor(IteratorType _firstIn) : m_first(_firstIn){}
};

namespace Impl{
template <class IteratorType>
IteratorType max_element_impl(const std::string & labelIn,
			      IteratorType first,
			      IteratorType end)
{
  if (first == end){
    return end;
  }

  const auto numOfElements = end-first;
  using it_scalar_type = typename IteratorType::value_type;
  using reducer_type = Kokkos::MaxLoc<it_scalar_type,int>;
  using reducer_value_type = typename reducer_type::value_type;

  reducer_value_type redValue;
  Kokkos::parallel_reduce(labelIn,
			  numOfElements,
			  _StdAlgoMaxElemFunctor<IteratorType, reducer_value_type>(first),
			  reducer_type(redValue));

  return first + redValue.loc;
}
}//end namespace Impl


template <class IteratorType>
IteratorType max_element(IteratorType first, IteratorType end)
{
  return Impl::max_element_impl("_std_max_element_1", first, end);
}

template <class IteratorType>
IteratorType max_element(const std::string & labelIn,
			 IteratorType first, IteratorType end)
{
  return Impl::max_element_impl(labelIn, first, end);
}

template <class DataType, class... Properties>
auto max_element(const Kokkos::View<DataType, Properties...>& v)
{
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_min_max_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::max_element only accepts 1D Views.");

  return Impl::max_element_impl("_std_max_element_2", cbegin(v), cend(v));
}

template <class DataType, class... Properties>
auto max_element(const std::string & labelIn,
		 const Kokkos::View<DataType, Properties...>& v)
{
  using ViewInType = Kokkos::View<DataType, Properties...>;
  static_assert(
      is_admissible_to_kokkos_std_min_max_op<ViewInType>::value,
      "Currently, Kokkos::Experimental::max_element only accepts 1D Views.");

  return Impl::max_element_impl(labelIn, cbegin(v), cend(v));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
