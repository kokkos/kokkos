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

#ifndef KOKKOS_STD_NUMERICS_ADJACENT_DIFFERENCE_HPP
#define KOKKOS_STD_NUMERICS_ADJACENT_DIFFERENCE_HPP

#include <Kokkos_Core.hpp>
#include "../Kokkos_Constraints.hpp"
#include "../Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {
namespace Impl {

// ------------------------
//
// functors
//
// ------------------------
template <class ValueType1, class ValueType2, class RetType = ValueType2>
struct StdAdjacentDifferenceDefaultBinaryOpFunctor {
  KOKKOS_FUNCTION
  constexpr RetType operator()(const ValueType1& a, const ValueType2& b) const {
    return a - b;
  }
};

template <class IteratorType, class DestViewType, class BinaryOperator>
struct StdAdjacentDiffItToViewFunctor {
  using index_type = typename IteratorType::difference_type;

  const IteratorType m_first;
  const DestViewType m_dest_view;
  BinaryOperator m_op;

  KOKKOS_FUNCTION
  void operator()(const index_type i) const {
    const auto& my_value = m_first[i];
    if (i == 0) {
      m_dest_view(i) = my_value;
    } else {
      const auto& left_value = m_first[i - 1];
      m_dest_view(i)         = m_op(my_value, left_value);
    }
  }

  KOKKOS_FUNCTION
  StdAdjacentDiffItToViewFunctor(IteratorType first, DestViewType dest_view,
                                 BinaryOperator op)
      : m_first(std::move(first)),
        m_dest_view(dest_view),
        m_op(std::move(op)) {}
};

template <class ViewTypeFrom, class OutputIteratorType>
struct StdAdjDiffCopyFunctor {
  ViewTypeFrom m_view_from;
  OutputIteratorType m_first_dest;

  KOKKOS_FUNCTION
  StdAdjDiffCopyFunctor(const ViewTypeFrom view_from,
                        OutputIteratorType first_dest)
      : m_view_from(view_from), m_first_dest(std::move(first_dest)) {}

  KOKKOS_FUNCTION
  void operator()(int i) const { m_first_dest[i] = m_view_from(i); }
};

// ------------------------------------------
// adjacent_difference_impl
// ------------------------------------------
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
OutputIteratorType adjacent_difference_impl(const std::string& label,
                                            const ExecutionSpace& ex,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOp bin_op) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  if (first_from == last_from) {
    return first_dest;
  }

  // potentially, we could handle better the case where
  // source interval and output interval are not the same.
  // in that case, we don't need the auxiliary view.

  // aliases
  using value_type    = typename OutputIteratorType::value_type;
  using aux_view_type = ::Kokkos::View<value_type*, ExecutionSpace>;
  using functor1_t    = StdAdjacentDiffItToViewFunctor<InputIteratorType,
                                                    aux_view_type, BinaryOp>;
  using functor2_t = StdAdjDiffCopyFunctor<aux_view_type, OutputIteratorType>;

  // run
  const auto num_elements = last_from - first_from;
  aux_view_type aux_view("aux_view", num_elements);
  ::Kokkos::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         functor1_t(first_from, aux_view, bin_op));
  ::Kokkos::parallel_for("copy",
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         functor2_t(aux_view, first_dest));
  ex.fence("Kokkos::adjacent_difference: fence after operation");

  // return
  return first_dest + num_elements;
}

}  // end namespace Impl

// ------------------------
//
// public API
//
// ------------------------
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType>
std::enable_if_t<!::Kokkos::is_view<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const ExecutionSpace& ex, InputIteratorType first_from,
                    InputIteratorType last_from,
                    OutputIteratorType first_dest) {
  using value_type1 = typename InputIteratorType::value_type;
  using value_type2 = typename OutputIteratorType::value_type;
  using binary_op =
      Impl::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return Impl::adjacent_difference_impl(
      "Kokkos::adjacent_difference_iterator_api", ex, first_from, last_from,
      first_dest, binary_op());
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
std::enable_if_t<!::Kokkos::is_view<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const ExecutionSpace& ex, InputIteratorType first_from,
                    InputIteratorType last_from, OutputIteratorType first_dest,
                    BinaryOp bin_op) {
  return Impl::adjacent_difference_impl(
      "Kokkos::adjacent_difference_iterator_api", ex, first_from, last_from,
      first_dest, bin_op);
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType>
std::enable_if_t<!::Kokkos::is_view<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType first_from, InputIteratorType last_from,
                    OutputIteratorType first_dest) {
  using value_type1 = typename InputIteratorType::value_type;
  using value_type2 = typename OutputIteratorType::value_type;
  using binary_op =
      Impl::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return Impl::adjacent_difference_impl(label, ex, first_from, last_from,
                                        first_dest, binary_op());
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class BinaryOp>
std::enable_if_t<!::Kokkos::is_view<InputIteratorType>::value,
                 OutputIteratorType>
adjacent_difference(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType first_from, InputIteratorType last_from,
                    OutputIteratorType first_dest, BinaryOp bin_op) {
  return Impl::adjacent_difference_impl(label, ex, first_from, last_from,
                                        first_dest, bin_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto adjacent_difference(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);

  using view_type1  = ::Kokkos::View<DataType1, Properties1...>;
  using view_type2  = ::Kokkos::View<DataType2, Properties2...>;
  using value_type1 = typename view_type1::value_type;
  using value_type2 = typename view_type2::value_type;
  using binary_op =
      Impl::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;
  return Impl::adjacent_difference_impl(
      "Kokkos::adjacent_difference_view_api", ex, KE::cbegin(view_from),
      KE::cend(view_from), KE::begin(view_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
    BinaryOp bin_op) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  return Impl::adjacent_difference_impl(
      "Kokkos::adjacent_difference_view_api", ex, KE::cbegin(view_from),
      KE::cend(view_from), KE::begin(view_dest), bin_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);

  using view_type1  = ::Kokkos::View<DataType1, Properties1...>;
  using view_type2  = ::Kokkos::View<DataType2, Properties2...>;
  using value_type1 = typename view_type1::value_type;
  using value_type2 = typename view_type2::value_type;
  using binary_op =
      Impl::StdAdjacentDifferenceDefaultBinaryOpFunctor<value_type1,
                                                        value_type2>;

  return Impl::adjacent_difference_impl(label, ex, KE::cbegin(view_from),
                                        KE::cend(view_from),
                                        KE::begin(view_dest), binary_op());
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryOp>
auto adjacent_difference(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
    BinaryOp bin_op) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  return Impl::adjacent_difference_impl(label, ex, KE::cbegin(view_from),
                                        KE::cend(view_from),
                                        KE::begin(view_dest), bin_op);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
