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

#ifndef KOKKOS_STD_NUMERICS_TRANSFORM_REDUCE_HPP
#define KOKKOS_STD_NUMERICS_TRANSFORM_REDUCE_HPP

#include <Kokkos_Core.hpp>
#include "../Kokkos_Constraints.hpp"
#include "../Kokkos_ModifyingOperations.hpp"
#include "../Kokkos_BeginEnd.hpp"
#include "../Kokkos_ReducerWithArbitraryJoinerNoNeutralElement.hpp"

namespace Kokkos {
namespace Experimental {
namespace Impl {

//
// helper functors
//
template <class ValueType>
struct StdTranformReduceDefaultBinaryTransformFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return (a * b);
  }
};

template <class ValueType>
struct StdTranformReduceDefaultJoinFunctor {
  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const ValueType& a, const ValueType& b) const {
    return a + b;
  }

  KOKKOS_INLINE_FUNCTION
  ValueType operator()(const volatile ValueType& a,
                       const volatile ValueType& b) const {
    return a + b;
  }
};

template <class IteratorType, class ReducerType, class TransformType>
struct StdTransformReduceSingleIntervalFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType::difference_type;

  const IteratorType m_first;
  const ReducerType m_reducer;
  const TransformType m_transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto my_iterator = m_first + i;
    auto tmp_wrapped_value = red_value_type{m_transform(*my_iterator), false};
    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdTransformReduceSingleIntervalFunctor(IteratorType first,
                                          ReducerType reducer,
                                          TransformType transform)
      : m_first(first),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_transform(::Kokkos::Experimental::move(transform)) {}
};

template <class IteratorType1, class IteratorType2, class ReducerType,
          class TransformType>
struct StdTransformReduceTwoIntervalsFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename IteratorType1::difference_type;

  const IteratorType1 m_first1;
  const IteratorType2 m_first2;
  const ReducerType m_reducer;
  const TransformType m_transform;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    const auto my_iterator1 = m_first1 + i;
    const auto my_iterator2 = m_first2 + i;
    auto tmp_wrapped_value =
        red_value_type{m_transform(*my_iterator1, *my_iterator2), false};

    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdTransformReduceTwoIntervalsFunctor(IteratorType1 first1,
                                        IteratorType2 first2,
                                        ReducerType reducer,
                                        TransformType transform)
      : m_first1(first1),
        m_first2(first2),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_transform(::Kokkos::Experimental::move(transform)) {}
};

template <class ExecutionSpace, class ValueType, class IteratorType1,
          class IteratorType2>
struct admissible_to_transform_reduce {
  static_assert(not_openmptarget<ExecutionSpace>::value,
                "transform_reduce not currently supported in OpenMPTarget");

  static_assert(
      are_random_access_iterators<IteratorType1, IteratorType2>::value,
      "Currently, Kokkos standard algorithms require random access iterators.");

  static_assert(are_accessible_iterators<ExecutionSpace, IteratorType1,
                                         IteratorType2>::value,
                "Incompatible views/iterators and execution space");

  // for now, we need to check iterators have same value type because reducers
  // need that: reducer concept infact has a single nested value_type typedef
  using iterator1_value_type = typename IteratorType1::value_type;
  using iterator2_value_type = typename IteratorType2::value_type;
  static_assert(
      std::is_same<iterator1_value_type, iterator2_value_type>::value,
      "transform_reduce currently only supports operands with same value_type");

  static_assert(std::is_same<std::remove_cv_t<iterator1_value_type>,
                             std::remove_cv_t<ValueType> >::value,
                "transform_reduce: iterator1/view1 value_type must be the same "
                "as type of init argument");
  static_assert(std::is_same<std::remove_cv_t<iterator2_value_type>,
                             std::remove_cv_t<ValueType> >::value,
                "transform_reduce: iterator2/view2 value_type must be the same "
                "as type of init argument");

  static constexpr bool value = true;
};

template <class ExecutionSpace, class IteratorType, class ValueType,
          class JoinerType, class UnaryTransformerType>
ValueType transform_reduce_custom_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, ValueType init_reduction_value, JoinerType joiner,
    UnaryTransformerType transformer) {
  static_assert(
      admissible_to_transform_reduce<ExecutionSpace, ValueType, IteratorType,
                                     IteratorType>::value,
      "");
  expect_valid_range(first, last);

  if (first == last) {
    // init is returned, unmodified
    return init_reduction_value;
  }

  using iterator_value_type = typename IteratorType::value_type;
  using reducer_type =
      ReducerWithArbitraryJoinerNoNeutralElement<iterator_value_type,
                                                 JoinerType, ExecutionSpace>;
  using functor_type =
      StdTransformReduceSingleIntervalFunctor<IteratorType, reducer_type,
                                              UnaryTransformerType>;

  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("transform_reduce_custom_functors_impl_result");
  reducer_type reducer(result, joiner);

  const auto num_elements = last - first;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            functor_type(first, reducer, transformer), reducer);

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  // as per standard, transform is not applied to the init value
  // https://en.cppreference.com/w/cpp/algorithm/transform_reduce
  return joiner(r_h().val, init_reduction_value);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class JoinerType, class BinaryTransformerType>
ValueType transform_reduce_custom_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, ValueType init_reduction_value,
    JoinerType joiner, BinaryTransformerType transformer) {
  static_assert(
      admissible_to_transform_reduce<ExecutionSpace, ValueType, IteratorType1,
                                     IteratorType2>::value,
      "");
  expect_valid_range(first1, last1);
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();

  if (first1 == last1) {
    // init is returned, unmodified
    return init_reduction_value;
  }

  using iterator_value_type = typename IteratorType1::value_type;
  using reducer_type =
      ReducerWithArbitraryJoinerNoNeutralElement<iterator_value_type,
                                                 JoinerType, ExecutionSpace>;

  using functor_type = StdTransformReduceTwoIntervalsFunctor<
      IteratorType1, IteratorType2, reducer_type, BinaryTransformerType>;

  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("transform_reduce_custom_functors_impl_result");
  reducer_type reducer(result, joiner);

  const auto num_elements = last1 - first1;
  ::Kokkos::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      functor_type(first1, first2, reducer, transformer), reducer);

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);
  return joiner(r_h().val, init_reduction_value);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce_default_functors_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, ValueType init_reduction_value) {
  static_assert(
      admissible_to_transform_reduce<ExecutionSpace, ValueType, IteratorType1,
                                     IteratorType2>::value,
      "");
  expect_valid_range(first1, last1);
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();

  using value_type = Kokkos::Impl::remove_cvref_t<ValueType>;
  using transformer_type =
      Impl::StdTranformReduceDefaultBinaryTransformFunctor<value_type>;
  using joiner_type = Impl::StdTranformReduceDefaultJoinFunctor<value_type>;

  return transform_reduce_custom_functors_impl(
      label, ex, first1, last1, first2,
      ::Kokkos::Experimental::move(init_reduction_value), joiner_type(),
      transformer_type());
}

}  // end namespace Impl

///////////////////////////////
//
// transform_reduce public API
//
///////////////////////////////

// ----------------------------
// overload1:
// no custom functors passed, so equivalent to
// transform_reduce(first1, last1, first2, init, plus<>(), multiplies<>());
// ----------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce(const ExecutionSpace& ex, IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value) {
  return Impl::transform_reduce_default_functors_impl(
      "kokkos_transform_reduce_default_functors_iterator_api", ex, first1,
      last1, first2, ::Kokkos::Experimental::move(init_reduction_value));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value) {
  return Impl::transform_reduce_default_functors_impl(
      label, ex, first1, last1, first2,
      ::Kokkos::Experimental::move(init_reduction_value));
}

// overload1 accepting views
template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType>
ValueType transform_reduce(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& first_view,
    const ::Kokkos::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value) {
  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(first_view);
  static_assert_is_admissible_to_kokkos_std_algorithms(second_view);

  return Impl::transform_reduce_default_functors_impl(
      "kokkos_transform_reduce_default_functors_iterator_api", ex,
      KE::cbegin(first_view), KE::cend(first_view), KE::cbegin(second_view),
      ::Kokkos::Experimental::move(init_reduction_value));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType>
ValueType transform_reduce(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& first_view,
    const ::Kokkos::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value) {
  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(first_view);
  static_assert_is_admissible_to_kokkos_std_algorithms(second_view);

  return Impl::transform_reduce_default_functors_impl(
      label, ex, KE::cbegin(first_view), KE::cend(first_view),
      KE::cbegin(second_view),
      ::Kokkos::Experimental::move(init_reduction_value));
}

// ---------------------------------------------
// overload2:
//
// accepts a custom transform and joiner functor
// ---------------------------------------------

// Note the std refers to the arg BinaryReductionOp
// but in the Kokkos naming convention, it corresponds
// to a "joiner" that knows how to join two values
// NOTE: "joiner/transformer" need to be commutative.

// https://en.cppreference.com/w/cpp/algorithm/transform_reduce

// api accepting iterators
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& ex, IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::transform_reduce_custom_functors_impl(
      "kokkos_transform_reduce_custom_functors_iterator_api", ex, first1, last1,
      first2, ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ValueType, class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2, ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::transform_reduce_custom_functors_impl(
      label, ex, first1, last1, first2,
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

// accepting views
template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& first_view,
    const ::Kokkos::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value, BinaryJoinerType joiner,
    BinaryTransform transformer) {
  namespace KE = ::Kokkos::Experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  static_assert_is_admissible_to_kokkos_std_algorithms(first_view);
  static_assert_is_admissible_to_kokkos_std_algorithms(second_view);

  return Impl::transform_reduce_custom_functors_impl(
      "kokkos_transform_reduce_custom_functors_view_api", ex,
      KE::cbegin(first_view), KE::cend(first_view), KE::cbegin(second_view),
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& first_view,
    const ::Kokkos::View<DataType2, Properties2...>& second_view,
    ValueType init_reduction_value, BinaryJoinerType joiner,
    BinaryTransform transformer) {
  namespace KE = ::Kokkos::Experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  static_assert_is_admissible_to_kokkos_std_algorithms(first_view);
  static_assert_is_admissible_to_kokkos_std_algorithms(second_view);

  return Impl::transform_reduce_custom_functors_impl(
      label, ex, KE::cbegin(first_view), KE::cend(first_view),
      KE::cbegin(second_view),
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

// ----------------------------
//
// overload3:
//
// ----------------------------

// accepting iterators
template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryJoinerType, class UnaryTransform>
// need this to avoid ambiguous call
std::enable_if_t< ::Kokkos::Experimental::are_iterators<IteratorType>::value,
                  ValueType>
transform_reduce(const ExecutionSpace& ex, IteratorType first1,
                 IteratorType last1, ValueType init_reduction_value,
                 BinaryJoinerType joiner, UnaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::transform_reduce_custom_functors_impl(
      "kokkos_transform_reduce_custom_functors_iterator_api", ex, first1, last1,
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryJoinerType, class UnaryTransform>
// need this to avoid ambiguous call
std::enable_if_t< ::Kokkos::Experimental::are_iterators<IteratorType>::value,
                  ValueType>
transform_reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first1, IteratorType last1,
                 ValueType init_reduction_value, BinaryJoinerType joiner,
                 UnaryTransform transformer) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::transform_reduce_custom_functors_impl(
      label, ex, first1, last1,
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

// accepting views
template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& ex,
                           const ::Kokkos::View<DataType, Properties...>& view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform transformer) {
  namespace KE = ::Kokkos::Experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::transform_reduce_custom_functors_impl(
      "kokkos_transform_reduce_custom_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label, const ExecutionSpace& ex,
                           const ::Kokkos::View<DataType, Properties...>& view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform transformer) {
  namespace KE = ::Kokkos::Experimental;
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::transform_reduce_custom_functors_impl(
      label, ex, KE::cbegin(view), KE::cend(view),
      ::Kokkos::Experimental::move(init_reduction_value),
      ::Kokkos::Experimental::move(joiner),
      ::Kokkos::Experimental::move(transformer));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
