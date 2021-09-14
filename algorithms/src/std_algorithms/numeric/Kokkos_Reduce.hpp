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

#ifndef KOKKOS_STD_NUMERICS_REDUCE_HPP
#define KOKKOS_STD_NUMERICS_REDUCE_HPP

#include <Kokkos_Core.hpp>
#include "../Kokkos_BeginEnd.hpp"
#include "../Kokkos_Constraints.hpp"
#include "../Kokkos_ModifyingOperations.hpp"
#include "../Kokkos_ReducerWithArbitraryJoinerNoNeutralElement.hpp"

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ValueType>
struct StdReduceDefaultJoinFunctor {
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

template <class IndexType, class IteratorType, class ReducerType>
struct StdReduceFunctor {
  using RedValueType = typename ReducerType::value_type;
  const IteratorType m_first;
  const ReducerType m_reducer;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, RedValueType& red_value) const {
    const auto my_iterator = m_first + i;
    auto tmp_wrapped_value = RedValueType{*my_iterator, false};
    if (red_value.is_initial) {
      red_value = tmp_wrapped_value;
    } else {
      m_reducer.join(red_value, tmp_wrapped_value);
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdReduceFunctor(IteratorType first, ReducerType reducer)
      : m_first(first), m_reducer(::Kokkos::Experimental::move(reducer)) {}
};

template <class ExecutionSpace, class ValueType, class IteratorType>
struct admissible_to_reduce {
  static_assert(::Kokkos::Experimental::not_openmptarget<ExecutionSpace>::value,
                "transform_reduce not currently supported in OpenMPTarget");

  static_assert(
      are_random_access_iterators<IteratorType>::value,
      "Currently, Kokkos standard algorithms require random access iterators.");

  static_assert(::Kokkos::Experimental::are_accessible_iterators<
                    ExecutionSpace, IteratorType>::value,
                "Incompatible views/iterators and execution space");

  // for now, we need to check iterators have same value type because reducers
  // need that: reducer concept infact has a single nested value_type typedef
  using iterator_value_type = typename IteratorType::value_type;
  static_assert(std::is_same<std::remove_cv_t<iterator_value_type>,
                             std::remove_cv_t<ValueType> >::value,
                "reduce: iterator/view value_type must be the same "
                "as type of init argument");

  static constexpr bool value = true;
};

template <class ExecutionSpace, class IteratorType, class ValueType,
          class JoinerType>
ValueType reduce_custom_functors_impl(const std::string& label,
                                      const ExecutionSpace& ex,
                                      IteratorType first, IteratorType last,
                                      ValueType init_reduction_value,
                                      JoinerType joiner) {
  static_assert(
      admissible_to_reduce<ExecutionSpace, ValueType, IteratorType>::value,
      "types not admissible to reduce");

  if (first == last) {
    // init is returned, unmodified
    return init_reduction_value;
  }

  using iterator_value_type = typename IteratorType::value_type;
  using index_type          = std::size_t;
  using reducer_type =
      ReducerWithArbitraryJoinerNoNeutralElement<iterator_value_type,
                                                 JoinerType, ExecutionSpace>;
  using functor_type = StdReduceFunctor<index_type, IteratorType, reducer_type>;

  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("reduce_custom_functors_impl_result");
  reducer_type reducer(result, joiner);

  const auto num_elements = last - first;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            functor_type(first, reducer), reducer);

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);
  return joiner(r_h().val, init_reduction_value);
}

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce_default_functors_impl(const std::string& label,
                                       const ExecutionSpace& ex,
                                       IteratorType first, IteratorType last,
                                       ValueType init_reduction_value) {
  static_assert(
      admissible_to_reduce<ExecutionSpace, ValueType, IteratorType>::value,
      "types not admissible to reduce");

  using value_type  = Kokkos::Impl::remove_cvref_t<ValueType>;
  using joiner_type = Impl::StdReduceDefaultJoinFunctor<value_type>;

  return reduce_custom_functors_impl(
      label, ex, first, last,
      ::Kokkos::Experimental::move(init_reduction_value), joiner_type());
}

}  // end namespace Impl

///////////////////////////////
//
// reduce public API
//
///////////////////////////////

// overload1:
// reduce(first, last, typename std::iterator_traits<InputIt>::value_type{})
template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const ExecutionSpace& ex,
                                         IteratorType first,
                                         IteratorType last) {
  return Impl::reduce_default_functors_impl(
      "kokkos_reduce_default_functors_iterator_api", ex, first, last,
      typename IteratorType::value_type());
}

template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const std::string& label,
                                         const ExecutionSpace& ex,
                                         IteratorType first,
                                         IteratorType last) {
  return Impl::reduce_default_functors_impl(
      label, ex, first, last, typename IteratorType::value_type());
}

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view) {
  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  using view_type  = ::Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::value_type;

  return Impl::reduce_default_functors_impl(
      "kokkos_reduce_default_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), value_type());
}

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view) {
  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  using view_type  = ::Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::value_type;

  return Impl::reduce_default_functors_impl(label, ex, KE::cbegin(view),
                                            KE::cend(view), value_type());
}

//
// overload2:
//
// reduce(first, last, T init_value)
template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_default_functors_impl(
      "kokkos_reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value);
}

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_default_functors_impl(label, ex, first, last,
                                            init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
ValueType reduce(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_default_functors_impl(
      "kokkos_reduce_default_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_default_functors_impl(
      label, ex, KE::cbegin(view), KE::cend(view), init_reduction_value);
}

// overload3
// template< class IteratorType, class T, class BinaryOp >
// T reduce( IteratorType first, IteratorType last, T init, BinaryOp binary_op
// );
template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
ValueType reduce(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ValueType init_reduction_value,
                 BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_custom_functors_impl(
      "kokkos_reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value, joiner);
}

template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_custom_functors_impl(label, ex, first, last,
                                           init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
ValueType reduce(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_custom_functors_impl(
      "kokkos_reduce_custom_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_custom_functors_impl(label, ex, KE::cbegin(view),
                                           KE::cend(view), init_reduction_value,
                                           joiner);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
