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

#ifndef KOKKOS_STD_PARTITIONING_OPERATIONS_HPP
#define KOKKOS_STD_PARTITIONING_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_Constraints.hpp"
#include "Kokkos_MinMaxOperations.hpp"
#include "Kokkos_ModifyingOperations.hpp"
#include "Kokkos_NonModifyingSequenceOperations.hpp"

namespace Kokkos {
namespace Experimental {

// ------------------------------------------
// begin Impl namespace
// ------------------------------------------
namespace Impl {

// impl functors
template <class IteratorType, class ReducerType, class PredicateType>
struct StdIsPartitionedFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename red_value_type::index_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& redValue) const {
    auto my_iterator           = m_first + i;
    const auto predicate_value = m_p(*my_iterator);

    auto rv =
        predicate_value
            ? red_value_type{i, ::Kokkos::reduction_identity<index_type>::min()}
            : red_value_type{::Kokkos::reduction_identity<index_type>::max(),
                             i};

    m_reducer.join(redValue, rv);
  }

  KOKKOS_INLINE_FUNCTION
  StdIsPartitionedFunctor(IteratorType first, ReducerType reducer,
                          PredicateType p)
      : m_first(first),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_p(::Kokkos::Experimental::move(p)) {}
};

template <class IteratorType, class ReducerType, class PredicateType>
struct StdPartitionPointFunctor {
  using red_value_type = typename ReducerType::value_type;
  using index_type     = typename red_value_type::index_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& redValue) const {
    auto my_iterator           = m_first + i;
    const auto predicate_value = m_p(*my_iterator);

    auto rv =
        predicate_value
            ? red_value_type{::Kokkos::reduction_identity<index_type>::min()}
            : red_value_type{i};
    m_reducer.join(redValue, rv);
  }

  KOKKOS_INLINE_FUNCTION
  StdPartitionPointFunctor(IteratorType first, ReducerType reducer,
                           PredicateType p)
      : m_first(first),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_p(::Kokkos::Experimental::move(p)) {}
};

template <class ValueType>
struct StdPartitionCopyScalar {
  ValueType true_count_;
  ValueType false_count_;

  KOKKOS_INLINE_FUNCTION
  StdPartitionCopyScalar& operator=(const StdPartitionCopyScalar& other) {
    true_count_  = other.true_count_;
    false_count_ = other.false_count_;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const volatile StdPartitionCopyScalar& other) {
    true_count_  = other.true_count_;
    false_count_ = other.false_count_;
  }

  // this is needed for
  // OpenMPTarget/Kokkos_OpenMPTarget_Parallel.hpp:699:21: error: no viable
  // overloaded '=' m_returnvalue = 0;
  //
  KOKKOS_INLINE_FUNCTION
  void operator=(const ValueType value) {
    true_count_  = value;
    false_count_ = value;
  }
};

template <class IndexType, class FirstFrom, class FirstDestTrue,
          class FirstDestFalse, class PredType>
struct StdPartitionCopyFunctor {
  using value_type = StdPartitionCopyScalar<IndexType>;

  FirstFrom m_first_from;
  FirstDestTrue m_first_dest_true;
  FirstDestFalse m_first_dest_false;
  PredType m_pred;

  KOKKOS_INLINE_FUNCTION
  StdPartitionCopyFunctor(FirstFrom first_from, FirstDestTrue first_dest_true,
                          FirstDestFalse first_dest_false, PredType pred)
      : m_first_from(first_from),
        m_first_dest_true(first_dest_true),
        m_first_dest_false(first_dest_false),
        m_pred(::Kokkos::Experimental::move(pred)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    const auto& myval = *(m_first_from + i);
    if (final_pass) {
      if (m_pred(myval)) {
        *(m_first_dest_true + update.true_count_) = myval;
      } else {
        *(m_first_dest_false + update.false_count_) = myval;
      }
    }

    if (m_pred(myval)) {
      update.true_count_ += 1;
    } else {
      update.false_count_ += 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& update) const {
    update.true_count_  = 0;
    update.false_count_ = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& update,
            volatile const value_type& input) const {
    update.true_count_ += input.true_count_;
    update.false_count_ += input.false_count_;
  }
};

//
// impl functions
//
template <class ExecutionSpace, class IteratorType, class PredicateType>
bool is_partitioned_impl(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         PredicateType pred) {
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();

  if (first == last) {
    return true;
  }

  const auto num_elements = last - first;
  using index_type        = int;
  using reducer_type      = StdIsPartitioned<index_type, ExecutionSpace>;

  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("is_partitioned_impl_result_view");
  reducer_type reducer(result);
  ::Kokkos::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      StdIsPartitionedFunctor<IteratorType, reducer_type, PredicateType>(
          first, reducer, pred),
      reducer);
  ex.fence("is_partitioned: fence after operation");

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  if (r_h().max_loc_true != ::Kokkos::reduction_identity<index_type>::max() &&
      r_h().min_loc_false != ::Kokkos::reduction_identity<index_type>::min()) {
    return r_h().max_loc_true < r_h().min_loc_false;
  } else if (first + r_h().max_loc_true == --last) {
    return true;
  } else {
    return false;
  }
}

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, PredicateType pred) {
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();

  if (first == last) {
    return first;
  }

  const auto num_elements = last - first;
  using index_type        = int;
  using reducer_type      = StdPartitionPoint<index_type, ExecutionSpace>;

  using result_view_type = typename reducer_type::result_view_type;
  result_view_type result("partition_point_impl_result_view");
  reducer_type reducer(result);
  ::Kokkos::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      StdPartitionPointFunctor<IteratorType, reducer_type, PredicateType>(
          first, reducer, pred),
      reducer);
  ex.fence("partition_point: fence after operation");

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  if (r_h().min_loc_false == ::Kokkos::reduction_identity<index_type>::min()) {
    // if all elements are true, return last
    return last;
  } else {
    return first + r_h().min_loc_false;
  }
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy_impl(const std::string& label, const ExecutionSpace& ex,
                    InputIteratorType from_first, InputIteratorType from_last,
                    OutputIteratorTrueType to_first_true,
                    OutputIteratorFalseType to_first_false,
                    PredicateType pred) {
  static_assert_random_access_and_accessible<ExecutionSpace, InputIteratorType,
                                             OutputIteratorTrueType,
                                             OutputIteratorFalseType>();

  if (from_first == from_last) {
    return {to_first_true, to_first_false};
  }

  using index_type = std::size_t;
  using func_type =
      StdPartitionCopyFunctor<index_type, InputIteratorType,
                              OutputIteratorTrueType, OutputIteratorFalseType,
                              PredicateType>;

  const auto num_elements = from_last - from_first;
  typename func_type::value_type counts{0, 0};
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(from_first, to_first_true, to_first_false, pred), counts);
  return {to_first_true + counts.true_count_,
          to_first_false + counts.false_count_};
}

// ------------------------------------------
}  // end namespace Impl
// ------------------------------------------

// ----------------------
// is_partitioned public API
// ----------------------
template <class ExecutionSpace, class IteratorType, class PredicateType>
bool is_partitioned(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, PredicateType p) {
  return Impl::is_partitioned_impl("kokkos_is_partitioned_iterator_api_default",
                                   ex, first, last, p);
}

template <class ExecutionSpace, class IteratorType, class PredicateType>
bool is_partitioned(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last, PredicateType p) {
  return Impl::is_partitioned_impl(label, ex, first, last, p);
}

template <class ExecutionSpace, class PredicateType, class DataType,
          class... Properties>
bool is_partitioned(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    PredicateType p) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::is_partitioned_impl("kokkos_is_partitioned_view_api_default", ex,
                                   cbegin(v), cend(v), p);
}

template <class ExecutionSpace, class PredicateType, class DataType,
          class... Properties>
bool is_partitioned(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    PredicateType p) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::is_partitioned_impl(label, ex, cbegin(v), cend(v), p);
}

// ----------------------
// partition_copy
// ----------------------
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType> partition_copy(
    const ExecutionSpace& ex, InputIteratorType from_first,
    InputIteratorType from_last, OutputIteratorTrueType to_first_true,
    OutputIteratorFalseType to_first_false, PredicateType p) {
  return Impl::partition_copy_impl("kokkos_partition_copy_iterator_api_default",
                                   ex, from_first, from_last, to_first_true,
                                   to_first_false, p);
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorTrueType, class OutputIteratorFalseType,
          class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType> partition_copy(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType from_first, InputIteratorType from_last,
    OutputIteratorTrueType to_first_true,
    OutputIteratorFalseType to_first_false, PredicateType p) {
  return Impl::partition_copy_impl(label, ex, from_first, from_last,
                                   to_first_true, to_first_false, p);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class PredicateType>
auto partition_copy(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
    PredicateType p) {
  return Impl::partition_copy_impl(
      "kokkos_partition_copy_view_api_default", ex, cbegin(view_from),
      cend(view_from), begin(view_dest_true), begin(view_dest_false), p);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class DataType3,
          class... Properties3, class PredicateType>
auto partition_copy(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
    const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
    PredicateType p) {
  return Impl::partition_copy_impl(label, ex, cbegin(view_from),
                                   cend(view_from), begin(view_dest_true),
                                   begin(view_dest_false), p);
}

// ----------------------
// partition_point
// ----------------------
template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
IteratorType partition_point(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last, UnaryPredicate p) {
  return Impl::partition_point_impl(
      "kokkos_partitioned_point_iterator_api_default", ex, first, last, p);
}

template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
IteratorType partition_point(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last,
                             UnaryPredicate p) {
  return Impl::partition_point_impl(label, ex, first, last, p);
}

template <class ExecutionSpace, class UnaryPredicate, class DataType,
          class... Properties>
auto partition_point(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& v,
                     UnaryPredicate p) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  return Impl::partition_point_impl(label, ex, cbegin(v), cend(v), p);
}

template <class ExecutionSpace, class UnaryPredicate, class DataType,
          class... Properties>
auto partition_point(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& v,
                     UnaryPredicate p) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  return Impl::partition_point_impl("kokkos_partition_point_view_api_default",
                                    ex, cbegin(v), cend(v), p);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
