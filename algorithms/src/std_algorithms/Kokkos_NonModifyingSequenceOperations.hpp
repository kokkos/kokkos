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

#ifndef KOKKOS_NON_MODIFYING_SEQUENCE_OPERATIONS_HPP
#define KOKKOS_NON_MODIFYING_SEQUENCE_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_Constraints.hpp"
#include "Kokkos_MinMaxOperations.hpp"
#include "Kokkos_ModifyingOperations.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include "Kokkos_RandomAccessIterator.hpp"
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

// ------------------------------------------
//
// functors
//
// ------------------------------------------

template <bool is_find_if, class IndexType, class IteratorType,
          class ReducerType, class PredicateType>
struct StdFindIfOrNotFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    const auto& my_value = m_first[i];

    // if doing find_if, look for when predicate is true
    // if doing find_if_not, look for when predicate is false
    const bool found_condition = is_find_if ? m_p(my_value) : !m_p(my_value);

    auto rv =
        found_condition
            ? red_value_type{i}
            : red_value_type{::Kokkos::reduction_identity<IndexType>::min()};

    m_reducer.join(red_value, rv);
  }

  KOKKOS_INLINE_FUNCTION
  StdFindIfOrNotFunctor(IteratorType first, ReducerType reducer,
                        PredicateType p)
      : m_first(first),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_p(::Kokkos::Experimental::move(p)) {}
};

template <class IteratorType, class UnaryFunctorType>
struct StdForEachFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  UnaryFunctorType m_functor;

  KOKKOS_INLINE_FUNCTION
  void operator()(index_type i) const { m_functor(m_first[i]); }

  KOKKOS_INLINE_FUNCTION
  StdForEachFunctor(IteratorType _first, UnaryFunctorType _functor)
      : m_first(_first), m_functor(::Kokkos::Experimental::move(_functor)) {}
};

template <class IteratorType, class Predicate>
struct StdCountIfFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  Predicate m_predicate;

  KOKKOS_INLINE_FUNCTION
  void operator()(index_type i, index_type& lsum) const {
    if (m_predicate(m_first[i])) {
      lsum++;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdCountIfFunctor(IteratorType _first, Predicate _predicate)
      : m_first(_first),
        m_predicate(::Kokkos::Experimental::move(_predicate)) {}
};

template <class IndexType, class IteratorType1, class IteratorType2,
          class ReducerType, class BinaryPredicateType>
struct StdMismatchRedFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType1 m_first1;
  IteratorType2 m_first2;
  ReducerType m_reducer;
  BinaryPredicateType m_predicate;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    const auto& my_value1 = m_first1[i];
    const auto& my_value2 = m_first2[i];
    m_reducer.join(red_value,
                   red_value_type{!m_predicate(my_value1, my_value2), i});
  }

  KOKKOS_INLINE_FUNCTION
  StdMismatchRedFunctor(IteratorType1 first1, IteratorType2 first2,
                        ReducerType reducer, BinaryPredicateType predicate)
      : m_first1(first1),
        m_first2(first2),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_predicate(::Kokkos::Experimental::move(predicate)) {}
};

template <class IndexType, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
struct StdEqualFunctor {
  IteratorType1 m_first1;
  IteratorType2 m_first2;
  BinaryPredicateType m_predicate;

  KOKKOS_INLINE_FUNCTION
  void operator()(IndexType i, std::size_t& lsum) const {
    if (!m_predicate(m_first1[i], m_first2[i])) {
      lsum = 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdEqualFunctor(IteratorType1 _first1, IteratorType2 _first2,
                  BinaryPredicateType _predicate)
      : m_first1(_first1), m_first2(_first2), m_predicate(_predicate) {}
};

template <class IndexType, class IteratorType1, class IteratorType2,
          class ReducerType, class ComparatorType>
struct StdLexicographicalCompareFunctor {
  using red_value_type = typename ReducerType::value_type;
  IteratorType1 m_first1;
  IteratorType2 m_first2;
  ReducerType m_reducer;
  ComparatorType m_comparator;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    const auto& my_value1 = m_first1[i];
    const auto& my_value2 = m_first2[i];

    bool different = m_comparator(my_value1, my_value2) ||
                     m_comparator(my_value2, my_value1);
    auto rv =
        different
            ? red_value_type{i}
            : red_value_type{::Kokkos::reduction_identity<IndexType>::min()};

    m_reducer.join(red_value, rv);
  }

  KOKKOS_INLINE_FUNCTION
  StdLexicographicalCompareFunctor(IteratorType1 _first1, IteratorType2 _first2,
                                   ReducerType _reducer, ComparatorType _comp)
      : m_first1(_first1),
        m_first2(_first2),
        m_reducer(_reducer),
        m_comparator(_comp) {}
};

template <class IndexType, class IteratorType1, class IteratorType2,
          class ComparatorType>
struct StdCompareFunctor {
  IteratorType1 m_it1;
  IteratorType2 m_it2;
  ComparatorType m_predicate;

  KOKKOS_INLINE_FUNCTION
  void operator()(IndexType /* i is unused */, int& lsum) const {
    if (m_predicate(*m_it1, *m_it2)) {
      lsum = 1;
    }
  }

  KOKKOS_INLINE_FUNCTION
  StdCompareFunctor(IteratorType1 _it1, IteratorType2 _it2,
                    ComparatorType _predicate)
      : m_it1(_it1),
        m_it2(_it2),
        m_predicate(::Kokkos::Experimental::move(_predicate)) {}
};

template <class IndexType, class IteratorType, class ReducerType,
          class PredicateType>
struct StdAdjacentFindFunctor {
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;
  PredicateType m_p;

  KOKKOS_INLINE_FUNCTION
  void operator()(const IndexType i, red_value_type& red_value) const {
    const auto& my_value   = m_first[i];
    const auto& next_value = m_first[i + 1];
    const bool are_equal   = m_p(my_value, next_value);

    auto rv =
        are_equal
            ? red_value_type{i}
            : red_value_type{::Kokkos::reduction_identity<IndexType>::min()};

    m_reducer.join(red_value, rv);
  }

  KOKKOS_INLINE_FUNCTION
  StdAdjacentFindFunctor(IteratorType first, ReducerType reducer,
                         PredicateType p)
      : m_first(first),
        m_reducer(::Kokkos::Experimental::move(reducer)),
        m_p(::Kokkos::Experimental::move(p)) {}
};

// ------------------------------------------
// find_if_or_not_impl
// ------------------------------------------
template <bool is_find_if, class ExecutionSpace, class IteratorType,
          class PredicateType>
IteratorType find_if_or_not_impl(const std::string& label,
                                 const ExecutionSpace& ex, IteratorType first,
                                 IteratorType last, PredicateType pred) {
  // checks
  static_assert_random_access_and_accessible(ex, first, last);
  expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type       = typename IteratorType::difference_type;
  using reducer_type     = FirstLoc<index_type, ExecutionSpace>;
  using result_view_type = typename reducer_type::result_view_type;
  using func_t = StdFindIfOrNotFunctor<is_find_if, index_type, IteratorType,
                                       reducer_type, PredicateType>;

  // run
  result_view_type result("kokkos_find_if_impl_result_view");
  reducer_type reducer(result);
  const auto num_elements = last - first;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer, pred), reducer);
  ex.fence("find_if_or_not: fence after operation");

  // decide and return
  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  if (r_h().min_loc_true == ::Kokkos::reduction_identity<index_type>::min()) {
    // here, it means a valid loc has not been found,
    return last;
  } else {
    // a location has been found
    return first + r_h().min_loc_true;
  }
}

// ------------------------------------------
// find_impl
// ------------------------------------------
template <class ExecutionSpace, class InputIterator, class T>
InputIterator find_impl(const std::string& label, ExecutionSpace ex,
                        InputIterator first, InputIterator last,
                        const T& value) {
  return find_if_or_not_impl<true>(
      label, ex, first, last,
      ::Kokkos::Experimental::Impl::StdAlgoEqualsValUnaryPredicate<T>(value));
}

// ------------------------------------------
// for_each_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each_impl(const std::string& label,
                               const ExecutionSpace& ex, IteratorType first,
                               IteratorType last, UnaryFunctorType functor) {
  // checks
  static_assert_random_access_and_accessible(ex, first, last);
  expect_valid_range(first, last);

  // run
  const auto num_elements = last - first;
  ::Kokkos::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      StdForEachFunctor<IteratorType, UnaryFunctorType>(first, functor));
  ex.fence("for_each: fence after operation");

  // return
  return functor;
}

// ------------------------------------------
// for_each_n_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType, class SizeType,
          class UnaryFunctorType>
IteratorType for_each_n_impl(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, SizeType n,
                             UnaryFunctorType functor) {
  auto last = first + n;
  static_assert_random_access_and_accessible(ex, first, last);
  expect_valid_range(first, last);

  if (n == 0) {
    return first;
  }

  for_each_impl(label, ex, first, last, ::Kokkos::Experimental::move(functor));
  // no neeed to fence since for_each_impl fences already

  return last;
}

// ------------------------------------------
// count_if_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if_impl(const std::string& label,
                                                     const ExecutionSpace& ex,
                                                     IteratorType first,
                                                     IteratorType last,
                                                     Predicate predicate) {
  // checks
  static_assert_random_access_and_accessible(ex, first, last);
  expect_valid_range(first, last);

  // aliases
  using func_t = StdCountIfFunctor<IteratorType, Predicate>;

  // run
  const auto num_elements                      = last - first;
  typename IteratorType::difference_type count = 0;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, predicate), count);
  ex.fence("count_if: fence after operation");

  // return
  return count;
}

// ------------------------------------------
// count_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType, class T>
std::size_t count_impl(const std::string& label, const ExecutionSpace& ex,
                       IteratorType first, IteratorType last, const T& value) {
  return count_if_impl(
      label, ex, first, last,
      ::Kokkos::Experimental::Impl::StdAlgoEqualsValUnaryPredicate<T>(value));
}

// ------------------------------------------
// mismatch_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2,
    BinaryPredicateType predicate) {
  // checks
  static_assert_random_access_and_accessible(ex, first1, last1, first2, last2);
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();
  expect_valid_range(first1, last1);
  expect_valid_range(first2, last2);

  // aliases
  using index_type       = typename IteratorType1::difference_type;
  using reducer_type     = StdMismatch<index_type, ExecutionSpace>;
  using result_view_type = typename reducer_type::result_view_type;
  using functor_type =
      StdMismatchRedFunctor<index_type, IteratorType1, IteratorType2,
                            reducer_type, BinaryPredicateType>;

  // run
  const auto num_e1                = last1 - first1;
  const auto num_e2                = last2 - first2;
  auto num_elements_for_par_reduce = (num_e1 <= num_e2) ? num_e1 : num_e2;
  result_view_type result("mismatch_impl_result");
  reducer_type reducer(result);
  ::Kokkos::parallel_reduce(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_for_par_reduce),
      functor_type(first1, first2, reducer, std::move(predicate)), reducer);
  ex.fence("mismatch: fence after operation");

  // decide and return
  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  using return_type = ::Kokkos::pair<IteratorType1, IteratorType2>;
  if (r_h().loc == ::Kokkos::reduction_identity<index_type>::min()) {
    // in here means mismatch has not been found

    if (num_e1 == num_e2) {
      return return_type(last1, last2);
    } else if (num_e1 < num_e2) {
      return return_type(last1, first2 + num_e1);
    } else {
      return return_type(last1 + num_e2, last2);
    }
  } else {
    // in here means mismatch has been found
    return return_type(first1 + r_h().loc, first2 + r_h().loc);
  }
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2) {
  // checks
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();
  expect_valid_range(first1, last1);
  expect_valid_range(first2, last2);

  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return mismatch_impl(label, ex, first1, last1, first2, last2, pred_t());
}

// ------------------------------------------
// all_of_impl, any_of_impl, none_of_impl
// ------------------------------------------
template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of_impl(const std::string& label, const ExecutionSpace& ex,
                 InputIterator first, InputIterator last, Predicate predicate) {
  return (find_if_or_not_impl<false>(label, ex, first, last, predicate) ==
          last);
}

template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of_impl(const std::string& label, const ExecutionSpace& ex,
                 InputIterator first, InputIterator last, Predicate predicate) {
  return (find_if_or_not_impl<true>(label, ex, first, last, predicate) != last);
}

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of_impl(const std::string& label, const ExecutionSpace& ex,
                  IteratorType first, IteratorType last, Predicate predicate) {
  return (find_if_or_not_impl<true>(label, ex, first, last, predicate) == last);
}

// ------------------------------------------
// equal_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal_impl(const std::string& label, const ExecutionSpace& ex,
                IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
                BinaryPredicateType predicate) {
  // checks
  static_assert_random_access_and_accessible(ex, first1, last1, first2);
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();
  expect_valid_range(first1, last1);

  // aliases
  using index_type = typename IteratorType1::difference_type;
  using func_t     = StdEqualFunctor<index_type, IteratorType1, IteratorType2,
                                 BinaryPredicateType>;

  // run
  const auto num_elements = last1 - first1;
  std::size_t different   = 0;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first1, first2, predicate), different);
  ex.fence("equal: fence after operation");

  // return
  return !different;
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal_impl(const std::string& label, const ExecutionSpace& ex,
                IteratorType1 first1, IteratorType1 last1,
                IteratorType2 first2) {
  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_impl(label, ex, first1, last1, first2, pred_t());
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal_impl(const std::string& label, const ExecutionSpace& ex,
                IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
                IteratorType2 last2, BinaryPredicateType predicate) {
  const auto d1 = ::Kokkos::Experimental::distance(first1, last1);
  const auto d2 = ::Kokkos::Experimental::distance(first2, last2);
  if (d1 != d2) {
    return false;
  }

  return equal_impl(label, ex, first1, last1, first2, predicate);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal_impl(const std::string& label, const ExecutionSpace& ex,
                IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
                IteratorType2 last2) {
  expect_valid_range(first1, last1);
  expect_valid_range(first2, last2);

  using value_type1 = typename IteratorType1::value_type;
  using value_type2 = typename IteratorType2::value_type;
  using pred_t      = StdAlgoEqualBinaryPredicate<value_type1, value_type2>;
  return equal_impl(label, ex, first1, last1, first2, last2, pred_t());
}

// ------------------------------------------
// lexicographical_compare_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ComparatorType>
bool lexicographical_compare_impl(const std::string& label,
                                  const ExecutionSpace& ex,
                                  IteratorType1 first1, IteratorType1 last1,
                                  IteratorType2 first2, IteratorType2 last2,
                                  ComparatorType comp) {
  // checks
  static_assert_random_access_and_accessible(ex, first1, last1, first2, last2);
  static_assert_iterators_have_matching_difference_type<IteratorType1,
                                                        IteratorType2>();
  expect_valid_range(first1, last1);
  expect_valid_range(first2, last2);

  // aliases
  using index_type       = typename IteratorType1::difference_type;
  using reducer_type     = FirstLoc<index_type, ExecutionSpace>;
  using result_view_type = typename reducer_type::result_view_type;

  // run
  auto d1    = Kokkos::Experimental::distance(first1, last1);
  auto d2    = Kokkos::Experimental::distance(first2, last2);
  auto range = Kokkos::Experimental::min(d1, d2);
  result_view_type result("kokkos_lexicographical_compare_impl_result_view");
  reducer_type reducer(result);
  using func1_t =
      StdLexicographicalCompareFunctor<index_type, IteratorType1, IteratorType2,
                                       reducer_type, ComparatorType>;

  ::Kokkos::parallel_reduce(label, RangePolicy<ExecutionSpace>(ex, 0, range),
                            func1_t(first1, first2, reducer, comp), reducer);
  ex.fence("lexicographical_compare: fence after first reduce operation");

  const auto r_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);

  // no mismatch
  if (r_h().min_loc_true == ::Kokkos::reduction_identity<index_type>::min()) {
    auto new_last1 = first1 + range;
    auto new_last2 = first2 + range;
    bool is_prefix = (new_last1 == last1) && (new_last2 != last2);
    return is_prefix;
  }

  // check mismatched
  int less      = 0;
  auto it1      = first1 + r_h().min_loc_true;
  auto it2      = first2 + r_h().min_loc_true;
  using func2_t = StdCompareFunctor<index_type, IteratorType1, IteratorType2,
                                    ComparatorType>;
  ::Kokkos::parallel_reduce(label, RangePolicy<ExecutionSpace>(ex, 0, 1),
                            func2_t(it1, it2, comp), less);
  ex.fence("lexicographical_compare: fence after second reduce operation");

  return static_cast<bool>(less);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool lexicographical_compare_impl(const std::string& label,
                                  const ExecutionSpace& ex,
                                  IteratorType1 first1, IteratorType1 last1,
                                  IteratorType2 first2, IteratorType2 last2) {
  using value_type_1 = typename IteratorType1::value_type;
  using value_type_2 = typename IteratorType2::value_type;
  using predicate_t =
      Impl::StdAlgoLessThanBinaryPredicate<value_type_1, value_type_2>;
  return lexicographical_compare_impl(label, ex, first1, last1, first2, last2,
                                      predicate_t());
}

// ------------------------------------------
// adjacent_find_impl
// ------------------------------------------
template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType adjacent_find_impl(const std::string& label,
                                const ExecutionSpace& ex, IteratorType first,
                                IteratorType last, PredicateType pred) {
  // checks
  static_assert_random_access_and_accessible(ex, first, last);
  expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  const auto num_elements = last - first;

  if (num_elements == 1) {
    return first + 1;
  } else {
    using index_type       = typename IteratorType::difference_type;
    using reducer_type     = FirstLoc<index_type, ExecutionSpace>;
    using result_view_type = typename reducer_type::result_view_type;
    using func_t           = StdAdjacentFindFunctor<index_type, IteratorType,
                                          reducer_type, PredicateType>;

    result_view_type result("kokkos_adjacent_find_impl_result_view");
    reducer_type reducer(result);

    // note that we use below num_elements-1 because
    // each index i in the reduction checks i and (i+1).
    ::Kokkos::parallel_reduce(
        label, RangePolicy<ExecutionSpace>(ex, 0, num_elements - 1),
        func_t(first, reducer, pred), reducer);
    ex.fence("adjacent_find: fence after operation");

    const auto r_h =
        ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);
    if (r_h().min_loc_true == ::Kokkos::reduction_identity<index_type>::min()) {
      return last;
    } else {
      return first + r_h().min_loc_true;
    }
  }
}

template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find_impl(const std::string& label,
                                const ExecutionSpace& ex, IteratorType first,
                                IteratorType last) {
  using value_type     = typename IteratorType::value_type;
  using default_pred_t = StdAlgoEqualBinaryPredicate<value_type>;
  return adjacent_find_impl(label, ex, first, last, default_pred_t());
}

}  // namespace Impl

// ----------------------------------
// find public API
// ----------------------------------
template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const ExecutionSpace& ex, InputIterator first,
                   InputIterator last, const T& value) {
  return Impl::find_impl("kokkos_find_iterator_api_default", ex, first, last,
                         value);
}

template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const std::string& label, const ExecutionSpace& ex,
                   InputIterator first, InputIterator last, const T& value) {
  return Impl::find_impl(label, ex, first, last, value);
}

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const ExecutionSpace& ex,
          const ::Kokkos::View<DataType, Properties...>& view, const T& value) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_impl("kokkos_find_view_api_default", ex, KE::cbegin(view),
                         KE::cend(view), value);
}

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const std::string& label, const ExecutionSpace& ex,
          const ::Kokkos::View<DataType, Properties...>& view, const T& value) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_impl(label, ex, KE::cbegin(view), KE::cend(view), value);
}

// -------------------
// find_if public API
// -------------------
template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType find_if(const ExecutionSpace& ex, IteratorType first,
                     IteratorType last, PredicateType predicate) {
  return Impl::find_if_or_not_impl<true>(
      "kokkos_find_if_iterator_api_default", ex, first, last,
      ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType find_if(const std::string& label, const ExecutionSpace& ex,
                     IteratorType first, IteratorType last,
                     PredicateType predicate) {
  return Impl::find_if_or_not_impl<true>(
      label, ex, first, last, ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto find_if(const ExecutionSpace& ex,
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::find_if_or_not_impl<true>("kokkos_find_if_view_api_default", ex,
                                         KE::cbegin(v), KE::cend(v),
                                         KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto find_if(const std::string& label, const ExecutionSpace& ex,
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::find_if_or_not_impl<true>(label, ex, KE::cbegin(v), KE::cend(v),
                                         KE::move(predicate));
}

// ----------------------------------
// find_if_not public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class Predicate>
IteratorType find_if_not(const ExecutionSpace& ex, IteratorType first,
                         IteratorType last, Predicate predicate) {
  return Impl::find_if_or_not_impl<false>(
      "kokkos_find_if_not_iterator_api_default", ex, first, last,
      ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class IteratorType, class Predicate>
IteratorType find_if_not(const std::string& label, const ExecutionSpace& ex,
                         IteratorType first, IteratorType last,
                         Predicate predicate) {
  return Impl::find_if_or_not_impl<false>(
      label, ex, first, last, ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto find_if_not(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_if_or_not_impl<false>("kokkos_find_if_not_view_api_default",
                                          ex, KE::cbegin(v), KE::cend(v),
                                          KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto find_if_not(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::find_if_or_not_impl<false>(label, ex, KE::cbegin(v), KE::cend(v),
                                          KE::move(predicate));
}

// ----------------------------------
// for_each public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& ex,
                          IteratorType first, IteratorType last,
                          UnaryFunctorType functor) {
  return Impl::for_each_impl(label, ex, first, last,
                             ::Kokkos::Experimental::move(functor));
}

template <class ExecutionSpace, class IteratorType, class UnaryFunctorType>
UnaryFunctorType for_each(const ExecutionSpace& ex, IteratorType first,
                          IteratorType last, UnaryFunctorType functor) {
  return Impl::for_each_impl("kokkos_for_each_iterator_api_default", ex, first,
                             last, ::Kokkos::Experimental::move(functor));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryFunctorType>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& ex,
                          const ::Kokkos::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::for_each_impl(label, ex, KE::begin(v), KE::end(v),
                             KE::move(functor));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryFunctorType>
UnaryFunctorType for_each(const ExecutionSpace& ex,
                          const ::Kokkos::View<DataType, Properties...>& v,
                          UnaryFunctorType functor) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::for_each_impl("kokkos_for_each_view_api_default", ex,
                             KE::begin(v), KE::end(v), KE::move(functor));
}

// ----------------------------------
// for_each_n public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class SizeType,
          class UnaryFunctorType>
IteratorType for_each_n(const std::string& label, const ExecutionSpace& ex,
                        IteratorType first, SizeType n,
                        UnaryFunctorType functor) {
  return Impl::for_each_n_impl(label, ex, first, n,
                               ::Kokkos::Experimental::move(functor));
}

template <class ExecutionSpace, class IteratorType, class SizeType,
          class UnaryFunctorType>
IteratorType for_each_n(const ExecutionSpace& ex, IteratorType first,
                        SizeType n, UnaryFunctorType functor) {
  return Impl::for_each_n_impl("kokkos_for_each_n_iterator_api_default", ex,
                               first, n, ::Kokkos::Experimental::move(functor));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class UnaryFunctorType>
auto for_each_n(const std::string& label, const ExecutionSpace& ex,
                const ::Kokkos::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::for_each_n_impl(label, ex, KE::begin(v), n, KE::move(functor));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class UnaryFunctorType>
auto for_each_n(const ExecutionSpace& ex,
                const ::Kokkos::View<DataType, Properties...>& v, SizeType n,
                UnaryFunctorType functor) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::for_each_n_impl("kokkos_for_each_n_view_api_default", ex,
                               KE::begin(v), n, KE::move(functor));
}

// ----------------------------------
// count_if public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const ExecutionSpace& ex,
                                                IteratorType first,
                                                IteratorType last,
                                                Predicate predicate) {
  return Impl::count_if_impl("kokkos_count_if_iterator_api_default", ex, first,
                             last, ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const std::string& label,
                                                const ExecutionSpace& ex,
                                                IteratorType first,
                                                IteratorType last,
                                                Predicate predicate) {
  return Impl::count_if_impl(label, ex, first, last,
                             ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& v,
              Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::count_if_impl("kokkos_count_if_view_api_default", ex,
                             KE::cbegin(v), KE::cend(v), KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& v,
              Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::count_if_impl(label, ex, KE::cbegin(v), KE::cend(v),
                             KE::move(predicate));
}

// ----------------------------------
// count public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class T>
typename IteratorType::difference_type count(const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,
                                             const T& value) {
  return Impl::count_impl("kokkos_count_iterator_api_default", ex, first, last,
                          value);
}

template <class ExecutionSpace, class IteratorType, class T>
typename IteratorType::difference_type count(const std::string& label,
                                             const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,
                                             const T& value) {
  return Impl::count_impl(label, ex, first, last, value);
}

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto count(const ExecutionSpace& ex,
           const ::Kokkos::View<DataType, Properties...>& v, const T& value) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::count_impl("kokkos_count_view_api_default", ex, KE::cbegin(v),
                          KE::cend(v), value);
}

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto count(const std::string& label, const ExecutionSpace& ex,
           const ::Kokkos::View<DataType, Properties...>& v, const T& value) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::count_impl(label, ex, KE::cbegin(v), KE::cend(v), value);
}

// ----------------------------------
// mismatch public API
// ----------------------------------
// FIXME: add mismatch overloads accepting 3 iterators
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& ex,
                                                      IteratorType1 first1,
                                                      IteratorType1 last1,
                                                      IteratorType2 first2,
                                                      IteratorType2 last2) {
  return Impl::mismatch_impl("kokkos_mismatch_iterator_api_default", ex, first1,
                             last1, first2, last2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch(
    const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
    IteratorType2 first2, IteratorType2 last2,
    BinaryPredicateType&& predicate) {
  return Impl::mismatch_impl("kokkos_mismatch_iterator_api_default", ex, first1,
                             last1, first2, last2,
                             std::forward<BinaryPredicateType>(predicate));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2) {
  return Impl::mismatch_impl(label, ex, first1, last1, first2, last2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
::Kokkos::pair<IteratorType1, IteratorType2> mismatch(
    const std::string& label, const ExecutionSpace& ex, IteratorType1 first1,
    IteratorType1 last1, IteratorType2 first2, IteratorType2 last2,
    BinaryPredicateType&& predicate) {
  return Impl::mismatch_impl(label, ex, first1, last1, first2, last2,
                             std::forward<BinaryPredicateType>(predicate));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view1,
              const ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::mismatch_impl("kokkos_mismatch_view_api_default", ex,
                             KE::cbegin(view1), KE::cend(view1),
                             KE::cbegin(view2), KE::cend(view2));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view1,
              const ::Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::mismatch_impl("kokkos_mismatch_view_api_default", ex,
                             KE::cbegin(view1), KE::cend(view1),
                             KE::cbegin(view2), KE::cend(view2),
                             std::forward<BinaryPredicateType>(predicate));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view1,
              const ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::mismatch_impl(label, ex, KE::cbegin(view1), KE::cend(view1),
                             KE::cbegin(view2), KE::cend(view2));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view1,
              const ::Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::mismatch_impl(label, ex, KE::cbegin(view1), KE::cend(view1),
                             KE::cbegin(view2), KE::cend(view2),
                             std::forward<BinaryPredicateType>(predicate));
}

// ----------------------------------
// all_of public API
// ----------------------------------
template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of(const ExecutionSpace& ex, InputIterator first, InputIterator last,
            Predicate predicate) {
  return Impl::all_of_impl("kokkos_all_of_iterator_api_default", ex, first,
                           last, predicate);
}

template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator last, Predicate predicate) {
  return Impl::all_of_impl(label, ex, first, last, predicate);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool all_of(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& v,
            Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::all_of_impl("kokkos_all_of_view_api_default", ex, KE::cbegin(v),
                           KE::cend(v), KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool all_of(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& v,
            Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::all_of_impl(label, ex, KE::cbegin(v), KE::cend(v),
                           KE::move(predicate));
}

// ----------------------------------
// any_of public API
// ----------------------------------
template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of(const ExecutionSpace& ex, InputIterator first, InputIterator last,
            Predicate predicate) {
  return Impl::any_of_impl("kokkos_any_of_view_api_default", ex, first, last,
                           predicate);
}

template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator last, Predicate predicate) {
  return Impl::any_of_impl(label, ex, first, last, predicate);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool any_of(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& v,
            Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::any_of_impl("kokkos_any_of_view_api_default", ex, KE::cbegin(v),
                           KE::cend(v), KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool any_of(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& v,
            Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::any_of_impl(label, ex, KE::cbegin(v), KE::cend(v),
                           KE::move(predicate));
}

// ----------------------------------
// none_of public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const ExecutionSpace& ex, IteratorType first, IteratorType last,
             Predicate predicate) {
  return Impl::none_of_impl("kokkos_none_of_iterator_api_default", ex, first,
                            last, predicate);
}

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& ex,
             IteratorType first, IteratorType last, Predicate predicate) {
  return Impl::none_of_impl(label, ex, first, last, predicate);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const ExecutionSpace& ex,
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::none_of_impl("kokkos_none_of_view_api_default", ex,
                            KE::cbegin(v), KE::cend(v), KE::move(predicate));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& ex,
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  namespace KE = ::Kokkos::Experimental;
  return Impl::none_of_impl(label, ex, KE::cbegin(v), KE::cend(v),
                            KE::move(predicate));
}

// ----------------------------------
// equal public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2) {
  return Impl::equal_impl("kokkos_equal_iterator_api_default", ex, first1,
                          last1, first2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2) {
  return Impl::equal_impl(label, ex, first1, last1, first2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, BinaryPredicateType predicate) {
  return Impl::equal_impl("kokkos_equal_iterator_api_default", ex, first1,
                          last1, first2,
                          ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           BinaryPredicateType predicate) {
  return Impl::equal_impl(label, ex, first1, last1, first2,
                          ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const ExecutionSpace& ex,
           const ::Kokkos::View<DataType1, Properties1...>& view1,
           ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::equal_impl("kokkos_equal_view_api_default", ex,
                          KE::cbegin(view1), KE::cend(view1),
                          KE::cbegin(view2));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const ::Kokkos::View<DataType1, Properties1...>& view1,
           ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::equal_impl(label, ex, KE::cbegin(view1), KE::cend(view1),
                          KE::cbegin(view2));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
bool equal(const ExecutionSpace& ex,
           const ::Kokkos::View<DataType1, Properties1...>& view1,
           ::Kokkos::View<DataType2, Properties2...>& view2,
           BinaryPredicateType predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::equal_impl("kokkos_equal_view_api_default", ex,
                          KE::cbegin(view1), KE::cend(view1), KE::cbegin(view2),
                          KE::move(predicate));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const ::Kokkos::View<DataType1, Properties1...>& view1,
           ::Kokkos::View<DataType2, Properties2...>& view2,
           BinaryPredicateType predicate) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::equal_impl(label, ex, KE::cbegin(view1), KE::cend(view1),
                          KE::cbegin(view2), KE::move(predicate));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, IteratorType2 last2) {
  return Impl::equal_impl("kokkos_equal_iterator_api_default", ex, first1,
                          last1, first2, last2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           IteratorType2 last2) {
  return Impl::equal_impl(label, ex, first1, last1, first2, last2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,
           IteratorType2 first2, IteratorType2 last2,
           BinaryPredicateType predicate) {
  return Impl::equal_impl("kokkos_equal_iterator_api_default", ex, first1,
                          last1, first2, last2,
                          ::Kokkos::Experimental::move(predicate));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,
           IteratorType2 last2, BinaryPredicateType predicate) {
  return Impl::equal_impl(label, ex, first1, last1, first2, last2,
                          ::Kokkos::Experimental::move(predicate));
}

// ----------------------------------
// lexicographical_compare public API
// ----------------------------------
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,
                             IteratorType2 last2) {
  return Impl::lexicographical_compare_impl(
      "kokkos_lexicographical_compare_iterator_api_default", ex, first1, last1,
      first2, last2);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,
                             IteratorType2 first2, IteratorType2 last2) {
  return Impl::lexicographical_compare_impl(label, ex, first1, last1, first2,
                                            last2);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,
    ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::lexicographical_compare_impl(
      "kokkos_lexicographical_compare_view_api_default", ex, KE::cbegin(view1),
      KE::cend(view1), KE::cbegin(view2), KE::cend(view2));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,
    ::Kokkos::View<DataType2, Properties2...>& view2) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::lexicographical_compare_impl(label, ex, KE::cbegin(view1),
                                            KE::cend(view1), KE::cbegin(view2),
                                            KE::cend(view2));
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ComparatorType>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,
                             IteratorType2 last2, ComparatorType comp) {
  return Impl::lexicographical_compare_impl(
      "kokkos_lexicographical_compare_iterator_api_default", ex, first1, last1,
      first2, last2, comp);
}

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ComparatorType>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,
                             IteratorType2 first2, IteratorType2 last2,
                             ComparatorType comp) {
  return Impl::lexicographical_compare_impl(label, ex, first1, last1, first2,
                                            last2, comp);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ComparatorType>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,
    ::Kokkos::View<DataType2, Properties2...>& view2, ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::lexicographical_compare_impl(
      "kokkos_lexicographical_compare_view_api_default", ex, KE::cbegin(view1),
      KE::cend(view1), KE::cbegin(view2), KE::cend(view2), comp);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ComparatorType>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,
    ::Kokkos::View<DataType2, Properties2...>& view2, ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(view1);
  static_assert_is_admissible_to_kokkos_std_algorithms(view2);

  namespace KE = ::Kokkos::Experimental;
  return Impl::lexicographical_compare_impl(label, ex, KE::cbegin(view1),
                                            KE::cend(view1), KE::cbegin(view2),
                                            KE::cend(view2), comp);
}

// ----------------------------------
// adjacent_find
// ----------------------------------
// overload set1
template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find(const ExecutionSpace& ex, IteratorType first,
                           IteratorType last) {
  return Impl::adjacent_find_impl("kokkos_adjacent_find_iterator_api_default",
                                  ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find(const std::string& label, const ExecutionSpace& ex,
                           IteratorType first, IteratorType last) {
  return Impl::adjacent_find_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_impl("kokkos_adjacent_find_view_api_default", ex,
                                  KE::cbegin(v), KE::cend(v));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_impl(label, ex, KE::cbegin(v), KE::cend(v));
}

// overload set2
template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
IteratorType adjacent_find(const ExecutionSpace& ex, IteratorType first,
                           IteratorType last, BinaryPredicateType pred) {
  return Impl::adjacent_find_impl("kokkos_adjacent_find_iterator_api_default",
                                  ex, first, last, pred);
}

template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
IteratorType adjacent_find(const std::string& label, const ExecutionSpace& ex,
                           IteratorType first, IteratorType last,
                           BinaryPredicateType pred) {
  return Impl::adjacent_find_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_impl("kokkos_adjacent_find_view_api_default", ex,
                                  KE::cbegin(v), KE::cend(v), pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_impl(label, ex, KE::cbegin(v), KE::cend(v), pred);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
