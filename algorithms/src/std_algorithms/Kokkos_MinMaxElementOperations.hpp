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

#ifndef KOKKOS_STD_MIN_MAX_ELEMENT_OPERATIONS_HPP
#define KOKKOS_STD_MIN_MAX_ELEMENT_OPERATIONS_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_BeginEnd.hpp"
#include "Kokkos_Constraints.hpp"
#include "Kokkos_ModifyingOperations.hpp"

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class ReducerType>
struct StdMinOrMaxElemFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;

  IteratorType m_first;
  ReducerType m_reducer;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    auto my_iterator = m_first + i;
    m_reducer.join(red_value, red_value_type{*my_iterator, i});
  }

  KOKKOS_INLINE_FUNCTION
  StdMinOrMaxElemFunctor(IteratorType first, ReducerType reducer)
      : m_first(first), m_reducer(::Kokkos::Experimental::move(reducer)) {}
};

template <class IteratorType, class ReducerType>
struct StdMinMaxElemFunctor {
  using index_type     = typename IteratorType::difference_type;
  using red_value_type = typename ReducerType::value_type;
  IteratorType m_first;
  ReducerType m_reducer;

  KOKKOS_INLINE_FUNCTION
  void operator()(const index_type i, red_value_type& red_value) const {
    auto my_iterator = m_first + i;
    m_reducer.join(red_value, red_value_type{*my_iterator, *my_iterator, i, i});
  }

  KOKKOS_INLINE_FUNCTION
  StdMinMaxElemFunctor(IteratorType first, ReducerType reducer)
      : m_first(first), m_reducer(::Kokkos::Experimental::move(reducer)) {}
};

// ------------------------------------------
// min_or_max_element_impl
// ------------------------------------------
template <template <class... Args> class ReducerType, class ExecutionSpace,
          class IteratorType, class... Args>
IteratorType min_or_max_element_impl(const std::string& label,
                                     const ExecutionSpace& ex,
                                     IteratorType first, IteratorType last,
                                     Args&&... args) {
  // checks
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();
  expect_valid_range(first, last);

  if (first == last) {
    return last;
  }

  // aliases
  using index_type   = typename IteratorType::difference_type;
  using value_type   = typename IteratorType::value_type;
  using reducer_type = typename std::conditional<
      (0 < sizeof...(Args)),
      ReducerType<value_type, index_type, Args..., ExecutionSpace>,
      ReducerType<value_type, index_type, ExecutionSpace> >::type;

  using result_view_type = typename reducer_type::result_view_type;
  using func_t           = StdMinOrMaxElemFunctor<IteratorType, reducer_type>;

  // run
  result_view_type result("min_or_max_elem_impl_result");
  reducer_type reducer(result, std::forward<Args>(args)...);
  const auto num_elements = last - first;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer), reducer);
  ex.fence("min_or_max_element: fence after operation");

  // return
  const auto result_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);
  return first + result_h().loc;
}

// ------------------------------------------
// minmax_element_impl
// ------------------------------------------
template <template <class... Args> class ReducerType, class ExecutionSpace,
          class IteratorType, class... Args>
::Kokkos::pair<IteratorType, IteratorType> minmax_element_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, Args&&... args) {
  // checks
  static_assert_random_access_and_accessible<ExecutionSpace, IteratorType>();
  expect_valid_range(first, last);

  if (first == last) {
    return {first, first};
  }

  // aliases
  using index_type   = typename IteratorType::difference_type;
  using value_type   = typename IteratorType::value_type;
  using reducer_type = typename std::conditional<
      (0 < sizeof...(Args)),
      ReducerType<value_type, index_type, Args..., ExecutionSpace>,
      ReducerType<value_type, index_type, ExecutionSpace> >::type;

  using result_view_type = typename reducer_type::result_view_type;
  using func_t           = StdMinMaxElemFunctor<IteratorType, reducer_type>;

  // run
  result_view_type result("minmax_elem_impl_result");
  reducer_type reducer(result, std::forward<Args>(args)...);
  const auto num_elements = last - first;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, reducer), reducer);
  ex.fence("minmax_element: fence after operation");

  // return
  const auto result_h =
      ::Kokkos::create_mirror_view_and_copy(::Kokkos::HostSpace(), result);
  return {first + result_h().min_loc, first + result_h().max_loc};
}

}  // end namespace Impl

// ----------------------
// min_element public API
// ----------------------
template <class ExecutionSpace, class IteratorType>
auto min_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last) {
  return Impl::min_or_max_element_impl<MinFirstLoc>(
      "kokkos_min_element_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last) {
  return Impl::min_or_max_element_impl<MinFirstLoc>(label, ex, first, last);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto min_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MinFirstLocCustomComparator>(
      "kokkos_min_element_iterator_api_default", ex, first, last,
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MinFirstLocCustomComparator>(
      label, ex, first, last, ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto min_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_impl<MinFirstLoc>(
      "kokkos_min_element_view_api_default", ex, cbegin(v), cend(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto min_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MinFirstLocCustomComparator>(
      "kokkos_min_element_view_api_default", ex, cbegin(v), cend(v),
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_impl<MinFirstLoc>(label, ex, cbegin(v),
                                                    cend(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto min_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MinFirstLocCustomComparator>(
      label, ex, cbegin(v), cend(v), ::Kokkos::Experimental::move(comp));
}

// ----------------------
// max_element public API
// ----------------------
template <class ExecutionSpace, class IteratorType>
auto max_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last) {
  return Impl::min_or_max_element_impl<MaxFirstLoc>(
      "kokkos_max_element_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
auto max_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last) {
  return Impl::min_or_max_element_impl<MaxFirstLoc>(label, ex, first, last);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto max_element(const ExecutionSpace& ex, IteratorType first,
                 IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MaxFirstLocCustomComparator>(
      "kokkos_max_element_iterator_api_default", ex, first, last,
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto max_element(const std::string& label, const ExecutionSpace& ex,
                 IteratorType first, IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MaxFirstLocCustomComparator>(
      label, ex, first, last, ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto max_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_impl<MaxFirstLoc>(
      "kokkos_max_element_view_api_default", ex, cbegin(v), cend(v));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto max_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::min_or_max_element_impl<MaxFirstLoc>(label, ex, cbegin(v),
                                                    cend(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto max_element(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MaxFirstLocCustomComparator>(
      "kokkos_max_element_view_api_default", ex, cbegin(v), cend(v),
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto max_element(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType, Properties...>& v,
                 ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::min_or_max_element_impl<MaxFirstLocCustomComparator>(
      label, ex, cbegin(v), cend(v), ::Kokkos::Experimental::move(comp));
}

// -------------------------
// minmax_element public API
// -------------------------
template <class ExecutionSpace, class IteratorType>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last) {
  return Impl::minmax_element_impl<MinMaxFirstLastLoc>(
      "kokkos_minmax_element_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  return Impl::minmax_element_impl<MinMaxFirstLastLoc>(label, ex, first, last);
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, ComparatorType comp) {
  static_assert_is_not_opemnptarget(ex);

  return Impl::minmax_element_impl<MinMaxFirstLastLocCustomComparator>(
      "kokkos_minmax_element_iterator_api_default", ex, first, last,
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {
  static_assert(not_openmptarget<ExecutionSpace>::value,
                "minmax_element with custom comparator not currently supported "
                "in OpenMPTarget");

  return Impl::minmax_element_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, first, last, ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_impl<MinMaxFirstLastLoc>(
      "kokkos_minmax_element_view_api_default", ex, cbegin(v), cend(v));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_impl<MinMaxFirstLastLoc>(label, ex, cbegin(v),
                                                       cend(v));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::minmax_element_impl<MinMaxFirstLastLocCustomComparator>(
      "kokkos_minmax_element_view_api_default", ex, cbegin(v), cend(v),
      ::Kokkos::Experimental::move(comp));
}

template <class ExecutionSpace, class DataType, class ComparatorType,
          class... Properties>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  static_assert_is_admissible_to_kokkos_std_algorithms(v);
  static_assert_is_not_opemnptarget(ex);

  return Impl::minmax_element_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, cbegin(v), cend(v), ::Kokkos::Experimental::move(comp));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
