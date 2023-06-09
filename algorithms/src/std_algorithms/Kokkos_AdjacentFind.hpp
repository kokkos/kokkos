//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_STD_ALGORITHMS_ADJACENT_FIND_HPP
#define KOKKOS_STD_ALGORITHMS_ADJACENT_FIND_HPP

#include "impl/Kokkos_AdjacentFind.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//

// overload set1
template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const ExecutionSpace& ex, IteratorType first, IteratorType last) {
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const std::string& label, const ExecutionSpace& ex,
              IteratorType first, IteratorType last) {
  return Impl::adjacent_find_exespace_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_view_api_default", ex, KE::begin(v), KE::end(v));
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(label, ex, KE::begin(v), KE::end(v));
}

// overload set2
template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const ExecutionSpace& ex, IteratorType first, IteratorType last,
              BinaryPredicateType pred) {
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_iterator_api_default", ex, first, last, pred);
}

template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 IteratorType>
adjacent_find(const std::string& label, const ExecutionSpace& ex,
              IteratorType first, IteratorType last, BinaryPredicateType pred) {
  return Impl::adjacent_find_exespace_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(
      "Kokkos::adjacent_find_view_api_default", ex, KE::begin(v), KE::end(v),
      pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& v,
                   BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_exespace_impl(label, ex, KE::begin(v), KE::end(v),
                                           pred);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

// overload set1
template <class TeamHandleType, class IteratorType>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, IteratorType>
adjacent_find(const TeamHandleType& teamHandle, IteratorType first,
              IteratorType last) {
  return Impl::adjacent_find_team_impl(teamHandle, first, last);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto adjacent_find(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_team_impl(teamHandle, KE::begin(v), KE::end(v));
}

// overload set2
template <class TeamHandleType, class IteratorType, class BinaryPredicateType>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, IteratorType>
adjacent_find(const TeamHandleType& teamHandle, IteratorType first,
              IteratorType last, BinaryPredicateType pred) {
  return Impl::adjacent_find_team_impl(teamHandle, first, last, pred);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    class BinaryPredicateType,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto adjacent_find(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v,
    BinaryPredicateType pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  namespace KE = ::Kokkos::Experimental;
  return Impl::adjacent_find_team_impl(teamHandle, KE::begin(v), KE::end(v),
                                       pred);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
