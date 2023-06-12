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

#ifndef KOKKOS_STD_ALGORITHMS_REDUCE_HPP
#define KOKKOS_STD_ALGORITHMS_REDUCE_HPP

#include "impl/Kokkos_Reduce.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//

//
// overload set 1
//
template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 typename IteratorType::value_type>
reduce(const ExecutionSpace& ex, IteratorType first, IteratorType last) {
  return Impl::reduce_default_functors_exespace_impl(
      "Kokkos::reduce_default_functors_iterator_api", ex, first, last,
      typename IteratorType::value_type());
}

template <class ExecutionSpace, class IteratorType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 typename IteratorType::value_type>
reduce(const std::string& label, const ExecutionSpace& ex, IteratorType first,
       IteratorType last) {
  return Impl::reduce_default_functors_exespace_impl(
      label, ex, first, last, typename IteratorType::value_type());
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto reduce(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  using view_type  = ::Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::value_type;

  return Impl::reduce_default_functors_exespace_impl(
      "Kokkos::reduce_default_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), value_type());
}

template <class ExecutionSpace, class DataType, class... Properties,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto reduce(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  using view_type  = ::Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::value_type;

  return Impl::reduce_default_functors_exespace_impl(
      label, ex, KE::cbegin(view), KE::cend(view), value_type());
}

//
// overload set2:
//
template <class ExecutionSpace, class IteratorType, class ValueType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const ExecutionSpace& ex, IteratorType first, IteratorType last,
       ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_default_functors_exespace_impl(
      "Kokkos::reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value);
}

template <class ExecutionSpace, class IteratorType, class ValueType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const std::string& label, const ExecutionSpace& ex, IteratorType first,
       IteratorType last, ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_default_functors_exespace_impl(label, ex, first, last,
                                                     init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const ExecutionSpace& ex,
       const ::Kokkos::View<DataType, Properties...>& view,
       ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_default_functors_exespace_impl(
      "Kokkos::reduce_default_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), init_reduction_value);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const std::string& label, const ExecutionSpace& ex,
       const ::Kokkos::View<DataType, Properties...>& view,
       ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_default_functors_exespace_impl(
      label, ex, KE::cbegin(view), KE::cend(view), init_reduction_value);
}

//
// overload set 3
//
template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const ExecutionSpace& ex, IteratorType first, IteratorType last,
       ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_custom_functors_exespace_impl(
      "Kokkos::reduce_default_functors_iterator_api", ex, first, last,
      init_reduction_value, joiner);
}

template <class ExecutionSpace, class IteratorType, class ValueType,
          class BinaryOp>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const std::string& label, const ExecutionSpace& ex, IteratorType first,
       IteratorType last, ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_custom_functors_exespace_impl(
      label, ex, first, last, init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const ExecutionSpace& ex,
       const ::Kokkos::View<DataType, Properties...>& view,
       ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_custom_functors_exespace_impl(
      "Kokkos::reduce_custom_functors_view_api", ex, KE::cbegin(view),
      KE::cend(view), init_reduction_value, joiner);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ValueType, class BinaryOp>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, ValueType>
reduce(const std::string& label, const ExecutionSpace& ex,
       const ::Kokkos::View<DataType, Properties...>& view,
       ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_custom_functors_exespace_impl(
      label, ex, KE::cbegin(view), KE::cend(view), init_reduction_value,
      joiner);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//

//
// overload set 1
//
template <class TeamHandleType, class IteratorType>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value,
                     typename IteratorType::value_type>
    reduce(const TeamHandleType& teamHandle, IteratorType first,
           IteratorType last) {
  return Impl::reduce_default_functors_team_impl(
      teamHandle, first, last, typename IteratorType::value_type());
}

template <
    class TeamHandleType, class DataType, class... Properties,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto reduce(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& view) {
  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  using view_type  = ::Kokkos::View<DataType, Properties...>;
  using value_type = typename view_type::value_type;

  return Impl::reduce_default_functors_team_impl(teamHandle, KE::cbegin(view),
                                                 KE::cend(view), value_type());
}

//
// overload set2:
//
template <class TeamHandleType, class IteratorType, class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, ValueType>
    reduce(const TeamHandleType& teamHandle, IteratorType first,
           IteratorType last, ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_default_functors_team_impl(teamHandle, first, last,
                                                 init_reduction_value);
}

template <class TeamHandleType, class DataType, class... Properties,
          class ValueType>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, ValueType>
    reduce(const TeamHandleType& teamHandle,
           const ::Kokkos::View<DataType, Properties...>& view,
           ValueType init_reduction_value) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_default_functors_team_impl(
      teamHandle, KE::cbegin(view), KE::cend(view), init_reduction_value);
}

//
// overload set 3
//
template <class TeamHandleType, class IteratorType, class ValueType,
          class BinaryOp>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, ValueType>
    reduce(const TeamHandleType& teamHandle, IteratorType first,
           IteratorType last, ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  return Impl::reduce_custom_functors_team_impl(teamHandle, first, last,
                                                init_reduction_value, joiner);
}

template <class TeamHandleType, class DataType, class... Properties,
          class ValueType, class BinaryOp>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, ValueType>
    reduce(const TeamHandleType& teamHandle,
           const ::Kokkos::View<DataType, Properties...>& view,
           ValueType init_reduction_value, BinaryOp joiner) {
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");

  namespace KE = ::Kokkos::Experimental;
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::reduce_custom_functors_team_impl(teamHandle, KE::cbegin(view),
                                                KE::cend(view),
                                                init_reduction_value, joiner);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
