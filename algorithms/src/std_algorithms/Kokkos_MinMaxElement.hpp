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

#ifndef KOKKOS_STD_ALGORITHMS_MINMAX_ELEMENT_HPP
#define KOKKOS_STD_ALGORITHMS_MINMAX_ELEMENT_HPP

#include "impl/Kokkos_MinMaxMinmaxElement.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last) {
  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "Kokkos::minmax_element_iterator_api_default", ex, first, last);
}

template <
    typename ExecutionSpace, typename IteratorType,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last) {
  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(label, ex,
                                                                first, last);
}

template <
    typename ExecutionSpace, typename IteratorType, typename ComparatorType,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex, IteratorType first,
                    IteratorType last, ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "Kokkos::minmax_element_iterator_api_default", ex, first, last,
      std::move(comp));
}

template <
    typename ExecutionSpace, typename IteratorType, typename ComparatorType,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    IteratorType first, IteratorType last,
                    ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, first, last, std::move(comp));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      "Kokkos::minmax_element_view_api_default", ex, begin(v), end(v));
}

template <
    typename ExecutionSpace, typename DataType, typename... Properties,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLoc>(
      label, ex, begin(v), end(v));
}

template <
    typename ExecutionSpace, typename DataType, typename ComparatorType,
    typename... Properties,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      "Kokkos::minmax_element_view_api_default", ex, begin(v), end(v),
      std::move(comp));
}

template <
    typename ExecutionSpace, typename DataType, typename ComparatorType,
    typename... Properties,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto minmax_element(const std::string& label, const ExecutionSpace& ex,
                    const ::Kokkos::View<DataType, Properties...>& v,
                    ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::minmax_element_exespace_impl<MinMaxFirstLastLocCustomComparator>(
      label, ex, begin(v), end(v), std::move(comp));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename IteratorType,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last) {
  return Impl::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle, first,
                                                            last);
}

template <typename TeamHandleType, typename IteratorType,
          typename ComparatorType,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto minmax_element(const TeamHandleType& teamHandle,
                                    IteratorType first, IteratorType last,
                                    ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(teamHandle);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, first, last, std::move(comp));
}

template <typename TeamHandleType, typename DataType, typename... Properties,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLoc>(teamHandle,
                                                            begin(v), end(v));
}

template <typename TeamHandleType, typename DataType, typename ComparatorType,
          typename... Properties,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto minmax_element(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& v, ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(v);
  Impl::static_assert_is_not_openmptarget(teamHandle);

  return Impl::minmax_element_team_impl<MinMaxFirstLastLocCustomComparator>(
      teamHandle, begin(v), end(v), std::move(comp));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
