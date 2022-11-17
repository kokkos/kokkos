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

#ifndef KOKKOS_STD_ALGORITHMS_REVERSE_HPP
#define KOKKOS_STD_ALGORITHMS_REVERSE_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_Reverse.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class InputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value> reverse(
    const ExecutionSpace& ex, InputIterator first, InputIterator last) {
  return Impl::reverse_exespace_impl("Kokkos::reverse_iterator_api_default", ex,
                                     first, last);
}

template <class ExecutionSpace, class InputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value> reverse(
    const std::string& label, const ExecutionSpace& ex, InputIterator first,
    InputIterator last) {
  return Impl::reverse_exespace_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value> reverse(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType, Properties...>& view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::reverse_exespace_impl("Kokkos::reverse_view_api_default", ex,
                                     KE::begin(view), KE::end(view));
}

template <class ExecutionSpace, class DataType, class... Properties>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value> reverse(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType, Properties...>& view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::reverse_exespace_impl(label, ex, KE::begin(view), KE::end(view));
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class InputIterator>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value>
    reverse(const TeamHandleType& teamHandle, InputIterator first,
            InputIterator last) {
  return Impl::reverse_team_impl(teamHandle, first, last);
}

template <class TeamHandleType, class DataType, class... Properties>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value>
    reverse(const TeamHandleType& teamHandle,
            const ::Kokkos::View<DataType, Properties...>& view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::reverse_team_impl(teamHandle, KE::begin(view), KE::end(view));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
