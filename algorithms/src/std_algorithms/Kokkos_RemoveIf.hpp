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

#ifndef KOKKOS_STD_ALGORITHMS_REMOVE_IF_HPP
#define KOKKOS_STD_ALGORITHMS_REMOVE_IF_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_RemoveAllVariants.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class Iterator, class UnaryPredicate>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, Iterator>
remove_if(const ExecutionSpace& ex, Iterator first, Iterator last,
          UnaryPredicate pred) {
  return Impl::remove_if_exespace_impl("Kokkos::remove_if_iterator_api_default",
                                       ex, first, last, pred);
}

template <class ExecutionSpace, class Iterator, class UnaryPredicate>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, Iterator>
remove_if(const std::string& label, const ExecutionSpace& ex, Iterator first,
          Iterator last, UnaryPredicate pred) {
  return Impl::remove_if_exespace_impl(label, ex, first, last, pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryPredicate,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto remove_if(const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view,
               UnaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::remove_if_exespace_impl("Kokkos::remove_if_iterator_api_default",
                                       ex, ::Kokkos::Experimental::begin(view),
                                       ::Kokkos::Experimental::end(view), pred);
}

template <class ExecutionSpace, class DataType, class... Properties,
          class UnaryPredicate,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto remove_if(const std::string& label, const ExecutionSpace& ex,
               const ::Kokkos::View<DataType, Properties...>& view,
               UnaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  return Impl::remove_if_exespace_impl(label, ex,
                                       ::Kokkos::Experimental::begin(view),
                                       ::Kokkos::Experimental::end(view), pred);
}

//
// overload set accepting a team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <class TeamHandleType, class Iterator, class UnaryPredicate>
KOKKOS_FUNCTION
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, Iterator>
    remove_if(const TeamHandleType& teamHandle, Iterator first, Iterator last,
              UnaryPredicate pred) {
  return Impl::remove_if_team_impl(teamHandle, first, last, pred);
}

template <
    class TeamHandleType, class DataType, class... Properties,
    class UnaryPredicate,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto remove_if(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType, Properties...>& view, UnaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  return Impl::remove_if_team_impl(teamHandle,
                                   ::Kokkos::Experimental::begin(view),
                                   ::Kokkos::Experimental::end(view), pred);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
