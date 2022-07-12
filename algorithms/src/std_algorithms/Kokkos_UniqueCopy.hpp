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

#ifndef KOKKOS_STD_ALGORITHMS_UNIQUE_COPY_HPP
#define KOKKOS_STD_ALGORITHMS_UNIQUE_COPY_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_UniqueCopy.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

// overload set1, accepting execution space
template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<!::Kokkos::is_view<InputIterator>::value
		 && ::Kokkos::is_execution_space<ExecutionSpace>::value, OutputIterator>
unique_copy(const ExecutionSpace& ex, InputIterator first, InputIterator last,
            OutputIterator d_first) {
  return Impl::unique_copy_impl("Kokkos::unique_copy_iterator_api_default", ex,
                                first, last, d_first);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator>
std::enable_if_t<!::Kokkos::is_view<InputIterator>::value
		 && ::Kokkos::is_execution_space<ExecutionSpace>::value, OutputIterator>
unique_copy(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator last, OutputIterator d_first) {
  return Impl::unique_copy_impl(label, ex, first, last, d_first);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
	  std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value, int> = 0
	  >
auto unique_copy(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return ::Kokkos::Experimental::unique_copy(
      "Kokkos::unique_copy_view_api_default", ex, cbegin(source), cend(source),
      begin(dest));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
	  std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value, int> = 0>
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return ::Kokkos::Experimental::unique_copy(label, ex, cbegin(source),
                                             cend(source), begin(dest));
}

// overload set2, accepting execution space
template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class BinaryPredicate>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, OutputIterator >
unique_copy(const ExecutionSpace& ex, InputIterator first,
                           InputIterator last, OutputIterator d_first,
                           BinaryPredicate pred) {
  return Impl::unique_copy_impl("Kokkos::unique_copy_iterator_api_default", ex,
                                first, last, d_first, pred);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class BinaryPredicate>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value, OutputIterator >
unique_copy(const std::string& label, const ExecutionSpace& ex,
                           InputIterator first, InputIterator last,
                           OutputIterator d_first, BinaryPredicate pred) {
  return Impl::unique_copy_impl(label, ex, first, last, d_first, pred);
}

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class BinaryPredicate,
  std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value, int> = 0
  >
auto unique_copy(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_impl("Kokkos::unique_copy_view_api_default", ex,
                                cbegin(source), cend(source), begin(dest),
                                std::move(pred));
}

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class BinaryPredicate,
  std::enable_if_t< ::Kokkos::is_execution_space<ExecutionSpace>::value, int> = 0
  >
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_impl(label, ex, cbegin(source), cend(source),
                                begin(dest), std::move(pred));
}


// overload set1, accepting team handle, omit the overloads
// accepting a label since they cause
// issues on device because of the string allocation.
template <class TeamHandleType, class InputIterator, class OutputIterator>
KOKKOS_FUNCTION
std::enable_if_t<!::Kokkos::is_view<InputIterator>::value
		 && Impl::is_team_handle<TeamHandleType>::value, OutputIterator>
unique_copy(const TeamHandleType& teamHandle, InputIterator first, InputIterator last,
            OutputIterator d_first) {
  return Impl::unique_copy_team_impl(teamHandle, first, last, d_first);
}

template <class TeamHandleType, class DataType1, class... Properties1,
          class DataType2, class... Properties2,
	  std::enable_if_t< Impl::is_team_handle<TeamHandleType>::value, int> = 0
	  >
KOKKOS_FUNCTION
auto unique_copy(const TeamHandleType& teamHandle,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_team_impl(teamHandle, cbegin(source), cend(source),
				     begin(dest));
}

// overload set2, accepting team handle, omit the overloads
// accepting a label since they cause
// issues on device because of the string allocation.
template <class TeamHandleType, class InputIterator, class OutputIterator,
          class BinaryPredicate>
KOKKOS_FUNCTION
std::enable_if_t<Impl::is_team_handle<TeamHandleType>::value, OutputIterator >
unique_copy(const TeamHandleType& teamHandle, InputIterator first,
                           InputIterator last, OutputIterator d_first,
                           BinaryPredicate pred)
{
  return Impl::unique_copy_team_impl(teamHandle, first, last, d_first, pred);
}

template <
  class TeamHandleType, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class BinaryPredicate,
  std::enable_if_t< Impl::is_team_handle<TeamHandleType>::value, int> = 0
  >
KOKKOS_FUNCTION
auto unique_copy(const TeamHandleType& teamHandle,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_team_impl(teamHandle, cbegin(source), cend(source),
				     begin(dest), std::move(pred));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
