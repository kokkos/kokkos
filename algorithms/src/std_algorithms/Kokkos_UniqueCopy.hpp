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

#include "impl/Kokkos_UniqueCopy.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set1: default predicate, accepting execution space
//
template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    std::enable_if_t<Impl::are_iterators_v<InputIterator, OutputIterator> &&
                         is_execution_space_v<ExecutionSpace>,
                     int> = 0>
OutputIterator unique_copy(const ExecutionSpace& ex, InputIterator first,
                           InputIterator last, OutputIterator d_first) {
  return Impl::unique_copy_exespace_impl(
      "Kokkos::unique_copy_iterator_api_default", ex, first, last, d_first);
}

template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    std::enable_if_t<Impl::are_iterators_v<InputIterator, OutputIterator> &&
                         is_execution_space_v<ExecutionSpace>,
                     int> = 0>
OutputIterator unique_copy(const std::string& label, const ExecutionSpace& ex,
                           InputIterator first, InputIterator last,
                           OutputIterator d_first) {
  return Impl::unique_copy_exespace_impl(label, ex, first, last, d_first);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto unique_copy(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_exespace_impl("Kokkos::unique_copy_view_api_default",
                                         ex, cbegin(source), cend(source),
                                         begin(dest));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_exespace_impl(label, ex, cbegin(source),
                                         cend(source), begin(dest));
}

//
// overload set2: custom predicate, accepting execution space
//

template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    typename BinaryPredicate,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator unique_copy(const ExecutionSpace& ex, InputIterator first,
                           InputIterator last, OutputIterator d_first,
                           BinaryPredicate pred) {
  return Impl::unique_copy_exespace_impl(
      "Kokkos::unique_copy_iterator_api_default", ex, first, last, d_first,
      pred);
}

template <
    typename ExecutionSpace, typename InputIterator, typename OutputIterator,
    typename BinaryPredicate,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
OutputIterator unique_copy(const std::string& label, const ExecutionSpace& ex,
                           InputIterator first, InputIterator last,
                           OutputIterator d_first, BinaryPredicate pred) {
  return Impl::unique_copy_exespace_impl(label, ex, first, last, d_first, pred);
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicate,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto unique_copy(const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_exespace_impl("Kokkos::unique_copy_view_api_default",
                                         ex, cbegin(source), cend(source),
                                         begin(dest), std::move(pred));
}

template <
    typename ExecutionSpace, typename DataType1, typename... Properties1,
    typename DataType2, typename... Properties2, typename BinaryPredicate,
    std::enable_if_t<::Kokkos::is_execution_space_v<ExecutionSpace>, int> = 0>
auto unique_copy(const std::string& label, const ExecutionSpace& ex,
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 const ::Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_exespace_impl(
      label, ex, cbegin(source), cend(source), begin(dest), std::move(pred));
}

//
// overload set3: default predicate, accepting team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <
    typename TeamHandleType, typename InputIterator, typename OutputIterator,
    std::enable_if_t<Impl::are_iterators_v<InputIterator, OutputIterator> &&
                         Kokkos::is_team_handle_v<TeamHandleType>,
                     int> = 0>
KOKKOS_FUNCTION OutputIterator unique_copy(const TeamHandleType& teamHandle,
                                           InputIterator first,
                                           InputIterator last,
                                           OutputIterator d_first) {
  return Impl::unique_copy_team_impl(teamHandle, first, last, d_first);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto unique_copy(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source,
    const ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  return Impl::unique_copy_team_impl(teamHandle, cbegin(source), cend(source),
                                     begin(dest));
}

//
// overload set4: custom predicate, accepting team handle
// Note: for now omit the overloads accepting a label
// since they cause issues on device because of the string allocation.
//
template <typename TeamHandleType, typename InputIterator,
          typename OutputIterator, typename BinaryPredicate,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION OutputIterator unique_copy(const TeamHandleType& teamHandle,
                                           InputIterator first,
                                           InputIterator last,
                                           OutputIterator d_first,
                                           BinaryPredicate pred) {
  return Impl::unique_copy_team_impl(teamHandle, first, last, d_first, pred);
}

template <typename TeamHandleType, typename DataType1, typename... Properties1,
          typename DataType2, typename... Properties2, typename BinaryPredicate,
          std::enable_if_t<::Kokkos::is_team_handle_v<TeamHandleType>, int> = 0>
KOKKOS_FUNCTION auto unique_copy(
    const TeamHandleType& teamHandle,
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
