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

#ifndef KOKKOS_STD_ALGORITHMS_COPY_N_HPP
#define KOKKOS_STD_ALGORITHMS_COPY_N_HPP

#include "./impl/Kokkos_IsTeamHandle.hpp"
#include "impl/Kokkos_CopyCopyN.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

//
// overload set accepting execution space
//
template <class ExecutionSpace, class InputIterator, class Size,
          class OutputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 OutputIterator>
copy_n(const ExecutionSpace& ex, InputIterator first, Size count,
       OutputIterator result) {
  return Impl::copy_n_exespace_impl("Kokkos::copy_n_iterator_api_default", ex,
                                    first, count, result);
}

template <class ExecutionSpace, class InputIterator, class Size,
          class OutputIterator>
std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                 OutputIterator>
copy_n(const std::string& label, const ExecutionSpace& ex, InputIterator first,
       Size count, OutputIterator result) {
  return Impl::copy_n_exespace_impl(label, ex, first, count, result);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class Size, class DataType2, class... Properties2,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto copy_n(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& source, Size count,
            ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  namespace KE = ::Kokkos::Experimental;
  return Impl::copy_n_exespace_impl("Kokkos::copy_n_view_api_default", ex,
                                    KE::cbegin(source), count, KE::begin(dest));
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class Size, class DataType2, class... Properties2,
          std::enable_if_t<::Kokkos::is_execution_space<ExecutionSpace>::value,
                           int> = 0>
auto copy_n(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& source, Size count,
            ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  namespace KE = ::Kokkos::Experimental;
  return Impl::copy_n_exespace_impl(label, ex, KE::cbegin(source), count,
                                    KE::begin(dest));
}

//
// overload set accepting team handle
//
template <class TeamHandleType, class InputIterator, class Size,
          class OutputIterator>
KOKKOS_FUNCTION std::enable_if_t<
    ::Kokkos::is_team_handle<TeamHandleType>::value, OutputIterator>
copy_n(const TeamHandleType& teamHandle, InputIterator first, Size count,
       OutputIterator result) {
  return Impl::copy_n_team_impl(teamHandle, first, count, result);
}

template <
    class TeamHandleType, class DataType1, class... Properties1, class Size,
    class DataType2, class... Properties2,
    std::enable_if_t<::Kokkos::is_team_handle<TeamHandleType>::value, int> = 0>
KOKKOS_FUNCTION auto copy_n(
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source, Size count,
    ::Kokkos::View<DataType2, Properties2...>& dest) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(source);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(dest);

  namespace KE = ::Kokkos::Experimental;
  return Impl::copy_n_team_impl(teamHandle, KE::cbegin(source), count,
                                KE::begin(dest));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
