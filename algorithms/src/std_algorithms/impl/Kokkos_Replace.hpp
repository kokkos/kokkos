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

#ifndef KOKKOS_STD_ALGORITHMS_REPLACE_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_REPLACE_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include "Kokkos_IsTeamHandle.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class InputIterator, class ValueType>
struct StdReplaceFunctor {
  using index_type = typename InputIterator::difference_type;
  InputIterator m_first;
  ValueType m_old_value;
  ValueType m_new_value;

  KOKKOS_FUNCTION
  void operator()(index_type i) const {
    if (m_first[i] == m_old_value) {
      m_first[i] = m_new_value;
    }
  }

  KOKKOS_FUNCTION
  StdReplaceFunctor(InputIterator first, ValueType old_value,
                    ValueType new_value)
      : m_first(std::move(first)),
        m_old_value(std::move(old_value)),
        m_new_value(std::move(new_value)) {}
};

template <class ExeSpaceOrTeamHandle, class IteratorType, class ValueType,
          class... Args>
KOKKOS_FUNCTION void replace_impl(const ExeSpaceOrTeamHandle& ExOrTh,
                                  IteratorType first, IteratorType last,
                                  const ValueType& old_value,
                                  const ValueType& new_value,
                                  Args&&... labelOrEmpty) {
  // checks
  Impl::static_assert_random_access_and_accessible(ExOrTh, first);
  Impl::expect_valid_range(first, last);

  // aliases
  using func_t = StdReplaceFunctor<IteratorType, ValueType>;

  // run
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  if constexpr (is_team_handle<ExeSpaceOrTeamHandle>::value) {
    static_assert(sizeof...(Args) == 0);
    ::Kokkos::parallel_for(TeamThreadRange(ExOrTh, 0, num_elements),
                           func_t(first, old_value, new_value));
    ExOrTh.team_barrier();
  } else {
    ::Kokkos::parallel_for(
        std::forward<Args>(labelOrEmpty)...,
        RangePolicy<ExeSpaceOrTeamHandle>(ExOrTh, 0, num_elements),
        func_t(first, old_value, new_value));
    ExOrTh.fence("Kokkos::replace: fence after operation");
  }
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
