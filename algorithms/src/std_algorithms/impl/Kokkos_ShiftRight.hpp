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

#ifndef KOKKOS_STD_ALGORITHMS_SHIFT_RIGHT_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_SHIFT_RIGHT_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Move.hpp>
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ExecutionSpace, class IteratorType>
IteratorType shift_right_exespace_impl(
    const std::string& label, const ExecutionSpace& ex, IteratorType first,
    IteratorType last, typename IteratorType::difference_type n) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);
  KOKKOS_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return first;
  }

  if (n >= Kokkos::Experimental::distance(first, last)) {
    return last;
  }

  /*
    Suppose that n = 3, and [first,last) spans:

    | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 | -3 | 1  | -6 | *
      ^                         				  ^
    first							 last

    shift_right modifies the range such that we have this data:
    |  x | x  | x  | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 | *
                     ^
             return it points here

    and returns an iterator pointing to the new beginning.
    Note: elements marked x are in undefined state because have been moved.

    We implement this in two steps:
    step 1:
      we create a temporary view with extent = distance(first, last-n)
      and *move* assign the elements from [first, last-n) to tmp view, such that
      tmp view becomes:

      | 0  | 1  |  2 | 1  | 2  | 1  | 2  | 2  | 10 |

    step 2:
      move elements of tmp view back to range starting at first+n.
   */

  const auto num_elements_to_move =
      ::Kokkos::Experimental::distance(first, last - n);

  // create tmp view
  using value_type    = typename IteratorType::value_type;
  using tmp_view_type = Kokkos::View<value_type*, ExecutionSpace>;
  tmp_view_type tmp_view("shift_right_impl", num_elements_to_move);
  using tmp_readwrite_iterator_type = decltype(begin(tmp_view));

  using index_type = typename IteratorType::difference_type;

  // step 1
  using step1_func_type =
      StdMoveFunctor<index_type, IteratorType, tmp_readwrite_iterator_type>;
  ::Kokkos::parallel_for(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_to_move),
      step1_func_type(first, begin(tmp_view)));

  // step 2
  using step2_func_type =
      StdMoveFunctor<index_type, tmp_readwrite_iterator_type, IteratorType>;
  ::Kokkos::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, tmp_view.extent(0)),
                         step2_func_type(begin(tmp_view), first + n));

  ex.fence("Kokkos::shift_right: fence after operation");

  return first + n;
}

template <class Iterator>
struct StdShiftRightTeamSingleFunctor {
  Iterator m_first;
  Iterator m_last;
  std::size_t m_shift;

  KOKKOS_FUNCTION
  void operator()() const {
    // the impl function calling this functor guarantees that
    // - m_shift is non-negative
    // - m_first, m_last identify a valid range with m_last > m_first
    // - m_shift is less than m_last - m_first
    // so I can safely use std::size_t here
  }

  KOKKOS_FUNCTION
  StdShiftRightTeamSingleFunctor(Iterator _first, Iterator _last, std::size_t n)
      : m_first(std::move(_first)), m_last(std::move(_last)), m_shift(n) {}
};

template <class TeamHandleType, class IteratorType>
KOKKOS_FUNCTION IteratorType shift_right_team_impl(
    const TeamHandleType& teamHandle, IteratorType first, IteratorType last,
    typename IteratorType::difference_type n) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first);
  Impl::expect_valid_range(first, last);
  KOKKOS_EXPECTS(n >= 0);

  // handle trivial cases
  if (n == 0) {
    return first;
  }

  if (n >= Kokkos::Experimental::distance(first, last)) {
    return last;
  }

  // we cannot use here a new allocation like we do for the
  // execution space impl because for this team impl we are
  // within a parallel region, so for now we solve serially

  const std::size_t numElementsToMove =
      ::Kokkos::Experimental::distance(first, last - n);
  Kokkos::single(Kokkos::PerTeam(teamHandle), [=]() {
    for (std::size_t i = 0; i < numElementsToMove; ++i) {
      last[-i - 1] = std::move(last[-n - i - 1]);
    }
  });
  teamHandle.team_barrier();

  return first + n;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
