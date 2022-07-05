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

#ifndef KOKKOS_STD_ALGORITHMS_COPY_IF_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_COPY_IF_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IndexType, class FirstFrom, class FirstDest, class PredType>
struct StdCopyIfFunctor {
  FirstFrom m_first_from;
  FirstDest m_first_dest;
  PredType m_pred;

  KOKKOS_FUNCTION
  StdCopyIfFunctor(FirstFrom first_from, FirstDest first_dest, PredType pred)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_pred(std::move(pred)) {}

  KOKKOS_FUNCTION
  void operator()(const IndexType i, IndexType& update,
                  const bool final_pass) const {
    const auto& myval = m_first_from[i];
    if (final_pass) {
      if (m_pred(myval)) {
        m_first_dest[update] = myval;
      }
    }

    if (m_pred(myval)) {
      update += 1;
    }
  }
};

template <class IndexType, class FirstFrom, class FirstDest, class PredType>
struct StdCopyIfTeamSingleFunctor {
  FirstFrom m_first_from;
  FirstDest m_first_dest;
  PredType m_pred;
  IndexType m_numElements;

  KOKKOS_FUNCTION
  StdCopyIfTeamSingleFunctor(FirstFrom first_from, FirstDest first_dest,
                             PredType pred, IndexType numElements)
      : m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_pred(std::move(pred)),
        m_numElements(numElements) {}

  KOKKOS_FUNCTION void operator()() const {
    int mycount = 0;
    for (IndexType i = 0; i < m_numElements; ++i) {
      const auto& myval = m_first_from[i];
      if (m_pred(myval)) {
        m_first_dest[mycount++] = myval;
      }
    }
  }
};

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class PredicateType>
OutputIterator copy_if_impl(const std::string& label, const ExecutionSpace& ex,
                            InputIterator first, InputIterator last,
                            OutputIterator d_first, PredicateType pred) {
  /*
    To explain the impl, suppose that our data is:

    | 1 | 1 | 2 | 2 | 3 | -2 | 4 | 4 | 4 | 5 | 7 | -10 |

    and we want to copy only the even entries,
    We can use an exclusive scan where the "update"
    is incremented only for the elements that satisfy the predicate.
    This way, the update allows us to track where in the destination
    we need to copy the elements:

    In this case, counting only the even entries, the exlusive scan
    during the final pass would yield:

    | 0 | 0 | 0 | 1 | 2 | 2 | 3 | 4 | 5 | 6 | 6 | 6 |
              *   *       *   *   *   *           *

    which provides the indexing in the destination where
    each starred (*) element needs to be copied to since
    the starred elements are those that satisfy the predicate.
   */

  // checks
  Impl::static_assert_random_access_and_accessible(ex, first, d_first);
  Impl::static_assert_iterators_have_matching_difference_type(first, d_first);
  Impl::expect_valid_range(first, last);

  if (first == last) {
    return d_first;
  } else {
    // aliases
    using index_type = typename InputIterator::difference_type;
    using func_type  = StdCopyIfFunctor<index_type, InputIterator,
                                       OutputIterator, PredicateType>;

    // run
    const auto num_elements = Kokkos::Experimental::distance(first, last);
    index_type count        = 0;
    ::Kokkos::parallel_scan(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_type(first, d_first, pred), count);

    // fence not needed because of the scan accumulating into count
    return d_first + count;
  }
}

template <class TeamHandleType, class InputIterator, class OutputIterator,
          class PredicateType>
KOKKOS_FUNCTION OutputIterator copy_if_team_impl(
    const TeamHandleType& teamHandle, InputIterator first, InputIterator last,
    OutputIterator d_first, PredicateType pred) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first, d_first);
  Impl::static_assert_iterators_have_matching_difference_type(first, d_first);
  Impl::expect_valid_range(first, last);

  if (first == last) {
    return d_first;
  } else {
    // paralle_scan does not yet support TeamThreadRange, so we do this:
    // first, since we return an iterator past the last element copied,
    // we need to compute how many elements satisfy the pred;
    // second, we use Kokkos::single() to copy the elements

    // count elements satisfying the condition
    const auto numElemCounted =
        ::Kokkos::Experimental::count_if(teamHandle, first, last, pred);
    // count_if already calls the team barrier

    // copy elements
    using index_type = typename InputIterator::difference_type;
    using func_type  = StdCopyIfTeamSingleFunctor<index_type, InputIterator,
                                                 OutputIterator, PredicateType>;
    const auto num_elements = Kokkos::Experimental::distance(first, last);
    ::Kokkos::single(PerTeam(teamHandle),
                     func_type(first, d_first, pred, num_elements));
    teamHandle.team_barrier();

    return d_first + numElemCounted;
  }
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
