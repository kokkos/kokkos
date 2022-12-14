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

#ifndef KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <std_algorithms/Kokkos_Find.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class IndicatorViewType, class ComparatorType>
struct StdIsSortedUntilFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  IndicatorViewType m_indicator;
  ComparatorType m_comparator;

  KOKKOS_FUNCTION
  void operator()(const index_type i, int& update, const bool final) const {
    const auto& val_i   = m_first[i];
    const auto& val_ip1 = m_first[i + 1];

    if (m_comparator(val_ip1, val_i)) {
      ++update;
    }

    if (final) {
      m_indicator(i) = update;
    }
  }

  KOKKOS_FUNCTION
  StdIsSortedUntilFunctor(IteratorType _first1, IndicatorViewType indicator,
                          ComparatorType comparator)
      : m_first(std::move(_first1)),
        m_indicator(std::move(indicator)),
        m_comparator(std::move(comparator)) {}
};

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last, ComparatorType comp) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  const auto num_elements = Kokkos::Experimental::distance(first, last);

  // trivial case
  if (num_elements <= 1) {
    return last;
  }

  /*
    use scan and a helper "indicator" view
    such that we scan the data and fill the indicator with
    partial sum that is always 0 unless we find a pair that
    breaks the sorting, so in that case the indicator will
    have a 1 starting at the location where the sorting breaks.
    So finding that 1 means finding the location we want.
   */

  // aliases
  using indicator_value_type = std::size_t;
  using indicator_view_type =
      ::Kokkos::View<indicator_value_type*, ExecutionSpace>;
  using functor_type =
      StdIsSortedUntilFunctor<IteratorType, indicator_view_type,
                              ComparatorType>;

  // do scan
  // use num_elements-1 because each index handles i and i+1
  const auto num_elements_minus_one = num_elements - 1;
  indicator_view_type indicator("is_sorted_until_indicator_helper",
                                num_elements_minus_one);
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements_minus_one),
      functor_type(first, indicator, std::move(comp)));

  // try to find the first sentinel value, which indicates
  // where the sorting condition breaks
  namespace KE                                  = ::Kokkos::Experimental;
  constexpr indicator_value_type sentinel_value = 1;
  auto r =
      KE::find(ex, KE::cbegin(indicator), KE::cend(indicator), sentinel_value);
  const auto shift = r - ::Kokkos::Experimental::cbegin(indicator);

  return first + (shift + 1);
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until_impl(const std::string& label,
                                  const ExecutionSpace& ex, IteratorType first,
                                  IteratorType last) {
  using value_type = typename IteratorType::value_type;
  using pred_t     = Impl::StdAlgoLessThanBinaryPredicate<value_type>;
  return is_sorted_until_impl(label, ex, first, last, pred_t());
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
