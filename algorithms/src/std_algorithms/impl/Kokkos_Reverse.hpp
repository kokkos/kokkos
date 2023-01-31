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

#ifndef KOKKOS_STD_ALGORITHMS_REVERSE_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_REVERSE_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <std_algorithms/Kokkos_Swap.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class InputIterator>
struct StdReverseFunctor {
  using index_type = typename InputIterator::difference_type;
  static_assert(std::is_signed<index_type>::value,
                "Kokkos: StdReverseFunctor requires signed index type");

  InputIterator m_first;
  InputIterator m_last;

  KOKKOS_FUNCTION
  void operator()(index_type i) const {
    // the swap below is doing the same thing, but
    // for Intel 18.0.5 does not work.
    // But putting the impl directly here, it works.
#ifdef KOKKOS_COMPILER_INTEL
    typename InputIterator::value_type tmp = std::move(m_first[i]);
    m_first[i]                             = std::move(m_last[-i - 1]);
    m_last[-i - 1]                         = std::move(tmp);
#else
    ::Kokkos::Experimental::swap(m_first[i], m_last[-i - 1]);
#endif
  }

  StdReverseFunctor(InputIterator first, InputIterator last)
      : m_first(std::move(first)), m_last(std::move(last)) {}
};

template <class ExecutionSpace, class InputIterator>
void reverse_impl(const std::string& label, const ExecutionSpace& ex,
                  InputIterator first, InputIterator last) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  // aliases
  using func_t = StdReverseFunctor<InputIterator>;

  // run
  if (last >= first + 2) {
    // only need half
    const auto num_elements = Kokkos::Experimental::distance(first, last) / 2;
    ::Kokkos::parallel_for(label,
                           RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                           func_t(first, last));
    ex.fence("Kokkos::reverse: fence after operation");
  }
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
