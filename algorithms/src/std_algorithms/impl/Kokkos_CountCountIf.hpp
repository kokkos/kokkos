/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_STD_ALGORITHMS_COUNT_IF_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_COUNT_IF_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IteratorType, class Predicate>
struct StdCountIfFunctor {
  using index_type = typename IteratorType::difference_type;
  IteratorType m_first;
  Predicate m_predicate;

  KOKKOS_FUNCTION
  void operator()(index_type i, index_type& lsum) const {
    if (m_predicate(m_first[i])) {
      lsum++;
    }
  }

  KOKKOS_FUNCTION
  StdCountIfFunctor(IteratorType _first, Predicate _predicate)
      : m_first(std::move(_first)), m_predicate(std::move(_predicate)) {}
};

template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if_impl(const std::string& label,
                                                     const ExecutionSpace& ex,
                                                     IteratorType first,
                                                     IteratorType last,
                                                     Predicate predicate) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first);
  Impl::expect_valid_range(first, last);

  // aliases
  using func_t = StdCountIfFunctor<IteratorType, Predicate>;

  // run
  const auto num_elements = Kokkos::Experimental::distance(first, last);
  typename IteratorType::difference_type count = 0;
  ::Kokkos::parallel_reduce(label,
                            RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                            func_t(first, predicate), count);
  ex.fence("Kokkos::count_if: fence after operation");

  return count;
}

template <class ExecutionSpace, class IteratorType, class T>
auto count_impl(const std::string& label, const ExecutionSpace& ex,
                IteratorType first, IteratorType last, const T& value) {
  return count_if_impl(
      label, ex, first, last,
      ::Kokkos::Experimental::Impl::StdAlgoEqualsValUnaryPredicate<T>(value));
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
