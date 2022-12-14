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

#ifndef KOKKOS_STD_ALGORITHMS_REPLACE_COPY_IF_HPP
#define KOKKOS_STD_ALGORITHMS_REPLACE_COPY_IF_HPP

#include "impl/Kokkos_ReplaceCopyIf.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class PredicateType, class ValueType>
OutputIterator replace_copy_if(const ExecutionSpace& ex,
                               InputIterator first_from,
                               InputIterator last_from,
                               OutputIterator first_dest, PredicateType pred,
                               const ValueType& new_value) {
  return Impl::replace_copy_if_impl("Kokkos::replace_copy_if_iterator_api", ex,
                                    first_from, last_from, first_dest, pred,
                                    new_value);
}

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class PredicateType, class ValueType>
OutputIterator replace_copy_if(const std::string& label,
                               const ExecutionSpace& ex,
                               InputIterator first_from,
                               InputIterator last_from,
                               OutputIterator first_dest, PredicateType pred,
                               const ValueType& new_value) {
  return Impl::replace_copy_if_impl(label, ex, first_from, last_from,
                                    first_dest, pred, new_value);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class PredicateType,
          class ValueType>
auto replace_copy_if(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType1, Properties1...>& view_from,
                     const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                     PredicateType pred, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_copy_if_impl("Kokkos::replace_copy_if_view_api", ex,
                                    KE::cbegin(view_from), KE::cend(view_from),
                                    KE::begin(view_dest), pred, new_value);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class PredicateType,
          class ValueType>
auto replace_copy_if(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType1, Properties1...>& view_from,
                     const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                     PredicateType pred, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_copy_if_impl(label, ex, KE::cbegin(view_from),
                                    KE::cend(view_from), KE::begin(view_dest),
                                    pred, new_value);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
