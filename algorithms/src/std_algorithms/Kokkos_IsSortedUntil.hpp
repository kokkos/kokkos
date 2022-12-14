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

#ifndef KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_HPP
#define KOKKOS_STD_ALGORITHMS_IS_SORTED_UNTIL_HPP

#include "impl/Kokkos_IsSortedUntil.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last) {
  return Impl::is_sorted_until_impl(
      "Kokkos::is_sorted_until_iterator_api_default", ex, first, last);
}

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last) {
  return Impl::is_sorted_until_impl(label, ex, first, last);
}

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl("Kokkos::is_sorted_until_view_api_default",
                                    ex, KE::begin(view), KE::end(view));
}

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl(label, ex, KE::begin(view), KE::end(view));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const ExecutionSpace& ex, IteratorType first,
                             IteratorType last, ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);
  return Impl::is_sorted_until_impl(
      "Kokkos::is_sorted_until_iterator_api_default", ex, first, last,
      std::move(comp));
}

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                             IteratorType first, IteratorType last,
                             ComparatorType comp) {
  Impl::static_assert_is_not_openmptarget(ex);

  return Impl::is_sorted_until_impl(label, ex, first, last, std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
auto is_sorted_until(const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_not_openmptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl("Kokkos::is_sorted_until_view_api_default",
                                    ex, KE::begin(view), KE::end(view),
                                    std::move(comp));
}

template <class ExecutionSpace, class DataType, class... Properties,
          class ComparatorType>
auto is_sorted_until(const std::string& label, const ExecutionSpace& ex,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  Impl::static_assert_is_not_openmptarget(ex);

  namespace KE = ::Kokkos::Experimental;
  return Impl::is_sorted_until_impl(label, ex, KE::begin(view), KE::end(view),
                                    std::move(comp));
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
