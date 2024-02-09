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

#ifndef KOKKOS_SORT_BY_KEY_PUBLIC_API_HPP_
#define KOKKOS_SORT_BY_KEY_PUBLIC_API_HPP_

#include "./impl/Kokkos_SortByKeyImpl.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>

namespace Kokkos::Experimental {

// ---------------------------------------------------------------
// basic overloads
// ---------------------------------------------------------------

template <class ExecutionSpace, class KeysDataType, class... KeysProperties,
          class ValuesDataType, class... ValuesProperties>
void sort_by_key([[maybe_unused]] const ExecutionSpace& exec,
                 Kokkos::View<KeysDataType, KeysProperties...>& keys,
                 Kokkos::View<ValuesDataType, ValuesProperties...>& values) {
  // constraints
  using KeysType   = Kokkos::View<KeysDataType, KeysProperties...>;
  using ValuesType = Kokkos::View<ValuesDataType, ValuesProperties...>;
  ::Kokkos::Impl::static_assert_is_admissible_to_kokkos_sort_by_key(keys);
  ::Kokkos::Impl::static_assert_is_admissible_to_kokkos_sort_by_key(values);

  static_assert(SpaceAccessibility<ExecutionSpace,
                                   typename KeysType::memory_space>::accessible,
                "Kokkos::sort: execution space instance is not able to access "
                "the memory space of the keys View argument!");
  static_assert(
      SpaceAccessibility<ExecutionSpace,
                         typename ValuesType::memory_space>::accessible,
      "Kokkos::sort: execution space instance is not able to access "
      "the memory space of the values View argument!");

  // FIXME: what's the right way to check this condition?
  assert(values.extent(0) >= keys.extent(0));

  if (keys.extent(0) <= 1) {
    return;
  }

  ::Kokkos::Impl::sort_by_key_device_view_without_comparator(exec, keys,
                                                             values);
}

// ---------------------------------------------------------------
// overloads supporting a custom comparator
// ---------------------------------------------------------------

template <class ExecutionSpace, class ComparatorType, class KeysDataType,
          class... KeysProperties, class ValuesDataType,
          class... ValuesProperties>
void sort_by_key([[maybe_unused]] const ExecutionSpace& exec,
                 Kokkos::View<KeysDataType, KeysProperties...>& keys,
                 Kokkos::View<ValuesDataType, ValuesProperties...>& values,
                 const ComparatorType& comparator) {
  // constraints
  using KeysType   = Kokkos::View<KeysDataType, KeysProperties...>;
  using ValuesType = Kokkos::View<ValuesDataType, ValuesProperties...>;
  ::Kokkos::Impl::static_assert_is_admissible_to_kokkos_sort_by_key(keys);
  ::Kokkos::Impl::static_assert_is_admissible_to_kokkos_sort_by_key(values);

  static_assert(SpaceAccessibility<ExecutionSpace,
                                   typename KeysType::memory_space>::accessible,
                "Kokkos::sort: execution space instance is not able to access "
                "the memory space of the keys View argument!");
  static_assert(
      SpaceAccessibility<ExecutionSpace,
                         typename ValuesType::memory_space>::accessible,
      "Kokkos::sort: execution space instance is not able to access "
      "the memory space of the values View argument!");

  // FIXME: what's the right way to check this condition?
  assert(values.extent(0) >= keys.extent(0));

  if (keys.extent(0) <= 1) {
    return;
  }

  ::Kokkos::Impl::sort_by_key_device_view_with_comparator(exec, keys, values,
                                                          comparator);
}

}  // namespace Kokkos::Experimental
#endif
