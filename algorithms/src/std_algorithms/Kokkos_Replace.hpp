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

#ifndef KOKKOS_STD_ALGORITHMS_REPLACE_HPP
#define KOKKOS_STD_ALGORITHMS_REPLACE_HPP

#include "impl/Kokkos_Replace.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

template <class ExecutionSpace, class Iterator, class ValueType>
void replace(const ExecutionSpace& ex, Iterator first, Iterator last,
             const ValueType& old_value, const ValueType& new_value) {
  return Impl::replace_impl("Kokkos::replace_iterator_api", ex, first, last,
                            old_value, new_value);
}

template <class ExecutionSpace, class Iterator, class ValueType>
void replace(const std::string& label, const ExecutionSpace& ex, Iterator first,
             Iterator last, const ValueType& old_value,
             const ValueType& new_value) {
  return Impl::replace_impl(label, ex, first, last, old_value, new_value);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class ValueType>
void replace(const ExecutionSpace& ex,
             const ::Kokkos::View<DataType1, Properties1...>& view,
             const ValueType& old_value, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_impl("Kokkos::replace_view_api", ex, KE::begin(view),
                            KE::end(view), old_value, new_value);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class ValueType>
void replace(const std::string& label, const ExecutionSpace& ex,
             const ::Kokkos::View<DataType1, Properties1...>& view,
             const ValueType& old_value, const ValueType& new_value) {
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view);
  namespace KE = ::Kokkos::Experimental;
  return Impl::replace_impl(label, ex, KE::begin(view), KE::end(view),
                            old_value, new_value);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
