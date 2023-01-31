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

#ifndef KOKKOS_STD_ALGORITHMS_TRASFORM_EXCLUSIVE_SCAN_HPP
#define KOKKOS_STD_ALGORITHMS_TRASFORM_EXCLUSIVE_SCAN_HPP

#include "impl/Kokkos_TransformExclusiveScan.hpp"
#include "Kokkos_BeginEnd.hpp"

namespace Kokkos {
namespace Experimental {

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType,
          class UnaryOpType>
std::enable_if_t<::Kokkos::Experimental::Impl::are_iterators<
                     InputIteratorType, OutputIteratorType>::value,
                 OutputIteratorType>
transform_exclusive_scan(const ExecutionSpace& ex, InputIteratorType first,
                         InputIteratorType last, OutputIteratorType first_dest,
                         ValueType init_value, BinaryOpType binary_op,
                         UnaryOpType unary_op) {
  Impl::static_assert_is_not_openmptarget(ex);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  return Impl::transform_exclusive_scan_impl(
      "Kokkos::transform_exclusive_scan_custom_functors_iterator_api", ex,
      first, last, first_dest, init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType,
          class UnaryOpType>
std::enable_if_t<::Kokkos::Experimental::Impl::are_iterators<
                     InputIteratorType, OutputIteratorType>::value,
                 OutputIteratorType>
transform_exclusive_scan(const std::string& label, const ExecutionSpace& ex,
                         InputIteratorType first, InputIteratorType last,
                         OutputIteratorType first_dest, ValueType init_value,
                         BinaryOpType binary_op, UnaryOpType unary_op) {
  Impl::static_assert_is_not_openmptarget(ex);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  return Impl::transform_exclusive_scan_impl(label, ex, first, last, first_dest,
                                             init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
    ValueType init_value, BinaryOpType binary_op, UnaryOpType unary_op) {
  Impl::static_assert_is_not_openmptarget(ex);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  namespace KE = ::Kokkos::Experimental;
  return Impl::transform_exclusive_scan_impl(
      "Kokkos::transform_exclusive_scan_custom_functors_view_api", ex,
      KE::cbegin(view_from), KE::cend(view_from), KE::begin(view_dest),
      init_value, binary_op, unary_op);
}

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ValueType,
          class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view_from,
    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
    ValueType init_value, BinaryOpType binary_op, UnaryOpType unary_op) {
  Impl::static_assert_is_not_openmptarget(ex);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_from);
  Impl::static_assert_is_admissible_to_kokkos_std_algorithms(view_dest);
  static_assert(std::is_move_constructible<ValueType>::value,
                "ValueType must be move constructible.");
  namespace KE = ::Kokkos::Experimental;
  return Impl::transform_exclusive_scan_impl(
      label, ex, KE::cbegin(view_from), KE::cend(view_from),
      KE::begin(view_dest), init_value, binary_op, unary_op);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
