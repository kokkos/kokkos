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

#ifndef KOKKOS_STD_ALGORITHMS_TRANSFORM_EXCLUSIVE_SCAN_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_TRANSFORM_EXCLUSIVE_SCAN_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include "Kokkos_ValueWrapperForNoNeutralElement.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest, class BinaryOpType, class UnaryOpType>
struct TransformExclusiveScanFunctor {
  using execution_space = ExeSpace;
  using value_type =
      ::Kokkos::Experimental::Impl::ValueWrapperForNoNeutralElement<ValueType>;

  ValueType m_init_value;
  FirstFrom m_first_from;
  FirstDest m_first_dest;
  BinaryOpType m_binary_op;
  UnaryOpType m_unary_op;

  KOKKOS_FUNCTION
  TransformExclusiveScanFunctor(ValueType init, FirstFrom first_from,
                                FirstDest first_dest, BinaryOpType bop,
                                UnaryOpType uop)
      : m_init_value(std::move(init)),
        m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)),
        m_binary_op(std::move(bop)),
        m_unary_op(std::move(uop)) {}

  KOKKOS_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    if (final_pass) {
      if (i == 0) {
        // for both ExclusiveScan and TransformExclusiveScan,
        // init is unmodified
        m_first_dest[i] = m_init_value;
      } else {
        m_first_dest[i] = m_binary_op(update.val, m_init_value);
      }
    }

    const auto tmp = value_type{m_unary_op(m_first_from[i]), false};
    this->join(update, tmp);
  }

  KOKKOS_FUNCTION
  void init(value_type& update) const {
    update.val        = {};
    update.is_initial = true;
  }

  KOKKOS_FUNCTION
  void join(value_type& update, const value_type& input) const {
    if (update.is_initial) {
      update.val = input.val;
    } else {
      update.val = m_binary_op(update.val, input.val);
    }
    update.is_initial = false;
  }
};

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType,
          class UnaryOpType>
OutputIteratorType transform_exclusive_scan_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType first_from, InputIteratorType last_from,
    OutputIteratorType first_dest, ValueType init_value, BinaryOpType bop,
    UnaryOpType uop) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using func_type =
      TransformExclusiveScanFunctor<ExecutionSpace, index_type, ValueType,
                                    InputIteratorType, OutputIteratorType,
                                    BinaryOpType, UnaryOpType>;

  // run
  const auto num_elements =
      Kokkos::Experimental::distance(first_from, last_from);
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(init_value, first_from, first_dest, bop, uop));
  ex.fence("Kokkos::transform_exclusive_scan: fence after operation");

  // return
  return first_dest + num_elements;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
