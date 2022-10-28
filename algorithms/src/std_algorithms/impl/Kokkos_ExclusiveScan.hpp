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

#ifndef KOKKOS_STD_ALGORITHMS_EXCLUSIVE_SCAN_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_EXCLUSIVE_SCAN_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include "Kokkos_ValueWrapperForNoNeutralElement.hpp"
#include "Kokkos_IdentityReferenceUnaryFunctor.hpp"
#include <std_algorithms/Kokkos_TransformExclusiveScan.hpp>
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest>
struct ExclusiveScanDefaultFunctorForKnownNeutralElement {
  using execution_space = ExeSpace;

  ValueType m_init_value;
  FirstFrom m_first_from;
  FirstDest m_first_dest;

  KOKKOS_FUNCTION
  ExclusiveScanDefaultFunctorForKnownNeutralElement(ValueType init,
                                                    FirstFrom first_from,
                                                    FirstDest first_dest)
      : m_init_value(std::move(init)),
        m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)) {}

  KOKKOS_FUNCTION
  void operator()(const IndexType i, ValueType& update,
                  const bool final_pass) const {
    if (final_pass) m_first_dest[i] = update + m_init_value;
    update += m_first_from[i];
  }
};

template <class ExeSpace, class IndexType, class ValueType, class FirstFrom,
          class FirstDest>
struct ExclusiveScanDefaultFunctor {
  using execution_space = ExeSpace;
  using value_type =
      ::Kokkos::Experimental::Impl::ValueWrapperForNoNeutralElement<ValueType>;

  ValueType m_init_value;
  FirstFrom m_first_from;
  FirstDest m_first_dest;

  KOKKOS_FUNCTION
  ExclusiveScanDefaultFunctor(ValueType init, FirstFrom first_from,
                              FirstDest first_dest)
      : m_init_value(std::move(init)),
        m_first_from(std::move(first_from)),
        m_first_dest(std::move(first_dest)) {}

  KOKKOS_FUNCTION
  void operator()(const IndexType i, value_type& update,
                  const bool final_pass) const {
    if (final_pass) {
      if (i == 0) {
        m_first_dest[i] = m_init_value;
      } else {
        m_first_dest[i] = update.val + m_init_value;
      }
    }

    const auto tmp = value_type{m_first_from[i], false};
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
      update.val        = input.val;
      update.is_initial = false;
    } else {
      update.val = update.val + input.val;
    }
  }
};

//
// exespace impl
//
template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType>
OutputIteratorType exclusive_scan_custom_op_exespace_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType first_from, InputIteratorType last_from,
    OutputIteratorType first_dest, ValueType init_value, BinaryOpType bop) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  // aliases
  using unary_op_type = StdNumericScanIdentityReferenceUnaryFunctor<ValueType>;
  using func_type     = ExeSpaceTransformExclusiveScanFunctor<
      ExecutionSpace, ValueType, InputIteratorType, OutputIteratorType,
      BinaryOpType, unary_op_type>;

  // run
  const auto num_elements =
      Kokkos::Experimental::distance(first_from, last_from);
  ::Kokkos::parallel_scan(
      label, RangePolicy<ExecutionSpace>(ex, 0, num_elements),
      func_type(init_value, first_from, first_dest, bop, unary_op_type()));
  ex.fence("Kokkos::exclusive_scan_custom_op: fence after operation");

  // return
  return first_dest + num_elements;
}

template <typename ValueType>
using ex_scan_has_reduction_identity_sum_t =
    decltype(Kokkos::reduction_identity<ValueType>::sum());

template <class ExecutionSpace, class InputIteratorType,
          class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan_default_op_exespace_impl(
    const std::string& label, const ExecutionSpace& ex,
    InputIteratorType first_from, InputIteratorType last_from,
    OutputIteratorType first_dest, ValueType init_value) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first_from, first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  // does it make sense to do this static_assert too?
  // using input_iterator_value_type = typename InputIteratorType::value_type;
  // static_assert
  //   (std::is_convertible<std::remove_cv_t<input_iterator_value_type>,
  //   ValueType>::value,
  //    "exclusive_scan: InputIteratorType::value_type not convertible to
  //    ValueType");

  // we are unnecessarily duplicating code, but this is on purpose
  // so that we can use the default_op for OpenMPTarget.
  // Originally, I had this implemented as:
  // '''
  // using bop_type   = StdExclusiveScanDefaultJoinFunctor<ValueType>;
  // call exclusive_scan_custom_op_impl(..., bop_type());
  // '''
  // which avoids duplicating the functors, but for OpenMPTarget
  // I cannot use a custom binary op.
  // This is the same problem that occurs for reductions.

  // aliases
  using index_type = typename InputIteratorType::difference_type;
  using func_type  = std::conditional_t<
      ::Kokkos::is_detected<ex_scan_has_reduction_identity_sum_t,
                            ValueType>::value,
      ExclusiveScanDefaultFunctorForKnownNeutralElement<
          ExecutionSpace, index_type, ValueType, InputIteratorType,
          OutputIteratorType>,
      ExclusiveScanDefaultFunctor<ExecutionSpace, index_type, ValueType,
                                  InputIteratorType, OutputIteratorType>>;

  // run
  const auto num_elements =
      Kokkos::Experimental::distance(first_from, last_from);
  ::Kokkos::parallel_scan(label,
                          RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                          func_type(init_value, first_from, first_dest));

  ex.fence("Kokkos::exclusive_scan_default_op: fence after operation");

  return first_dest + num_elements;
}

//
// team impl
//
template <class TeamHandleType, class InputIteratorType,
          class OutputIteratorType, class ValueType, class BinaryOpType>
KOKKOS_FUNCTION OutputIteratorType exclusive_scan_custom_op_team_impl(
    const TeamHandleType& teamHandle, InputIteratorType first_from,
    InputIteratorType last_from, OutputIteratorType first_dest,
    ValueType init_value, BinaryOpType bop) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first_from,
                                                   first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  static_assert(
      ::Kokkos::is_detected_v<ex_scan_has_reduction_identity_sum_t, ValueType>,
      "At the moment exclusive_scan doesn't support types without reduction "
      "identity");

  // #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) ||
  // defined(KOKKOS_ENABLE_SYCL)

  const auto num_elements =
      Kokkos::Experimental::distance(first_from, last_from);

  // aliases
  using exe_space     = typename TeamHandleType::execution_space;
  using unary_op_type = StdNumericScanIdentityReferenceUnaryFunctor<ValueType>;
  using func_type =
      TeamTransformExclusiveScanFunctor<exe_space, ValueType, InputIteratorType,
                                        OutputIteratorType, BinaryOpType,
                                        unary_op_type>;

  ::Kokkos::parallel_scan(
      TeamThreadRange(teamHandle, 0, num_elements),
      func_type(init_value, first_from, first_dest, bop, unary_op_type()));
  teamHandle.team_barrier();

  return first_dest + num_elements;

  // #else

  //   std::size_t count = 0;
  //   if (teamHandle.team_rank() == 0) {
  //     while (first_from != last_from) {
  //       const auto val = init_value;
  //       init_value     = bop(init_value, *first_from);
  //       ++first_from;
  //       first_dest[count++] = val;
  //     }
  //   }

  //   teamHandle.team_broadcast(count, 0);
  //   return first_dest + count;

  // #endif
}

template <typename ValueType>
using ex_scan_has_reduction_identity_sum_t =
    decltype(Kokkos::reduction_identity<ValueType>::sum());

template <class TeamHandleType, class InputIteratorType,
          class OutputIteratorType, class ValueType>
KOKKOS_FUNCTION OutputIteratorType exclusive_scan_default_op_team_impl(
    const TeamHandleType& teamHandle, InputIteratorType first_from,
    InputIteratorType last_from, OutputIteratorType first_dest,
    ValueType init_value) {
  // checks
  Impl::static_assert_random_access_and_accessible(teamHandle, first_from,
                                                   first_dest);
  Impl::static_assert_iterators_have_matching_difference_type(first_from,
                                                              first_dest);
  Impl::expect_valid_range(first_from, last_from);

  // #if defined(KOKKOS_ENABLE_CUDA)

  // aliases
  using exe_space  = typename TeamHandleType::execution_space;
  using index_type = typename InputIteratorType::difference_type;
  using func_type  = std::conditional_t<
      ::Kokkos::is_detected_v<ex_scan_has_reduction_identity_sum_t, ValueType>,
      ExclusiveScanDefaultFunctorForKnownNeutralElement<
          exe_space, index_type, ValueType, InputIteratorType,
          OutputIteratorType>,
      ExclusiveScanDefaultFunctor<exe_space, index_type, ValueType,
                                  InputIteratorType, OutputIteratorType>>;

  const auto num_elements =
      Kokkos::Experimental::distance(first_from, last_from);

  ::Kokkos::parallel_scan(TeamThreadRange(teamHandle, 0, num_elements),
                          func_type(init_value, first_from, first_dest));
  teamHandle.team_barrier();
  return first_dest + num_elements;

  // #else

  //   return exclusive_scan_custom_op_team_impl(
  //       teamHandle, first_from, last_from, first_dest, init_value,
  //       [](const ValueType& lhs, const ValueType& rhs) { return lhs + rhs;
  //       });

  // #endif
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
