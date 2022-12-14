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

#ifndef KOKKOS_STD_ALGORITHMS_TRANSFORM_IMPL_HPP
#define KOKKOS_STD_ALGORITHMS_TRANSFORM_IMPL_HPP

#include <Kokkos_Core.hpp>
#include "Kokkos_Constraints.hpp"
#include "Kokkos_HelperPredicates.hpp"
#include <std_algorithms/Kokkos_Distance.hpp>
#include <string>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class IndexType, class InputIterator, class OutputIterator,
          class UnaryFunctorType>
struct StdTransformFunctor {
  InputIterator m_first;
  OutputIterator m_d_first;
  UnaryFunctorType m_unary_op;

  KOKKOS_FUNCTION
  void operator()(IndexType i) const { m_d_first[i] = m_unary_op(m_first[i]); }

  KOKKOS_FUNCTION
  StdTransformFunctor(InputIterator _first, OutputIterator _m_d_first,
                      UnaryFunctorType _functor)
      : m_first(std::move(_first)),
        m_d_first(std::move(_m_d_first)),
        m_unary_op(std::move(_functor)) {}
};

template <class IndexType, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryFunctorType>
struct StdTransformBinaryFunctor {
  InputIterator1 m_first1;
  InputIterator2 m_first2;
  OutputIterator m_d_first;
  BinaryFunctorType m_binary_op;

  KOKKOS_FUNCTION
  void operator()(IndexType i) const {
    m_d_first[i] = m_binary_op(m_first1[i], m_first2[i]);
  }

  KOKKOS_FUNCTION
  StdTransformBinaryFunctor(InputIterator1 _first1, InputIterator2 _first2,
                            OutputIterator _m_d_first,
                            BinaryFunctorType _functor)
      : m_first1(std::move(_first1)),
        m_first2(std::move(_first2)),
        m_d_first(std::move(_m_d_first)),
        m_binary_op(std::move(_functor)) {}
};

template <class ExecutionSpace, class InputIterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform_impl(const std::string& label,
                              const ExecutionSpace& ex, InputIterator first1,
                              InputIterator last1, OutputIterator d_first,
                              UnaryOperation unary_op) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first1, d_first);
  Impl::static_assert_iterators_have_matching_difference_type(first1, d_first);
  Impl::expect_valid_range(first1, last1);

  // aliases
  using index_type = typename InputIterator::difference_type;
  using func_t = StdTransformFunctor<index_type, InputIterator, OutputIterator,
                                     UnaryOperation>;

  // run
  const auto num_elements = Kokkos::Experimental::distance(first1, last1);
  ::Kokkos::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         func_t(first1, d_first, unary_op));
  ex.fence("Kokkos::transform: fence after operation");

  // return
  return d_first + num_elements;
}

template <class ExecutionSpace, class InputIterator1, class InputIterator2,
          class OutputIterator, class BinaryOperation>
OutputIterator transform_impl(const std::string& label,
                              const ExecutionSpace& ex, InputIterator1 first1,
                              InputIterator1 last1, InputIterator2 first2,
                              OutputIterator d_first,
                              BinaryOperation binary_op) {
  // checks
  Impl::static_assert_random_access_and_accessible(ex, first1, first2, d_first);
  Impl::static_assert_iterators_have_matching_difference_type(first1, first2,
                                                              d_first);
  Impl::expect_valid_range(first1, last1);

  // aliases
  using index_type = typename InputIterator1::difference_type;
  using func_t =
      StdTransformBinaryFunctor<index_type, InputIterator1, InputIterator2,
                                OutputIterator, BinaryOperation>;

  // run
  const auto num_elements = Kokkos::Experimental::distance(first1, last1);
  ::Kokkos::parallel_for(label,
                         RangePolicy<ExecutionSpace>(ex, 0, num_elements),
                         func_t(first1, first2, d_first, binary_op));
  ex.fence("Kokkos::transform: fence after operation");
  return d_first + num_elements;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
