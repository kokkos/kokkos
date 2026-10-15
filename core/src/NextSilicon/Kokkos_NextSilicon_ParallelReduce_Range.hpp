// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce.hpp>
#include <Kokkos_Parallel.hpp>
#include <type_traits>
#include <mutex>

template <class CombinedFunctorReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<CombinedFunctorReducerType,
                                   Kokkos::RangePolicy<Traits...>,
                                   Kokkos::Experimental::NextSilicon> {
  using Policy      = RangePolicy<Traits...>;
  using IndexType   = typename Policy::index_type;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Pointer       = typename ReducerType::pointer_type;
  using ValueType     = typename ReducerType::value_type;
  using ReferenceType = typename ReducerType::reference_type;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  Pointer m_result_ptr;

 public:
  template <class ViewType>
  ParallelReduce(CombinedFunctorReducerType const& functor_reducer,
                 Policy const& policy, ViewType const& result)
      : m_functor_reducer(functor_reducer),
        m_policy(policy),
        m_result_ptr(result.data()) {}

  void execute() const {
    NextSiliconParallelReduceImpl<CombinedFunctorReducerType, Traits...>{
        m_functor_reducer, m_policy, m_result_ptr}
        .execute();
  }
};

#endif  // KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP
