// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Single.hpp
/// \brief Default implementation of the single operator
/// Can be specialized by the different backends

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_SINGLE_DEFAULT_IMPL_HPP
#define KOKKOS_SINGLE_DEFAULT_IMPL_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel_Reduce.hpp>

#include <impl/Kokkos_BuiltinReducers.hpp>
#include <impl/Kokkos_UnbindIndexArgument.hpp>

namespace Kokkos::Impl {

// This serves as a wrapper for the value inside single in order to have a
// neutral element for the allocation done in the NoOpReducer
template <class Content>
struct SingleInternalValueWrapper {
  Content content;
  bool init;
};

// Reducer performing no operation used within single to return a value without
// imposing the value's type to have any operation defined.
template <class Scalar, class Space>
struct NoOpReducer : Impl::BaseReducer<Scalar, Space> {
 private:
  using parent_type = Impl::BaseReducer<Scalar, Space>;

 public:
  using reducer    = NoOpReducer<Scalar, Space>;
  using value_type = typename parent_type::value_type;

  // Inherit constructors
  using parent_type::parent_type;

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    if (src.init) {
      dest = src;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const { val.init = false; }
};

template <class FunctorType>
struct SingleReductionWrapper {
  FunctorType m_functor;
  template <class WorkTag, class Index, class ReturnType>
  KOKKOS_INLINE_FUNCTION void operator()(
      WorkTag&& worktag, Index&&,
      SingleInternalValueWrapper<ReturnType>& ret) const {
    static_assert(std::is_empty_v<std::remove_cvref_t<WorkTag>>);
    m_functor(worktag, ret.content);
    ret.init = true;
  }

  template <class Index, class ReturnType>
  KOKKOS_INLINE_FUNCTION void operator()(
      Index&&, SingleInternalValueWrapper<ReturnType>& ret) const {
    m_functor(ret.content);
    ret.init = true;
  }
};

// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace>
struct Single {
  // Single without return value
  template <class FunctorType, class SinglePolicy>
  static void execute(const FunctorType& functor,
                      const SinglePolicy& single_policy) {
    using base_class = typename std::remove_cvref_t<SinglePolicy>::range_policy;
    const base_class& range_policy = single_policy.impl_get_range_policy();
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelFor<FunctorType, base_class>>(functor, range_policy);
    closure.execute();
  }

  // Single with return value
  template <class FunctorType, class PolicyType, class ReturnType>
  static inline void execute(const std::string&, const PolicyType& policy,
                             const FunctorType& functor,
                             ReturnType& return_value) {
    Impl::SingleReductionWrapper<FunctorType> functor_wrapper{functor};
    using FunctorWrapper = decltype(functor_wrapper);

    using InternalReturn = Impl::SingleInternalValueWrapper<ReturnType>;

    using ReturnValueAdapter =
        typename Kokkos::Impl::ParallelReduceReturnValue<void, InternalReturn,
                                                         FunctorWrapper>;

    using ReducerType =
        typename Impl::NoOpReducer<InternalReturn, ExecutionSpace>;

    using Analysis =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              typename PolicyType::range_policy, ReducerType,
                              InternalReturn>;

    using CombinedFunctorReducerType =
        Kokkos::Impl::CombinedFunctorReducer<FunctorWrapper,
                                             typename Analysis::Reducer>;

    InternalReturn container;

    CombinedFunctorReducerType functor_reducer(
        functor_wrapper, typename Analysis::Reducer(container));

    using ParallelReducer =
        Impl::ParallelReduce<CombinedFunctorReducerType,
                             typename PolicyType::range_policy, ExecutionSpace>;

    auto closure =
        construct_with_shared_allocation_tracking_disabled<ParallelReducer>(
            functor_reducer, policy.impl_get_range_policy(),
            ReturnValueAdapter::return_value(container, functor_wrapper));

    closure.execute();

    return_value = container.content;
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_SINGLE_DEFAULT_IMPL_HPP
