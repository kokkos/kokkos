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

#include <impl/Kokkos_Single_fwd.hpp>
#include <impl/Kokkos_BuiltinReducers.hpp>
#include <impl/Kokkos_UnbindIndexArgument.hpp>

namespace Kokkos::Impl {

// Reducer performing no operation used within single to return a value without
// imposing the value's type to have any operation defined.
template <class ReturnViewType>
struct NoOpReducer : BaseReducer<typename ReturnViewType::value_type,
                                 typename ReturnViewType::memory_space> {
 private:
  using parent_type = BaseReducer<typename ReturnViewType::value_type,
                                  typename ReturnViewType::memory_space>;

 public:
  using reducer    = NoOpReducer;
  using value_type = typename parent_type::value_type;

  NoOpReducer(const ReturnViewType& value_in) : parent_type(value_in) {}

  KOKKOS_FUNCTION
  void join(value_type& dest, const value_type& src) const { dest = src; }

  KOKKOS_FUNCTION
  void init(value_type&) const {}
};

// Functor to be used with the above Reducer
template <class FunctorType>
struct SingleReductionWrapper {
  FunctorType m_functor;
  template <class WorkTag, class Index, class ReturnType>
  KOKKOS_FUNCTION void operator()(WorkTag worktag, Index,
                                  ReturnType& ret) const {
    static_assert(std::is_empty_v<WorkTag>);
    m_functor(worktag, ret);
  }

  template <class Index, class ReturnType>
  KOKKOS_FUNCTION void operator()(Index, ReturnType& ret) const {
    m_functor(ret);
  }
};

// Functor to be used if the execution space can
// access the return view type directly
template <class FunctorType, class ReturnViewType>
struct SingleWithReturnFunctor {
  FunctorType m_functor;
  ReturnViewType m_return_view;
  static_assert(ReturnViewType::rank() == 0);

  template <class WorkTag, class Index>
  KOKKOS_FUNCTION void operator()(WorkTag worktag, Index) const {
    static_assert(std::is_empty_v<WorkTag>);
    m_functor(worktag, m_return_view());
  }

  template <class Index>
  KOKKOS_FUNCTION void operator()(Index) const {
    m_functor(m_return_view());
  }
};

// Default implementation for execution spaces that don't provide a definition

// This is the version for a single call without return value
// It forwards to ParallelFor with a RangePolicy
template <class FunctorType, class ExecutionSpace, class... Traits>
struct Single<FunctorType, SinglePolicy<Traits...>, ExecutionSpace> {
  FunctorType functor;
  SinglePolicy<Traits...> policy;

  void execute() const {
    // wrap the functor into one which accepts an index
    using FunctorWrapper = Kokkos::Impl::UnbindIndexArgument<FunctorType>;
    FunctorWrapper functor_wrapper{functor};

    using InnerPolicy =
        std::remove_cvref_t<decltype(policy.impl_get_range_policy())>;

    // Create the closure and call it. No need to turn ref count off, since
    // we did that already.
    auto closure = Impl::ParallelFor<FunctorWrapper, InnerPolicy>(
        functor_wrapper, policy.impl_get_range_policy());
    closure.execute();
  }
};

// This is the version for a single call with a return value.
template <class FunctorType, class ReturnViewType, class ExecutionSpace,
          class... Traits>
struct SingleWithReturn<FunctorType, SinglePolicy<Traits...>, ReturnViewType,
                        ExecutionSpace> {
  FunctorType functor;
  SinglePolicy<Traits...> policy;
  ReturnViewType return_value;

  static_assert(Kokkos::is_view_v<ReturnViewType>);

  void execute() const {
    // If the return View is accessile by the ExecutionSpace simply
    // directly write to it from a ParallelFor.
    // Otherwise run a reduction to deal with the extra required scratch.
    if constexpr (Kokkos::SpaceAccessibility<
                      ExecutionSpace,
                      typename ReturnViewType::memory_space>::accessible) {
      using FunctorWrapper =
          SingleWithReturnFunctor<FunctorType, ReturnViewType>;
      FunctorWrapper functor_wrapper(functor, return_value);

      using InnerPolicy =
          std::remove_cvref_t<decltype(policy.impl_get_range_policy())>;

      // Create the closure and call it. No need to turn ref count off, since
      // we did that already.
      auto closure = Impl::ParallelFor<FunctorWrapper, InnerPolicy>(
          functor_wrapper, policy.impl_get_range_policy());
      closure.execute();
    } else {
      // We are using the implementation detail ParallelReduce here
      // In order to do this, we need to wrap the functor multiple times

      // This path is not supported by OpenMP or HPX because it relies
      // on ParallelReduce behaving a certain way with respect to
      // non-participating threads. Specifically, initializing all values for
      // each thread and then combining will not work.

#ifdef KOKKOS_ENABLE_OPENMP
      static_assert(!std::is_same_v<ExecutionSpace, OpenMP>);
#endif
#ifdef KOKKOS_ENABLE_HPX
      static_assert(!std::is_same_v<ExecutionSpace, Experimental::HPX>);
#endif

      // The first Wrap is to make the user functor in a reduction one,
      // that takes an index
      using FunctorWrapper = Impl::SingleReductionWrapper<FunctorType>;
      FunctorWrapper functor_wrapper{functor};

      // Now we need a no-op reducer that doesn't have any requirements
      // on the value_type (i.e. no += etc. which the default reduction
      // expects).
      using ReducerType = typename Impl::NoOpReducer<ReturnViewType>;

      // Get the execution policy we are mapping to (RangePolicy with 1
      // iteration)
      using InnerPolicy =
          std::remove_cvref_t<decltype(policy.impl_get_range_policy())>;

      // This is our canonical reduction analysis
      using Analysis =
          Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                                InnerPolicy, ReducerType, ReturnViewType>;

      // Analysis::Reducer adds all kinds of extra info for the reduction
      // backend like does it have init, does it have join, value size
      using CombinedFunctorReducerType =
          Kokkos::Impl::CombinedFunctorReducer<FunctorWrapper,
                                               typename Analysis::Reducer>;

      CombinedFunctorReducerType functor_reducer(
          functor_wrapper, typename Analysis::Reducer(return_value));

      // The actual implementation for ParallelReduce
      using ParallelReducer = Impl::ParallelReduce<CombinedFunctorReducerType,
                                                   InnerPolicy, ExecutionSpace>;

      // Create the closure and call it. No need to turn ref count off, since
      // we did that already.
      auto closure = ParallelReducer(
          functor_reducer, policy.impl_get_range_policy(), return_value);

      closure.execute();
    }
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_SINGLE_DEFAULT_IMPL_HPP
