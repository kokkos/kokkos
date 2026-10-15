// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

/// \file Kokkos_Single.hpp
/// \brief Declaration of the single operator

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_SINGLE_HPP
#define KOKKOS_SINGLE_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_View.hpp>

#include <impl/Kokkos_BuiltinReducers.hpp>
#include <impl/Kokkos_UnbindIndexArgument.hpp>
#include <impl/Kokkos_Single_Default_Impl.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>

namespace Kokkos {

/** \brief Execute \c functor on a specific ExecutionSpace in a single thread.
 *
 */
// single clause returning no value (based on parallel_for)
template <class FunctorType, class... PolicyProperties>
inline void single(const std::string& label,
                   const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor) {
  // Single takes an indexless functor but we will use the default parallel_for
  // implementation so we wrap the functor in an other functor with an index.
  ::Kokkos::Impl::UnbindIndexArgument<FunctorType> functor_wrapper{functor};
  using FunctorWrapper = decltype(functor_wrapper);

  uint64_t kpID = 0;
  Kokkos::Tools::Impl::begin_single<SinglePolicy<PolicyProperties...>,
                                    FunctorType>(single_policy, label, kpID);

  using execution_space = typename Impl::FunctorPolicyExecutionSpace<
      FunctorWrapper, typename std::remove_cvref_t<decltype(single_policy)>::
                          range_policy>::execution_space;

  // Dispatch execution to either the default implementation or an
  // execution_space specific implementation if one is available
  Kokkos::Impl::Single<execution_space>::template execute<>(functor_wrapper,
                                                            single_policy);

  Kokkos::Tools::Impl::end_single<FunctorType>(kpID);
}

template <class FunctorType, class... PolicyProperties>
inline void single(const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor) {
  ::Kokkos::single("", single_policy, functor);
}
template <class FunctorType>
inline void single(const std::string& label, const FunctorType& functor) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(label, policy(), functor);
}

template <class FunctorType>
inline void single(const FunctorType& functor) {
  ::Kokkos::single("", functor);
}

namespace Impl {
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
  void join(value_type&, const value_type&) const {}

  KOKKOS_INLINE_FUNCTION
  void init(value_type&) const {}
};
}  // namespace Impl

// single clause returning values (based on parallel_reduce)
template <class FunctorType, class ReturnType, class... PolicyProperties>
  requires(!Kokkos::is_view<ReturnType>::value &&
           !Kokkos::is_reducer<ReturnType>::value &&
           !std::is_pointer_v<ReturnType>)
inline void single(const std::string& label,
                   const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor, ReturnType& return_value) {
  // Single takes an indexless functor but we will use the default
  // parallel_reduce implementation so we wrap the functor in an other functor
  // with an index.
  ::Kokkos::Impl::UnbindReductionIndexArgument<
      FunctorType, typename SinglePolicy<PolicyProperties...>::work_tag>
      functor_wrapper{functor};
  using FunctorWrapper = decltype(functor_wrapper);

  using return_value_adapter =
      Impl::ParallelReduceReturnValue<void, ReturnType, FunctorWrapper>;

  using execution_space = typename Impl::FunctorPolicyExecutionSpace<
      FunctorWrapper, typename std::remove_cvref_t<decltype(single_policy)>::
                          range_policy>::execution_space;

  using ReducerType = typename Impl::NoOpReducer<ReturnType, execution_space>;

  using Analysis = Impl::FunctorAnalysis<
      Impl::FunctorPatternInterface::REDUCE,
      typename SinglePolicy<PolicyProperties...>::range_policy, ReducerType,
      ReturnType>;

  using CombinedFunctorReducerType =
      Kokkos::Impl::CombinedFunctorReducer<FunctorWrapper,
                                           typename Analysis::Reducer>;

  CombinedFunctorReducerType functor_reducer(
      functor_wrapper, typename Analysis::Reducer(return_value));

  uint64_t kpID = 0;
  Kokkos::Tools::Impl::begin_single<SinglePolicy<PolicyProperties...>,
                                    FunctorWrapper>(single_policy, label, kpID);

  // Dispatch execution to either the default implementation or an
  // execution_space specific implementation if one is available
  Kokkos::Impl::Single<execution_space>::template execute<
      CombinedFunctorReducerType, return_value_adapter>(
      functor_wrapper, functor_reducer, single_policy, return_value);

  Kokkos::Tools::Impl::end_single<FunctorType>(kpID);
}

template <class FunctorType, class ReturnType, class... PolicyProperties>
  requires(!Kokkos::is_view<ReturnType>::value &&
           !Kokkos::is_reducer<ReturnType>::value &&
           !std::is_pointer_v<ReturnType> &&
           std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", single_policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(!Kokkos::is_view<ReturnType>::value &&
           !Kokkos::is_reducer<ReturnType>::value &&
           !std::is_pointer_v<ReturnType> &&
           std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const std::string label, const FunctorType& functor,
                   ReturnType& return_value) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(label, policy(), functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(!Kokkos::is_view<ReturnType>::value &&
           !Kokkos::is_reducer<ReturnType>::value &&
           !std::is_pointer_v<ReturnType> &&
           std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", functor, return_value);
}
}  // namespace Kokkos

#endif  // KOKKOS_SINGLE_HPP
