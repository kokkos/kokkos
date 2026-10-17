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

// single clause returning values (based on parallel_reduce)
template <class FunctorType, class ReturnType, class... PolicyProperties>
inline void single(const std::string& label,
                   const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor, ReturnType& return_value) {
  using execution_space = typename Impl::FunctorPolicyExecutionSpace<
      FunctorType, typename std::remove_cvref_t<
                       decltype(single_policy)>::range_policy>::execution_space;

  uint64_t kpID = 0;
  Kokkos::Tools::Impl::begin_single<SinglePolicy<PolicyProperties...>,
                                    FunctorType>(single_policy, label, kpID);

  // Dispatch execution to either the default implementation or an
  // execution_space specific implementation if one is available
  Kokkos::Impl::Single<execution_space>::template execute<
      FunctorType, SinglePolicy<PolicyProperties...>, ReturnType>(
      label, single_policy, functor, return_value);

  Kokkos::Tools::Impl::end_single<FunctorType>(kpID);
}

template <class FunctorType, class ReturnType, class... PolicyProperties>
  requires(std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", single_policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const std::string label, const FunctorType& functor,
                   ReturnType& return_value) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(label, policy(), functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(std::is_invocable_v<FunctorType, ReturnType&>)
inline void single(const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", functor, return_value);
}
}  // namespace Kokkos

#endif  // KOKKOS_SINGLE_HPP
