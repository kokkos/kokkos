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

#include <impl/Kokkos_FunctorWrapperUtil.hpp>
#include <impl/Kokkos_Single_Default_Impl.hpp>

namespace Kokkos {

/** \brief Execute \c functor on a specific ExecutionSpace in a single thread.
 *
 */
template <class FunctorType, class... PolicyProperties>
inline void single(const std::string& str,
                   const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor) {
  uint64_t kpID = 0;

  Kokkos::Tools::Impl::begin_single<SinglePolicy<PolicyProperties...>,
                                    FunctorType>(single_policy, str, kpID);

  using execution_space = typename Impl::FunctorPolicyExecutionSpace<
      FunctorType, typename std::remove_cvref_t<
                       decltype(single_policy)>::base_class>::execution_space;

  // Dispatch execution to either the default implementation or an
  // execution_space specific implementation if one is available
  Kokkos::Impl::Single<execution_space>::template execute(functor,
                                                          single_policy);

  Kokkos::Tools::Impl::end_single<FunctorType>(kpID);
}

template <class FunctorType, class... PolicyProperties>
inline void single(const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor) {
  ::Kokkos::single("", single_policy, functor);
}
template <class FunctorType>
inline void single(const std::string& str, const FunctorType& functor) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(str, policy(), functor);
}

template <class FunctorType>
inline void single(const FunctorType& functor) {
  ::Kokkos::single("", functor);
}

template <class FunctorType, class ReturnType, class... PolicyProperties>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
single(const std::string& label,
       const SinglePolicy<PolicyProperties...>& single_policy,
       const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::Impl::IndexlessReductionFunctorWrapper<
      FunctorType, typename SinglePolicy<PolicyProperties...>::work_tag>
      functor_wrapper{functor};

  ::Kokkos::parallel_reduce(label, single_policy, functor_wrapper,
                            return_value);
}

template <class FunctorType, class ReturnType, class... PolicyProperties>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)&&std::
                            is_invocable_v<FunctorType, ReturnType&>>
single(const SinglePolicy<PolicyProperties...>& single_policy,
       const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", single_policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<!(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)&&std::
                            is_invocable_v<FunctorType, ReturnType&>>
single(const std::string label, const FunctorType& functor,
       ReturnType& return_value) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(label, policy(), functor, return_value);
}

template <class FunctorType, class ReturnType>
inline std::enable_if_t<std::is_invocable_v<FunctorType, ReturnType&> &&
                        !(Kokkos::is_view<ReturnType>::value ||
                          Kokkos::is_reducer<ReturnType>::value ||
                          std::is_pointer_v<ReturnType>)>
single(const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", functor, return_value);
}
}  // namespace Kokkos

#endif  // KOKKOS_SINGLE_HPP
