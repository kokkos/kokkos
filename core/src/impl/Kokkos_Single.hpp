// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_SINGLE_HPP
#define KOKKOS_SINGLE_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <impl/Kokkos_Single_fwd.hpp>
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
  using ExecPolicy = SinglePolicy<PolicyProperties...>;

  uint64_t kpID = 0;
  Kokkos::Tools::Impl::begin_single<ExecPolicy, FunctorType>(single_policy,
                                                             label, kpID);

  auto closure = Impl::construct_with_shared_allocation_tracking_disabled<
      Impl::Single<FunctorType, ExecPolicy>>(functor, single_policy);

  closure.execute();

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
  using ExecPolicy = SinglePolicy<PolicyProperties...>;

  // We always pass a View to the implementation, wrap scalar values
  // into a Host View.
  using ReturnViewType =
      std::conditional_t<Kokkos::is_view_v<ReturnType>, ReturnType,
                         Kokkos::View<ReturnType, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;

  ReturnViewType return_view;
  if constexpr (Kokkos::is_view_v<ReturnType>) {
    return_view = return_value;
  } else {
    return_view = ReturnViewType(&return_value);
  }

  uint64_t kpID = 0;
  Kokkos::Tools::Impl::begin_single<ExecPolicy, FunctorType>(single_policy,
                                                             label, kpID);

  // Dispatch execution to either the default implementation or an
  // execution_space specific implementation if one is available
  auto closure =
      Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
          Impl::SingleWithReturn<FunctorType, ExecPolicy, ReturnViewType>>(
          functor, single_policy, return_view);

  closure.execute();

  Kokkos::Tools::Impl::end_single<FunctorType>(kpID);

  // if output is not a View we need to fence like with parallel_reduce
  // Note the tools call comes before: if a tool fences the time for the
  // single op is associated with the single op, otherwise with its
  // associated fence - this is consistent with parallel_reduce
  if constexpr (!Kokkos::is_view_v<ReturnType>) {
    single_policy.space().fence(
        "Kokkos::single: fence due to scalar value output");
  }
}

template <class FunctorType, class ReturnType, class... PolicyProperties>
  requires(std::invocable<FunctorType, ReturnType&> ||
           (Kokkos::is_view<ReturnType>::value &&
            std::invocable<FunctorType, typename ReturnType::data_type&> &&
            ReturnType::rank == 0))
inline void single(const SinglePolicy<PolicyProperties...>& single_policy,
                   const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", single_policy, functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(std::invocable<FunctorType, ReturnType&> ||
           (Kokkos::is_view<ReturnType>::value &&
            std::invocable<FunctorType, typename ReturnType::data_type&> &&
            ReturnType::rank == 0))
inline void single(const std::string label, const FunctorType& functor,
                   ReturnType& return_value) {
  using execution_space =
      typename Impl::FunctorPolicyExecutionSpace<FunctorType,
                                                 void>::execution_space;
  using policy = SinglePolicy<execution_space>;
  ::Kokkos::single(label, policy(), functor, return_value);
}

template <class FunctorType, class ReturnType>
  requires(std::invocable<FunctorType, ReturnType&> ||
           (Kokkos::is_view<ReturnType>::value &&
            std::invocable<FunctorType, typename ReturnType::data_type&> &&
            ReturnType::rank == 0))
inline void single(const FunctorType& functor, ReturnType& return_value) {
  ::Kokkos::single("", functor, return_value);
}
}  // namespace Kokkos

#endif  // KOKKOS_SINGLE_HPP
