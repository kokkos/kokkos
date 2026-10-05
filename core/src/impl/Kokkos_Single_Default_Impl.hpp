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
#include <impl/Kokkos_FunctorWrapperUtil.hpp>

namespace Kokkos::Impl {
// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace>
struct Single {
  template <class FunctorType, class SinglePolicy>
  static void execute(const FunctorType& functor,
                      const SinglePolicy& single_policy) {
    // We will use the standard function for parallel_for, so we need to modify
    // the functor in order to make it callable by the standard function by
    // giving it an index parameter
    ::Kokkos::Impl::IndexlessFunctorWrapper<FunctorType> functor_wrapper{
        functor};

    using WrapperType = decltype(functor_wrapper);

    using base_class = typename std::remove_cvref_t<SinglePolicy>::range_policy;
    const base_class& range_policy = single_policy.impl_get_range_policy();
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelFor<WrapperType, base_class>>(functor_wrapper,
                                                        range_policy);
    closure.execute();
  }

  template <class CombinedFunctorReducerType, class ReturnValueAdapter,
            class FunctorType, class FunctorReducerType, class SinglePolicy,
            class ReturnType>
  static void execute(const FunctorType& functor,
                      const FunctorReducerType& functor_reducer,
                      const SinglePolicy& single_policy,
                      ReturnType& return_value) {
    using base_class = typename std::remove_cvref_t<SinglePolicy>::range_policy;
    using parallel_reducer = Impl::ParallelReduce<CombinedFunctorReducerType,
                                                  base_class, ExecutionSpace>;

    const base_class& range_policy = single_policy.impl_get_range_policy();
    auto closure =
        construct_with_shared_allocation_tracking_disabled<parallel_reducer>(
            functor_reducer, range_policy,
            ReturnValueAdapter::return_value(return_value, functor));
    closure.execute();
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_SINGLE_DEFAULT_IMPL_HPP
