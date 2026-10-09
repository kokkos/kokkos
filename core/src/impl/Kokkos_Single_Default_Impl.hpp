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
#include <impl/Kokkos_CStyleMemoryManagement.hpp>
#include <impl/Kokkos_FunctorWrapperUtil.hpp>

namespace Kokkos::Impl {

// TODO: this really needs to be per execspace instance storage
inline void* get_single_value_buffer_ptr(size_t requested_size) {
  static size_t size = 0lu;
  static void* ptr   = nullptr;
  if (size < requested_size) {
    Kokkos::kokkos_free<Kokkos::SharedHostPinnedSpace>(ptr);
    ptr  = Kokkos::kokkos_malloc<Kokkos::SharedHostPinnedSpace>(requested_size);
    size = requested_size;
  }
  return ptr;
}

template <class ValueType>
auto get_single_value_buffer() {
  return Kokkos::View<ValueType, Kokkos::SharedHostPinnedSpace>(
      static_cast<ValueType*>(get_single_value_buffer_ptr(sizeof(ValueType))));
}

// Default implementation for execution spaces that don't provide a definition
template <typename ExecutionSpace>
struct Single {
  template <class FunctorType, class SinglePolicy>
  static void execute(const FunctorType& functor,
                      const SinglePolicy& single_policy) {
    // We will use the standard function for parallel_for, so we need to modify
    // the functor in order to make it callable by the standard function by
    // giving it an index parameter
    ::Kokkos::Impl::IndexlessFunctorWrapper<FunctorType,
                                            typename SinglePolicy::work_tag>
        functor_wrapper{functor};

    using WrapperType = decltype(functor_wrapper);

    using base_class = typename std::remove_cvref_t<SinglePolicy>::range_policy;
    const base_class& range_policy = single_policy.impl_get_range_policy();
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelFor<WrapperType, base_class>>(functor_wrapper,
                                                        range_policy);
    closure.execute();
  }

  template <class FunctorType, class SinglePolicy, class ReturnViewType>
  static void execute(const FunctorType& functor,
                      const SinglePolicy& single_policy,
                      const ReturnViewType& return_value) {
    constexpr bool need_deep_copy = !Kokkos::SpaceAccessibility<
        typename SinglePolicy::execution_space,
        typename ReturnViewType::memory_space>::accessible;

    using buffer_type = Kokkos::View<typename ReturnViewType::value_type,
                                     Kokkos::AnonymousSpace>;

    using wrapped_functor_type = ::Kokkos::Impl::IndexlessValueFunctorWrapper<
        FunctorType, buffer_type, typename SinglePolicy::work_tag>;

    buffer_type buffer;
    if constexpr (need_deep_copy) {
      buffer = get_single_value_buffer<typename ReturnViewType::value_type>();
    } else {
      buffer = return_value;
    }

    wrapped_functor_type functor_wrapper{functor, buffer};

    using base_policy =
        typename std::remove_cvref_t<SinglePolicy>::range_policy;
    const base_policy& range_policy = single_policy.impl_get_range_policy();
    auto closure =
        Kokkos::Impl::construct_with_shared_allocation_tracking_disabled<
            Impl::ParallelFor<wrapped_functor_type, base_policy>>(
            functor_wrapper, range_policy);
    closure.execute();

    if constexpr (need_deep_copy) {
      // TODO: do we want so support async return for inaccessible memory
      range_policy.space().fence();
      return_value() = buffer();
    }
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_SINGLE_DEFAULT_IMPL_HPP
