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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_VIEW_ALLOC_HPP
#define KOKKOS_VIEW_ALLOC_HPP

#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos::Impl
{

  template <typename T>
  inline bool is_zero_byte(const T& t) {
    using comparison_type = std::conditional_t<
        sizeof(T) % sizeof(long long int) == 0, long long int,
        std::conditional_t<
            sizeof(T) % sizeof(long int) == 0, long int,
            std::conditional_t<
                sizeof(T) % sizeof(int) == 0, int,
                std::conditional_t<sizeof(T) % sizeof(short int) == 0, short int,
                                  char>>>>;
    const auto* const ptr = reinterpret_cast<const comparison_type*>(&t);
    for (std::size_t i = 0; i < sizeof(T) / sizeof(comparison_type); ++i)
      if (ptr[i] != 0) return false;
    return true;
  }


  //----------------------------------------------------------------------------

  /*
  *  The construction, assignment to default, and destruction
  *  are merged into a single functor.
  *  Primarily to work around an unresolved CUDA back-end bug
  *  that would lose the destruction cuda device function when
  *  called from the shared memory tracking destruction.
  *  Secondarily to have two fewer partial specializations.
  */
  template <class DeviceType, class ValueType,
            bool IsScalar = std::is_scalar<ValueType>::value>
  struct ViewAllocFunctor;

  template <class DeviceType, class ValueType>
  struct ViewAllocFunctor<DeviceType, ValueType, false /* is_scalar */> {
    using ExecSpace = typename DeviceType::execution_space;

    struct DestroyTag {};
    struct ConstructTag {};

    ExecSpace space;
    ValueType* ptr;
    size_t n;
    std::string name;
    bool default_exec_space;

    template <class _ValueType = ValueType>
    KOKKOS_INLINE_FUNCTION
        std::enable_if_t<std::is_default_constructible<_ValueType>::value>
        operator()(ConstructTag const&, const size_t i) const {
      new (ptr + i) ValueType();
    }

    KOKKOS_INLINE_FUNCTION void operator()(DestroyTag const&,
                                          const size_t i) const {
      (ptr + i)->~ValueType();
    }

    ViewAllocFunctor()                        = default;
    ViewAllocFunctor(const ViewAllocFunctor&) = default;
    ViewAllocFunctor& operator=(const ViewAllocFunctor&) = default;

    ViewAllocFunctor(ExecSpace const& arg_space, ValueType* const arg_ptr,
                    size_t const arg_n, std::string arg_name)
        : space(arg_space),
          ptr(arg_ptr),
          n(arg_n),
          name(std::move(arg_name)),
          default_exec_space(false) {
      functor_instantiate_workaround();
    }

    ViewAllocFunctor(ValueType* const arg_ptr, size_t const arg_n,
                    std::string arg_name)
        : space(ExecSpace{}),
          ptr(arg_ptr),
          n(arg_n),
          name(std::move(arg_name)),
          default_exec_space(true) {
      functor_instantiate_workaround();
    }

    template <typename Dummy = ValueType>
    std::enable_if_t<std::is_trivial<Dummy>::value &&
                    std::is_trivially_copy_assignable<ValueType>::value>
    construct_dispatch() {
      ValueType value{};
  // On A64FX memset seems to do the wrong thing with regards to first touch
  // leading to the significant performance issues
  #ifndef KOKKOS_ARCH_A64FX
      if (Impl::is_zero_byte(value)) {
        uint64_t kpID = 0;
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          // We are not really using parallel_for here but using beginParallelFor
          // instead of begin_parallel_for (and adding "via memset") is the best
          // we can do to indicate that this is not supposed to be tunable (and
          // doesn't really execute a parallel_for).
          Kokkos::Profiling::beginParallelFor(
              "Kokkos::View::initialization [" + name + "] via memset",
              Kokkos::Profiling::Experimental::device_id(space), &kpID);
        }
        (void)ZeroMemset<
            ExecSpace, Kokkos::View<ValueType*, typename DeviceType::memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>(
            space,
            Kokkos::View<ValueType*, typename DeviceType::memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, n),
            value);

        if (Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
        if (default_exec_space)
          space.fence("Kokkos::Impl::ViewAllocFunctor: View init/destroy fence");
      } else {
  #endif
        parallel_for_implementation<ConstructTag>();
  #ifndef KOKKOS_ARCH_A64FX
      }
  #endif
    }

    template <typename Dummy = ValueType>
    std::enable_if_t<!(std::is_trivial<Dummy>::value &&
                      std::is_trivially_copy_assignable<ValueType>::value)>
    construct_dispatch() {
      parallel_for_implementation<ConstructTag>();
    }

    template <typename Tag>
    void parallel_for_implementation() {
      if (!space.in_parallel()) {
        using PolicyType =
            Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<int64_t>, Tag>;
        PolicyType policy(space, 0, n);
        uint64_t kpID = 0;
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          const std::string functor_name =
              (std::is_same_v<Tag, DestroyTag>
                  ? "Kokkos::View::destruction [" + name + "]"
                  : "Kokkos::View::initialization [" + name + "]");
          Kokkos::Profiling::beginParallelFor(
              functor_name, Kokkos::Profiling::Experimental::device_id(space),
              &kpID);
        }

  #ifdef KOKKOS_ENABLE_CUDA
        if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
          Kokkos::Impl::cuda_prefetch_pointer(space, ptr, sizeof(ValueType) * n,
                                              true);
        }
  #endif
        const Kokkos::Impl::ParallelFor<ViewAllocFunctor, PolicyType> closure(
            *this, policy);
        closure.execute();
        if (default_exec_space || std::is_same_v<Tag, DestroyTag>)
          space.fence("Kokkos::Impl::ViewAllocFunctor: View init/destroy fence");
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
      } else {
        for (size_t i = 0; i < n; ++i) operator()(Tag{}, i);
      }
    }

    void construct_shared_allocation() { construct_dispatch(); }

    void destroy_shared_allocation() {
      parallel_for_implementation<DestroyTag>();
    }

    // This function is to ensure that the functor with DestroyTag is instantiated
    // This is a workaround to avoid "cudaErrorInvalidDeviceFunction" error later
    // when the function is queried with cudaFuncGetAttributes
    void functor_instantiate_workaround() {
  #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
      defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
      if (false) {
        parallel_for_implementation<DestroyTag>();
      }
  #endif
    }
  };


  template <class DeviceType, class ValueType>
  struct ViewAllocFunctor<DeviceType, ValueType, true /* is_scalar */> {
    using ExecSpace  = typename DeviceType::execution_space;
    using PolicyType = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<int64_t>>;

    ExecSpace space;
    ValueType* ptr;
    size_t n;
    std::string name;
    bool default_exec_space;

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const { ptr[i] = ValueType(); }

    ViewAllocFunctor()                        = default;
    ViewAllocFunctor(const ViewAllocFunctor&) = default;
    ViewAllocFunctor& operator=(const ViewAllocFunctor&) = default;

    ViewAllocFunctor(ExecSpace const& arg_space, ValueType* const arg_ptr,
                    size_t const arg_n, std::string arg_name)
        : space(arg_space),
          ptr(arg_ptr),
          n(arg_n),
          name(std::move(arg_name)),
          default_exec_space(false) {}

    ViewAllocFunctor(ValueType* const arg_ptr, size_t const arg_n,
                    std::string arg_name)
        : space(ExecSpace{}),
          ptr(arg_ptr),
          n(arg_n),
          name(std::move(arg_name)),
          default_exec_space(true) {}

    template <typename Dummy = ValueType>
    std::enable_if_t<std::is_trivial<Dummy>::value &&
                    std::is_trivially_copy_assignable<Dummy>::value>
    construct_shared_allocation() {
      // Shortcut for zero initialization
  // On A64FX memset seems to do the wrong thing with regards to first touch
  // leading to the significant performance issues
  #ifndef KOKKOS_ARCH_A64FX
      ValueType value{};
      if (Impl::is_zero_byte(value)) {
        uint64_t kpID = 0;
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          // We are not really using parallel_for here but using beginParallelFor
          // instead of begin_parallel_for (and adding "via memset") is the best
          // we can do to indicate that this is not supposed to be tunable (and
          // doesn't really execute a parallel_for).
          Kokkos::Profiling::beginParallelFor(
              "Kokkos::View::initialization [" + name + "] via memset",
              Kokkos::Profiling::Experimental::device_id(space), &kpID);
        }

        (void)ZeroMemset<
            ExecSpace, Kokkos::View<ValueType*, typename DeviceType::memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>>(
            space,
            Kokkos::View<ValueType*, typename DeviceType::memory_space,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ptr, n),
            value);

        if (Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
        if (default_exec_space)
          space.fence("Kokkos::Impl::ViewAllocFunctor: View init/destroy fence");
      } else {
  #endif
        parallel_for_implementation();
  #ifndef KOKKOS_ARCH_A64FX
      }
  #endif
    }

    template <typename Dummy = ValueType>
    std::enable_if_t<!(std::is_trivial<Dummy>::value &&
                      std::is_trivially_copy_assignable<Dummy>::value)>
    construct_shared_allocation() {
      parallel_for_implementation();
    }

    void parallel_for_implementation() {
      if (!space.in_parallel()) {
        PolicyType policy(0, n);
        uint64_t kpID = 0;
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::beginParallelFor(
              "Kokkos::View::initialization [" + name + "]",
              Kokkos::Profiling::Experimental::device_id(space), &kpID);
        }
  #ifdef KOKKOS_ENABLE_CUDA
        if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
          Kokkos::Impl::cuda_prefetch_pointer(space, ptr, sizeof(ValueType) * n,
                                              true);
        }
  #endif
        const Kokkos::Impl::ParallelFor<ViewAllocFunctor, PolicyType> closure(
            *this, PolicyType(0, n));
        closure.execute();
        if (default_exec_space)
          space.fence(
              "Kokkos::Impl::ViewAllocFunctor: Fence after setting values in "
              "view");
        if (Kokkos::Profiling::profileLibraryLoaded()) {
          Kokkos::Profiling::endParallelFor(kpID);
        }
      } else {
        for (size_t i = 0; i < n; ++i) operator()(i);
      }
    }

    void destroy_shared_allocation() {}
  };

  template <class ElementType, class MappingType, class MemorySpace,
            class ExecutionSpace, bool AllowPadding, bool Initialize>
  Kokkos::Impl::SharedAllocationRecord<void, void>*
  make_shared_allocation_record(
      const MappingType& mapping, std::string_view label,
      const MemorySpace& memory_space, const ExecutionSpace* exec_space,
      [[maybe_unused]] std::integral_constant<bool, AllowPadding> allow_padding,
      [[maybe_unused]] std::integral_constant<bool, Initialize> initialize) {
    static_assert(SpaceAccessibility<ExecutionSpace, MemorySpace>::accessible);

    // Use this for constructing and destroying the view
    using alloc_functor_type = ViewAllocFunctor<Kokkos::Device<ExecutionSpace, MemorySpace>, ElementType>;
    using record_type = Kokkos::Impl::SharedAllocationRecord<MemorySpace, alloc_functor_type>;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    using padding = std::integral_constant<std::size_t, AllowPadding ? sizeof(ElementType) : 0>;

    static constexpr std::size_t align_mask = 0x7;
    static constexpr std::size_t element_size = sizeof(ElementType);

    // Calculate the total size of the memory, in bytes, and make sure it is byte-aligned
    const std::size_t alloc_size = (mapping.required_span_size() * element_size + align_mask) & ~align_mask;

    auto* record =
        exec_space ? record_type::allocate(*exec_space, memory_space, std::string{label},
                                           alloc_size)
                   : record_type::allocate(memory_space, std::string{label}, alloc_size);

    auto data_handle = static_cast<ElementType *>(record->data());

    auto functor = exec_space
                       ? alloc_functor_type(*exec_space, data_handle,
                                            mapping.required_span_size(), std::string{label})
                       : alloc_functor_type(data_handle,
                                            mapping.required_span_size(), std::string{label});

    //  Only initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if constexpr (Initialize)
    {
      if (alloc_size) {
        // Assume destruction is only required when construction is requested.
        // The ViewValueFunctor has both value construction and destruction
        // operators.
        record->m_destroy = std::move(functor);

        // Construct values
        record->m_destroy.construct_shared_allocation();
      }
    }

    return record;
  }

  template <typename MemorySpace, typename ExecutionSpace, bool AllowPadding = false,
            bool Initialize = false, typename PointerType = void>
  struct alloc_params;

  template <typename MemorySpace, typename ExecutionSpace, bool AllowPadding,
            bool Initialize>
  struct alloc_params<MemorySpace, ExecutionSpace, AllowPadding, Initialize,
                      void> {
    MemorySpace memory_space       = {};
    ExecutionSpace execution_space = {};
    std::integral_constant<bool, AllowPadding> allow_padding;
    std::integral_constant<bool, Initialize> initialize;
  };

  template <bool AllowPadding, bool Initialize, typename PointerType>
  struct alloc_params<void, void, AllowPadding, Initialize, PointerType> {
    PointerType pointer;
    std::integral_constant<bool, AllowPadding> allow_padding;
    std::integral_constant<bool, Initialize> initialize;
  };
}  // namespace Kokkos::Impl

#endif // KOKKOS_VIEW_ALLOC_HPP
