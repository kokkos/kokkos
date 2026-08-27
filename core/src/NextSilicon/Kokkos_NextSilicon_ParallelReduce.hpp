// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP

#include <type_traits>
#include <cstdint>
#include <mutex>
#include <Kokkos_Parallel.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Intrinsics.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ThreadSpaceGuard.hpp>
#include <nextapi/parallelism.hpp>
#include <nextapi/memory.h>

namespace Kokkos::Impl {

template <class CombinedFunctorReducerType, class... Traits>
struct NextSiliconParallelReduceImpl {
  using ExecSpace   = Kokkos::Experimental::NextSilicon;
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy      = RangePolicy<Traits...>;
  using IndexType   = typename Policy::index_type;
  using WorkRange   = typename Policy::WorkRange;
  using WorkTag     = typename Policy::work_tag;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Pointer       = typename ReducerType::pointer_type;
  using ValueType     = typename ReducerType::value_type;
  using ReferenceType = typename ReducerType::reference_type;

  constexpr static uint32_t MAX_PARTIAL_PROD          = 16 * 1024;  // 16K
  constexpr static uint32_t MIN_ITER_PER_PARTIAL_PROD = 100;

  constexpr static uint32_t SCRATCH_ALLOC_ALIGNMENT = 64;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  Pointer m_result_ptr;

  NextSiliconParallelReduceImpl(
      CombinedFunctorReducerType const& functor_reducer, Policy const& policy,
      Pointer result_ptr)
      : m_functor_reducer(functor_reducer),
        m_policy(policy),
        m_result_ptr(result_ptr) {}

  void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    IndexType begin = m_policy.begin();
    IndexType end   = m_policy.end();

    const auto& reducer      = m_functor_reducer.get_reducer();
    IndexType num_iterations = end - begin;

    if (end <= begin) {
      num_iterations = 0;
    }

    // Calculate optimal number of partial products
    uint32_t num_partial_prod =
        (num_iterations + (MIN_ITER_PER_PARTIAL_PROD - 1)) /
        (MIN_ITER_PER_PARTIAL_PROD);
    num_partial_prod = std::min(num_partial_prod, MAX_PARTIAL_PROD);

    // Even if num_iterations 0, set num_partial_prod to at least 1 to handle
    // the case where we need to init/final the result_ptr.
    if (num_partial_prod == 0) num_partial_prod = 1;

    const auto value_count = reducer.value_count();
    const auto values_size = value_count * sizeof(ValueType);

    // Round up to SCRATCH_ALLOC_ALIGNMENT
    size_t result_arr_size =
        ((values_size * num_partial_prod + SCRATCH_ALLOC_ALIGNMENT - 1) /
         SCRATCH_ALLOC_ALIGNMENT) *
        SCRATCH_ALLOC_ALIGNMENT;

    const size_t default_reducer_values_size =
        ((values_size + SCRATCH_ALLOC_ALIGNMENT - 1) /
         SCRATCH_ALLOC_ALIGNMENT) *
        SCRATCH_ALLOC_ALIGNMENT;

    auto internal_instance = m_policy.space().impl_internal_space_instance();

    // Leave space in scratch memory for default reducer values
    size_t scratch_memory_byte_size =
        result_arr_size + default_reducer_values_size;

    std::byte* partial_buffer = internal_instance->resize_reduce_partial_buffer(
        scratch_memory_byte_size);

    ValueType* const result_arr = reinterpret_cast<ValueType*>(partial_buffer);
    partial_buffer += result_arr_size;

    // Clone default reducer values to HBM scratch so RISC workers can read
    // across the parallel region boundary without accessing the control
    // thread's stack frame.
    ValueType* const default_reducer_values =
        reinterpret_cast<ValueType*>(partial_buffer);

    // Handle case with no iterations
    if (end <= begin) {
      ValueType* ptr = (m_result_ptr) ? m_result_ptr : result_arr;

      reducer.init(ptr);
      reducer.final(ptr);
      return;
    }

    // Clone the driver before going into the handoff function.
    // This prevents the stack from getting migrated into device.
    auto cloned_driver = internal_instance->clone_driver(*this);
    cloned_driver->execute_internal(num_partial_prod, result_arr,
                                    default_reducer_values);

    // Final result is at the begining of values array
    if (m_result_ptr) {
      nextapi_memory_copy(m_result_ptr, result_arr,
                          sizeof(ValueType) * reducer.value_count());
    }
  }

  __attribute__((noinline)) void execute_internal(
      uint32_t num_partial_prod, ValueType* const result_arr,
      ValueType* default_reducer_values) const {
    const auto& functor = m_functor_reducer.get_functor();
    const auto& reducer = m_functor_reducer.get_reducer();

    IndexType begin      = 0;
    IndexType end        = num_partial_prod;
    IndexType chunk_size = 1;

    {
      // Set the thread's semantic execution space to device
      //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed
      //   description of the mechanism and rationale.
      NextSiliconThreadSpaceGuard thread_guard{};
      reducer.init(default_reducer_values);
    }

    nextapi::parallel_for<nextapi::work_distribution::STATIC>(
        begin, end, {.chunk_size = chunk_size}, microtask, num_partial_prod,
        &functor, &reducer, &m_policy, result_arr, default_reducer_values);

    {
      // Set the thread's semantic execution space to device
      //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed
      //   description of the mechanism and rationale.
      NextSiliconThreadSpaceGuard thread_guard{};
      const auto value_count = reducer.value_count();

      for (uint32_t i = 1; i < num_partial_prod; ++i) {
        reducer.join(&result_arr[0], &result_arr[i * value_count]);
      }

      reducer.final(&result_arr[0]);
    }
  }

  static void microtask(IndexType thread_index, uint32_t num_partial_prod,
                        const FunctorType* functor, const ReducerType* reducer,
                        const Policy* policy, ValueType* result_arr,
                        const ValueType* const default_reducer_values) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    // Communicate to the compiler that the functor is immutable and thread
    // invariant for the duration of the microtask.
    if (__next_is_in_handed_off_code()) {
      nextapi::detail::__next_immutable_thread_invariant_parameter_struct(
          functor);
    }

    WorkRange range(*policy, thread_index, num_partial_prod);

    auto value_count    = reducer->value_count();
    auto* interm_values = &result_arr[thread_index * value_count];

    for (size_t i = 0; i < value_count; ++i) {
      interm_values[i] = default_reducer_values[i];
    }

    ReferenceType ref = reducer->reference(interm_values);

    for (auto i = range.begin(); i < range.end(); ++i) {
      if constexpr (std::is_void_v<WorkTag>) {
        (*functor)(i, ref);
      } else {
        (*functor)(WorkTag{}, i, ref);
      }
    }
  }
};

}  // namespace Kokkos::Impl

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP */
