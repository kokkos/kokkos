// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_SCAN_RANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Intrinsics.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ThreadSpaceGuard.hpp>
#include <Kokkos_Parallel.hpp>
#include <nextapi/parallelism.hpp>
#include <type_traits>
#include <mutex>

namespace Kokkos::Impl {

template <class Functor, class GivenValueType, class... Traits>
class ParallelScanNextSilicon {
 protected:
  using ExecSpace   = Kokkos::Experimental::NextSilicon;
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using WorkTag     = typename Policy::work_tag;
  using WorkRange   = typename Policy::WorkRange;
  using Analysis =
      Kokkos::Impl::FunctorAnalysis<Kokkos::Impl::FunctorPatternInterface::SCAN,
                                    Policy, Functor, GivenValueType>;
  using PointerType   = typename Analysis::pointer_type;
  using ReferenceType = typename Analysis::reference_type;
  using ValueType     = typename Analysis::value_type;
  using MemberType    = typename Policy::member_type;
  using IndexType     = typename Policy::index_type;

  // Each thread does part of input array
  constexpr static uint32_t MAX_PARTIAL_PROD = 64 * 1024;  // 64K

  // Let threads reduce the problem by at least factor of
  // MIN_ITER_PER_PARTIAL_PROD In case of smaller inputs.
  constexpr static uint32_t MIN_ITER_PER_PARTIAL_PROD = 100;

  Functor m_functor;
  Policy m_policy;
  typename Analysis::Reducer m_final_reducer;
  ValueType* m_result_ptr;

  inline static void scan_range(const Functor& functor, const WorkRange& range,
                                ValueType* update, const bool final) {
    const auto iend = range.end();

    if constexpr (std::is_void_v<WorkTag>) {
      for (auto iwork = range.begin(); iwork < iend; ++iwork) {
        functor(iwork, *update, final);
      }
    } else {
      const WorkTag t{};

      for (auto iwork = range.begin(); iwork < iend; ++iwork) {
        functor(t, iwork, *update, final);
      }
    }
  }

 public:
  ParallelScanNextSilicon(Functor const& arg_functor, Policy const& arg_policy,
                          ValueType* arg_result_ptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_final_reducer(arg_functor),
        m_result_ptr(arg_result_ptr) {}

  void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    const IndexType input_count = m_policy.end() - m_policy.begin() + 1;

    if (input_count == 0) return;

    // Calculate optimal number of partial products
    uint32_t num_partial_prod =
        (input_count + (MIN_ITER_PER_PARTIAL_PROD - 1)) /
        (MIN_ITER_PER_PARTIAL_PROD);
    num_partial_prod = std::min(num_partial_prod, MAX_PARTIAL_PROD);

    size_t partial_prod_byte_size =
        num_partial_prod * sizeof(ValueType) * m_final_reducer.value_count();

    // Retrieve the scratch buffer.
    auto internal_instance = m_policy.space().impl_internal_space_instance();
    std::byte* partial_buffer =
        internal_instance->resize_reduce_partial_buffer(partial_prod_byte_size);
    // Clone the driver before going into the handoff function.
    // This prevents the stack from getting migrated into device.
    auto cloned_driver = internal_instance->clone_driver(*this);
    ;

    cloned_driver->execute_internal(
        reinterpret_cast<ValueType*>(partial_buffer), num_partial_prod);
  }

 private:
  __attribute__((noinline)) void execute_internal(
      ValueType* partial_prod, uint32_t num_partial_prod) const {
    IndexType begin                       = 0;
    IndexType end                         = num_partial_prod;
    [[maybe_unused]] IndexType chunk_size = 1;

    nextapi::parallel_for(begin, end, microtask<false>, num_partial_prod,
                          &m_functor, &m_final_reducer, &m_policy,
                          partial_prod);

    scan_partial_prod(m_final_reducer, m_result_ptr, partial_prod,
                      num_partial_prod);

    nextapi::parallel_for(begin, end, microtask<true>, num_partial_prod,
                          &m_functor, &m_final_reducer, &m_policy,
                          partial_prod);
  }

  template <bool Final>
  static void microtask(IndexType thread_index, uint32_t num_partial_prod,
                        const Functor* functor,
                        const typename Analysis::Reducer* final_reducer,
                        const Policy* policy, ValueType* partial_prod) {
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

    auto value_count = final_reducer->value_count();
    const WorkRange range(*policy, thread_index, num_partial_prod);

    ValueType* update = partial_prod + thread_index * value_count;

    // The initial run calculates the reduction of each thread's range and
    // stores the result in the thread's partial_prod entry. The final run
    // uses the result of the scan of the partial products as the initial value.
    if constexpr (!Final) {
      final_reducer->init(update);
    }
    scan_range(*functor, range, update, Final);
  }

  static void scan_partial_prod(const typename Analysis::Reducer& final_reducer,
                                ValueType* result_ptr, ValueType* partial_prod,
                                uint32_t num_partial_prod) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    // inclusive scan of the partial products
    auto value_count = final_reducer.value_count();
    for (unsigned int i = 1; i < num_partial_prod; ++i) {
      final_reducer.join(&partial_prod[i * value_count],
                         &partial_prod[(i - 1) * value_count]);
    }

    // copy the last value to result_ptr (if not null)
    if (result_ptr) {
      for (unsigned int j = 0; j < value_count; ++j) {
        result_ptr[j] = partial_prod[(num_partial_prod - 1) * value_count + j];
      }
    }

    // convert inclusive scan to exclusive scan by shuffling to the right
    for (unsigned int i = num_partial_prod - 1; i > 0; --i) {
      for (unsigned int j = 0; j < value_count; ++j) {
        partial_prod[i * value_count + j] =
            partial_prod[(i - 1) * value_count + j];
      }
    }
    final_reducer.init(partial_prod);
  }
};

}  // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelScan<Functor, Kokkos::RangePolicy<Traits...>,
                                 Kokkos::Experimental::NextSilicon>
    : public ParallelScanNextSilicon<Functor, void, Traits...> {
  using base_t    = ParallelScanNextSilicon<Functor, void, Traits...>;
  using IndexType = typename base_t::IndexType;

 public:
  ParallelScan(const Functor& arg_functor,
               const typename base_t::Policy& arg_policy)
      : base_t(arg_functor, arg_policy, nullptr) {}
};

template <class FunctorType, class ReturnType, class... Traits>
class Kokkos::Impl::ParallelScanWithTotal<
    FunctorType, Kokkos::RangePolicy<Traits...>, ReturnType,
    Kokkos::Experimental::NextSilicon>
    : public ParallelScanNextSilicon<FunctorType, ReturnType, Traits...> {
  using base_t = ParallelScanNextSilicon<FunctorType, ReturnType, Traits...>;

 public:
  template <class ViewType>
  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const typename base_t::Policy& arg_policy,
                        const ViewType& arg_result_view)
      : base_t(arg_functor, arg_policy, arg_result_view.data()) {}
};

#endif
