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

#ifndef KOKKOS_SYCL_PARALLEL_REDUCE_HPP
#define KOKKOS_SYCL_PARALLEL_REDUCE_HPP

#include <Kokkos_Macros.hpp>

#include <vector>
#if defined(KOKKOS_ENABLE_SYCL)
#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_BitManipulation.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

// FIXME_SYCL It appears that using shuffles is slower than going through local
// memory.
template <class ReducerType>
inline constexpr bool use_shuffle_based_algorithm = false;
// std::is_reference_v<typename ReducerType::reference_type>;

namespace SYCLReduction {
template <typename ValueType, typename ReducerType, int dim>
std::enable_if_t<!use_shuffle_based_algorithm<ReducerType>> workgroup_reduction(
    sycl::nd_item<dim>& item, sycl::local_accessor<ValueType> local_mem,
    sycl::device_ptr<ValueType> results_ptr,
    sycl::global_ptr<ValueType> device_accessible_result_ptr,
    const unsigned int value_count_, const ReducerType& final_reducer,
    bool final, unsigned int max_size) {
  const unsigned int value_count =
      std::is_reference_v<typename ReducerType::reference_type> ? 1
                                                                : value_count_;
  const int local_id = item.get_local_linear_id();

  // Perform the actual workgroup reduction in each subgroup
  // separately.
  auto sg            = item.get_sub_group();
  auto* result       = &local_mem[local_id * value_count];
  const int id_in_sg = sg.get_local_id()[0];
  const auto local_range =
      std::min<unsigned int>(sg.get_local_range()[0], max_size);
  const auto upper_stride_bound =
      std::min<unsigned int>(local_range - id_in_sg, max_size - local_id);
  for (unsigned int stride = 1; stride < local_range; stride <<= 1) {
    if (stride < upper_stride_bound)
      final_reducer.join(result, &local_mem[(local_id + stride) * value_count]);
    sycl::group_barrier(sg);
  }
  sycl::group_barrier(item.get_group());

  // Do the final reduction only using the first subgroup.
  if (sg.get_group_id()[0] == 0) {
    const unsigned int n_subgroups = sg.get_group_range()[0];
    const int max_subgroup_size    = sg.get_max_local_range()[0];
    auto* result_ = &local_mem[id_in_sg * max_subgroup_size * value_count];
    // In case the number of subgroups is larger than the range of
    // the first subgroup, we first combine the items with a higher
    // index.
    for (unsigned int offset = local_range; offset < n_subgroups;
         offset += local_range)
      if (id_in_sg + offset < n_subgroups)
        final_reducer.join(
            result_,
            &local_mem[(id_in_sg + offset) * max_subgroup_size * value_count]);
    sycl::group_barrier(sg);

    // Then, we proceed as before.
    for (unsigned int stride = 1; stride < local_range; stride <<= 1) {
      if (id_in_sg + stride < n_subgroups)
        final_reducer.join(
            result_,
            &local_mem[(id_in_sg + stride) * max_subgroup_size * value_count]);
      sycl::group_barrier(sg);
    }

    // Finally, we copy the workgroup results back to global memory
    // to be used in the next iteration. If this is the last
    // iteration, i.e., there is only one workgroup also call
    // final() if necessary.
    if (id_in_sg == 0) {
      if (final) {
        final_reducer.final(&local_mem[0]);
        if (device_accessible_result_ptr != nullptr)
          final_reducer.copy(&device_accessible_result_ptr[0], &local_mem[0]);
        else
          final_reducer.copy(&results_ptr[0], &local_mem[0]);
      } else
        final_reducer.copy(
            &results_ptr[(item.get_group_linear_id()) * value_count],
            &local_mem[0]);
    }
  }
}

template <typename ValueType, typename ReducerType, int dim>
std::enable_if_t<use_shuffle_based_algorithm<ReducerType>> workgroup_reduction(
    sycl::nd_item<dim>& item, sycl::local_accessor<ValueType> local_mem,
    ValueType local_value, sycl::device_ptr<ValueType> results_ptr,
    sycl::global_ptr<ValueType> device_accessible_result_ptr,
    const ReducerType& final_reducer, bool final, unsigned int max_size) {
  const auto local_id = item.get_local_linear_id();

  // Perform the actual workgroup reduction in each subgroup
  // separately.
  auto sg            = item.get_sub_group();
  const int id_in_sg = sg.get_local_id()[0];
  const auto local_range =
      std::min<unsigned int>(sg.get_local_range()[0], max_size);

  const auto upper_stride_bound =
      std::min<unsigned int>(local_range - id_in_sg, max_size - local_id);
  for (unsigned int stride = 1; stride < local_range; stride <<= 1) {
    auto tmp = sg.shuffle_down(local_value, stride);
    if (stride < upper_stride_bound) final_reducer.join(&local_value, &tmp);
  }

  // Copy the subgroup results into the first positions of the
  // reduction array.
  const int max_subgroup_size = sg.get_max_local_range()[0];
  const unsigned int n_active_subgroups =
      (max_size + max_subgroup_size - 1) / max_subgroup_size;
  const int sg_group_id = sg.get_group_id()[0];
  if (id_in_sg == 0 && sg_group_id <= static_cast<int>(n_active_subgroups))
    local_mem[sg_group_id] = local_value;

  item.barrier(sycl::access::fence_space::local_space);

  // Do the final reduction only using the first subgroup.
  if (sg.get_group_id()[0] == 0) {
    auto sg_value =
        local_mem[id_in_sg < static_cast<int>(n_active_subgroups) ? id_in_sg
                                                                  : 0];

    // In case the number of subgroups is larger than the range of
    // the first subgroup, we first combine the items with a higher
    // index.
    if (n_active_subgroups > local_range) {
      for (unsigned int offset = local_range; offset < n_active_subgroups;
           offset += local_range)
        if (id_in_sg + offset < n_active_subgroups) {
          final_reducer.join(&sg_value, &local_mem[(id_in_sg + offset)]);
        }
      sg.barrier();
    }

    // Then, we proceed as before.
    for (unsigned int stride = 1; stride < local_range; stride <<= 1) {
      auto tmp = sg.shuffle_down(sg_value, stride);
      if (id_in_sg + stride < n_active_subgroups)
        final_reducer.join(&sg_value, &tmp);
    }

    // Finally, we copy the workgroup results back to global memory
    // to be used in the next iteration. If this is the last
    // iteration, i.e., there is only one workgroup also call
    // final() if necessary.
    if (id_in_sg == 0) {
      if (final) {
        final_reducer.final(&sg_value);
        if (device_accessible_result_ptr != nullptr)
          device_accessible_result_ptr[0] = sg_value;
        else
          results_ptr[0] = sg_value;
      } else
        results_ptr[(item.get_group_linear_id())] = sg_value;
    }
  }
}

}  // namespace SYCLReduction

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, Kokkos::RangePolicy<Traits...>,
                     Kokkos::Experimental::SYCL> {
 public:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

 private:
  using value_type     = typename ReducerType::value_type;
  using pointer_type   = typename ReducerType::pointer_type;
  using reference_type = typename ReducerType::reference_type;

  using WorkTag = typename Policy::work_tag;

 public:
  // V - View
  template <typename View>
  ParallelReduce(const CombinedFunctorReducerType& f, const Policy& p,
                 const View& v)
      : m_functor_reducer(f),
        m_policy(p),
        m_result_ptr(v.data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename View::memory_space>::accessible),
        m_shared_memory_lock(
            p.space().impl_internal_space_instance()->m_mutexScratchSpace) {}

 private:
  template <typename PolicyType, typename CombinedFunctorReducerWrapper>
  sycl::event sycl_direct_launch(
      const PolicyType& policy,
      const CombinedFunctorReducerWrapper& functor_reducer_wrapper,
      const sycl::event& memcpy_event) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = space.sycl_queue();

    std::size_t size = policy.end() - policy.begin();
    const unsigned int value_count =
        std::max(m_functor_reducer.get_reducer().value_count(), 1u);
    sycl::device_ptr<value_type> results_ptr = nullptr;
    sycl::global_ptr<value_type> device_accessible_result_ptr =
        m_result_ptr_device_accessible ? m_result_ptr : nullptr;

    sycl::event last_reduction_event;

    // If size<=1 we only call init(), the functor and possibly final once
    // working with the global scratch memory but don't copy back to
    // m_result_ptr yet.
    if (size <= 1) {
      results_ptr = static_cast<sycl::device_ptr<value_type>>(
          instance.scratch_space(sizeof(value_type) * value_count));

      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
        const auto begin = policy.begin();
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
        cgh.depends_on(memcpy_event);
#else
        (void)memcpy_event;
#endif
        cgh.single_task([=]() {
          const CombinedFunctorReducerType& functor_reducer =
              functor_reducer_wrapper.get_functor();
          const FunctorType& functor = functor_reducer.get_functor();
          const ReducerType& reducer = functor_reducer.get_reducer();
          reference_type update      = reducer.init(results_ptr);
          if (size == 1) {
            if constexpr (std::is_void_v<WorkTag>)
              functor(begin, update);
            else
              functor(WorkTag(), begin, update);
          }
          reducer.final(results_ptr);
          if (device_accessible_result_ptr != nullptr)
            reducer.copy(device_accessible_result_ptr.get(), results_ptr.get());
        });
      });
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
      q.ext_oneapi_submit_barrier(
          std::vector<sycl::event>{parallel_reduce_event});
#endif
      last_reduction_event = parallel_reduce_event;
    }

    // Otherwise, we perform a reduction on the values in all workgroups
    // separately, write the workgroup results back to global memory and recurse
    // until only one workgroup does the reduction and thus gets the final
    // value.
    if (size > 1) {
      auto scratch_flags = static_cast<sycl::device_ptr<unsigned int>>(
          instance.scratch_flags(sizeof(unsigned int)));

      auto reduction_lambda_factory =
          [&](sycl::local_accessor<value_type> local_mem,
              sycl::local_accessor<unsigned int> num_teams_done,
              sycl::device_ptr<value_type> results_ptr, int values_per_thread) {
            const auto begin = policy.begin();

            auto lambda = [=](sycl::nd_item<1> item) {
              const auto n_wgroups   = item.get_group_range()[0];
              const auto wgroup_size = item.get_local_range()[0];

              const auto local_id = item.get_local_linear_id();
              const auto global_id =
                  wgroup_size * item.get_group_linear_id() * values_per_thread +
                  local_id;
              const CombinedFunctorReducerType& functor_reducer =
                  functor_reducer_wrapper.get_functor();
              const FunctorType& functor = functor_reducer.get_functor();
              const ReducerType& reducer = functor_reducer.get_reducer();

              using index_type       = typename Policy::index_type;
              const auto upper_bound = std::min<index_type>(
                  global_id + values_per_thread * wgroup_size, size);

              if constexpr (!use_shuffle_based_algorithm<ReducerType>) {
                reference_type update =
                    reducer.init(&local_mem[local_id * value_count]);
                for (index_type id = global_id; id < upper_bound;
                     id += wgroup_size) {
                  if constexpr (std::is_void_v<WorkTag>)
                    functor(id + begin, update);
                  else
                    functor(WorkTag(), id + begin, update);
                }
                item.barrier(sycl::access::fence_space::local_space);

                SYCLReduction::workgroup_reduction<>(
                    item, local_mem, results_ptr, device_accessible_result_ptr,
                    value_count, reducer, false, std::min(size, wgroup_size));

                if (local_id == 0) {
                  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                      scratch_flags_ref(*scratch_flags);
                  num_teams_done[0] = ++scratch_flags_ref;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (num_teams_done[0] == n_wgroups) {
                  if (local_id >= n_wgroups)
                    reducer.init(&local_mem[local_id * value_count]);
                  else {
                    reducer.copy(&local_mem[local_id * value_count],
                                 &results_ptr[local_id * value_count]);
                    for (unsigned int id = local_id + wgroup_size;
                         id < n_wgroups; id += wgroup_size) {
                      reducer.join(&local_mem[local_id * value_count],
                                   &results_ptr[id * value_count]);
                    }
                  }

                  SYCLReduction::workgroup_reduction<>(
                      item, local_mem, results_ptr,
                      device_accessible_result_ptr, value_count, reducer, true,
                      std::min(n_wgroups, wgroup_size));
                }
              } else {
                value_type local_value;
                reference_type update = reducer.init(&local_value);
                for (index_type id = global_id; id < upper_bound;
                     id += wgroup_size) {
                  if constexpr (std::is_void_v<WorkTag>)
                    functor(id + begin, update);
                  else
                    functor(WorkTag(), id + begin, update);
                }

                SYCLReduction::workgroup_reduction<>(
                    item, local_mem, local_value, results_ptr,
                    device_accessible_result_ptr, reducer, false,
                    std::min(size, wgroup_size));

                if (local_id == 0) {
                  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                      scratch_flags_ref(*scratch_flags);
                  num_teams_done[0] = ++scratch_flags_ref;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (num_teams_done[0] == n_wgroups) {
                  if (local_id >= n_wgroups)
                    reducer.init(&local_value);
                  else {
                    local_value = results_ptr[local_id];
                    for (unsigned int id = local_id + wgroup_size;
                         id < n_wgroups; id += wgroup_size) {
                      reducer.join(&local_value, &results_ptr[id]);
                    }
                  }

                  SYCLReduction::workgroup_reduction<>(
                      item, local_mem, local_value, results_ptr,
                      device_accessible_result_ptr, reducer, true,
                      std::min(n_wgroups, wgroup_size));
                }
              }
            };
            return lambda;
          };

      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<unsigned int> num_teams_done(1, cgh);

        auto dummy_reduction_lambda =
            reduction_lambda_factory({1, cgh}, num_teams_done, nullptr, 1);

        static sycl::kernel kernel = [&] {
          sycl::kernel_id functor_kernel_id =
              sycl::get_kernel_id<decltype(dummy_reduction_lambda)>();
          auto kernel_bundle =
              sycl::get_kernel_bundle<sycl::bundle_state::executable>(
                  q.get_context(), std::vector{functor_kernel_id});
          return kernel_bundle.get_kernel(functor_kernel_id);
        }();
        auto multiple = kernel.get_info<sycl::info::kernel_device_specific::
                                            preferred_work_group_size_multiple>(
            q.get_device());
        // FIXME_SYCL The code below queries the kernel for the maximum subgroup
        // size but it turns out that this is not accurate and choosing a larger
        // subgroup size gives better peformance (and is what the SYCL compiler
        // does for sycl::reductions).
#ifndef KOKKOS_ARCH_INTEL_GPU
        auto max =
            kernel
                .get_info<sycl::info::kernel_device_specific::work_group_size>(
                    q.get_device());
        // FIXME_SYCL 1024 seems to be invalid when running on a Volta70.
        if (max > 512) max = 512;
#else
        auto max =
            q.get_device().get_info<sycl::info::device::max_work_group_size>();
#endif

        auto max_local_memory =
            q.get_device().get_info<sycl::info::device::local_mem_size>();
        // Use a maximum of 99% of the available memory as a safe-guard.
        const auto wgroup_size = std::min(
            {Kokkos::bit_ceil(size),
             static_cast<size_t>(max / multiple) * multiple,
             Kokkos::bit_floor(static_cast<size_t>(max_local_memory * .99) /
                               (sizeof(value_type) * value_count))});

        // FIXME_SYCL Find a better way to determine a good limit for the
        // maximum number of work groups, also see
        // https://github.com/intel/llvm/blob/756ba2616111235bba073e481b7f1c8004b34ee6/sycl/source/detail/reduction.cpp#L51-L62
        size_t max_work_groups =
            2 *
            q.get_device().get_info<sycl::info::device::max_compute_units>();
        int values_per_thread = 1;
        size_t n_wgroups      = (size + wgroup_size - 1) / wgroup_size;
        while (n_wgroups > max_work_groups) {
          values_per_thread *= 2;
          n_wgroups = ((size + values_per_thread - 1) / values_per_thread +
                       wgroup_size - 1) /
                      wgroup_size;
        }

        results_ptr =
            static_cast<sycl::device_ptr<value_type>>(instance.scratch_space(
                sizeof(value_type) * value_count * n_wgroups));

        sycl::local_accessor<value_type> local_mem(
            sycl::range<1>(wgroup_size) * value_count, cgh);

#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
        cgh.depends_on(memcpy_event);
#else
        (void)memcpy_event;
#endif

        auto reduction_lambda = reduction_lambda_factory(
            local_mem, num_teams_done, results_ptr, values_per_thread);

        cgh.parallel_for(
            sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
            reduction_lambda);
      });
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
      q.ext_oneapi_submit_barrier(
          std::vector<sycl::event>{parallel_reduce_event});
#endif
      last_reduction_event = parallel_reduce_event;
    }

    // At this point, the reduced value is written to the entry in results_ptr
    // and all that is left is to copy it back to the given result pointer if
    // necessary.
    if (m_result_ptr && !m_result_ptr_device_accessible) {
      Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
                             Kokkos::Experimental::SYCLDeviceUSMSpace>(
          space, m_result_ptr, results_ptr,
          sizeof(*m_result_ptr) * value_count);
    }

    return last_reduction_event;
  }

 public:
  void execute() const {
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *m_policy.space().impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem = instance.get_indirect_kernel_mem();

    auto functor_reducer_wrapper =
        Experimental::Impl::make_sycl_function_wrapper(m_functor_reducer,
                                                       indirectKernelMem);

    sycl::event event =
        sycl_direct_launch(m_policy, functor_reducer_wrapper,
                           functor_reducer_wrapper.get_copy_event());
    functor_reducer_wrapper.register_event(event);
  }

 private:
  const CombinedFunctorReducerType m_functor_reducer;
  const Policy m_policy;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;

  // Only let one Parallel/Scan modify the shared memory. The
  // constructor acquires the mutex which is released in the destructor.
  std::scoped_lock<std::mutex> m_shared_memory_lock;
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     Kokkos::MDRangePolicy<Traits...>,
                     Kokkos::Experimental::SYCL> {
 public:
  using Policy      = Kokkos::MDRangePolicy<Traits...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

 private:
  using value_type     = typename ReducerType::value_type;
  using pointer_type   = typename ReducerType::pointer_type;
  using reference_type = typename ReducerType::reference_type;

  using WorkTag = typename Policy::work_tag;

  // MDRangePolicy is not trivially copyable. Hence, replicate the data we
  // really need in DeviceIterateTile in a trivially copyable struct.
  struct BarePolicy {
    using index_type = typename Policy::index_type;

    BarePolicy(const Policy& policy)
        : m_lower(policy.m_lower),
          m_upper(policy.m_upper),
          m_tile(policy.m_tile),
          m_tile_end(policy.m_tile_end),
          m_num_tiles(policy.m_num_tiles),
          m_prod_tile_dims(policy.m_prod_tile_dims) {}

    const typename Policy::point_type m_lower;
    const typename Policy::point_type m_upper;
    const typename Policy::tile_type m_tile;
    const typename Policy::point_type m_tile_end;
    const typename Policy::index_type m_num_tiles;
    const typename Policy::index_type m_prod_tile_dims;
    static constexpr Iterate inner_direction = Policy::inner_direction;
    static constexpr int rank                = Policy::rank;
  };

 public:
  // V - View
  template <typename View>
  ParallelReduce(const CombinedFunctorReducerType& f, const Policy& p,
                 const View& v)
      : m_functor_reducer(f),
        m_policy(p),
        m_space(p.space()),
        m_result_ptr(v.data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename View::memory_space>::accessible),
        m_shared_memory_lock(
            m_space.impl_internal_space_instance()->m_mutexScratchSpace) {}

 private:
  template <typename CombinedFunctorReducerWrapper>
  sycl::event sycl_direct_launch(
      const CombinedFunctorReducerWrapper& functor_reducer_wrapper,
      const sycl::event& memcpy_event) const {
    // Convenience references
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *m_space.impl_internal_space_instance();
    sycl::queue& q = m_space.sycl_queue();

    const typename Policy::index_type n_tiles = m_policy.m_num_tiles;
    const unsigned int value_count =
        std::max(m_functor_reducer.get_reducer().value_count(), 1u);
    sycl::device_ptr<value_type> results_ptr;

    sycl::event last_reduction_event;

    // If n_tiles==0 we only call init() and final() working with the global
    // scratch memory but don't copy back to m_result_ptr yet.
    if (n_tiles == 0) {
      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
        cgh.depends_on(memcpy_event);
#else
        (void)memcpy_event;
#endif
        results_ptr = static_cast<sycl::device_ptr<value_type>>(
            instance.scratch_space(sizeof(value_type) * value_count));
        sycl::global_ptr<value_type> device_accessible_result_ptr =
            m_result_ptr_device_accessible ? m_result_ptr : nullptr;
        cgh.single_task([=]() {
          const CombinedFunctorReducerType& functor_reducer =
              functor_reducer_wrapper.get_functor();
          const ReducerType& reducer = functor_reducer.get_reducer();
          reducer.init(results_ptr);
          reducer.final(results_ptr);
          if (device_accessible_result_ptr)
            reducer.copy(device_accessible_result_ptr.get(), results_ptr.get());
        });
      });
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
      q.ext_oneapi_submit_barrier(
          std::vector<sycl::event>{parallel_reduce_event});
#endif
      last_reduction_event = parallel_reduce_event;
    }

    // Otherwise, we perform a reduction on the values in all workgroups
    // separately, write the workgroup results back to global memory and recurse
    // until only one workgroup does the reduction and thus gets the final
    // value.
    if (n_tiles > 0) {
      const int wgroup_size = Kokkos::bit_ceil(
          static_cast<unsigned int>(m_policy.m_prod_tile_dims));

      // FIXME_SYCL Find a better way to determine a good limit for the
      // maximum number of work groups, also see
      // https://github.com/intel/llvm/blob/756ba2616111235bba073e481b7f1c8004b34ee6/sycl/source/detail/reduction.cpp#L51-L62
      size_t max_work_groups =
          2 * q.get_device().get_info<sycl::info::device::max_compute_units>();
      int values_per_thread = 1;
      size_t n_wgroups      = n_tiles;
      while (n_wgroups > max_work_groups) {
        values_per_thread *= 2;
        n_wgroups = (n_tiles + values_per_thread - 1) / values_per_thread;
      }

      results_ptr = static_cast<sycl::device_ptr<value_type>>(
          instance.scratch_space(sizeof(value_type) * value_count * n_wgroups));
      sycl::global_ptr<value_type> device_accessible_result_ptr =
          m_result_ptr_device_accessible ? m_result_ptr : nullptr;
      auto scratch_flags = static_cast<sycl::device_ptr<unsigned int>>(
          instance.scratch_flags(sizeof(unsigned int)));

      auto parallel_reduce_event = q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<value_type> local_mem(
            sycl::range<1>(wgroup_size) * value_count, cgh);
        sycl::local_accessor<unsigned int> num_teams_done(1, cgh);

        const BarePolicy bare_policy = m_policy;

#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
        cgh.depends_on(memcpy_event);
#else
        (void)memcpy_event;
#endif

        // REMEMBER swap local x<->y to be conforming with Cuda/HIP
        // implementation
        cgh.parallel_for(
            sycl::nd_range<1>{n_wgroups * wgroup_size, wgroup_size},
            [=](sycl::nd_item<1> item) {
              const int local_id = item.get_local_linear_id();
              const CombinedFunctorReducerType& functor_reducer =
                  functor_reducer_wrapper.get_functor();
              const FunctorType& functor = functor_reducer.get_functor();
              const ReducerType& reducer = functor_reducer.get_reducer();

              // In the first iteration, we call functor to initialize the local
              // memory. Otherwise, the local memory is initialized with the
              // results from the previous iteration that are stored in global
              // memory.
              using index_type = typename Policy::index_type;

              // SWAPPED here to be conforming with CUDA implementation
              const index_type local_x    = 0;
              const index_type local_y    = item.get_local_id(0);
              const index_type local_z    = 0;
              const index_type global_y   = 0;
              const index_type global_z   = 0;
              const index_type n_global_x = n_tiles;
              const index_type n_global_y = 1;
              const index_type n_global_z = 1;

              if constexpr (!use_shuffle_based_algorithm<ReducerType>) {
                reference_type update =
                    reducer.init(&local_mem[local_id * value_count]);

                for (index_type global_x = item.get_group(0);
                     global_x < n_tiles; global_x += item.get_group_range(0))
                  Kokkos::Impl::Reduce::DeviceIterateTile<
                      Policy::rank, BarePolicy, FunctorType,
                      typename Policy::work_tag, reference_type>(
                      bare_policy, functor, update,
                      {n_global_x, n_global_y, n_global_z},
                      {global_x, global_y, global_z},
                      {local_x, local_y, local_z})
                      .exec_range();
                item.barrier(sycl::access::fence_space::local_space);

                SYCLReduction::workgroup_reduction<>(
                    item, local_mem, results_ptr, device_accessible_result_ptr,
                    value_count, reducer, false, n_wgroups * wgroup_size);

                if (local_id == 0) {
                  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                      scratch_flags_ref(*scratch_flags);
                  num_teams_done[0] = ++scratch_flags_ref;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (num_teams_done[0] == n_wgroups) {
                  if (local_id >= static_cast<int>(n_wgroups))
                    reducer.init(&local_mem[local_id * value_count]);
                  else {
                    reducer.copy(&local_mem[local_id * value_count],
                                 &results_ptr[local_id * value_count]);
                    for (unsigned int id = local_id + wgroup_size;
                         id < n_wgroups; id += wgroup_size) {
                      reducer.join(&local_mem[local_id * value_count],
                                   &results_ptr[id * value_count]);
                    }
                  }

                  SYCLReduction::workgroup_reduction<>(
                      item, local_mem, results_ptr,
                      device_accessible_result_ptr, value_count, reducer, true,
                      std::min<int>(n_wgroups, wgroup_size));
                }
              } else {
                value_type local_value;
                reference_type update = reducer.init(&local_value);

                for (index_type global_x = item.get_group(0);
                     global_x < n_tiles; global_x += item.get_group_range(0))
                  Kokkos::Impl::Reduce::DeviceIterateTile<
                      Policy::rank, BarePolicy, FunctorType,
                      typename Policy::work_tag, reference_type>(
                      bare_policy, functor, update,
                      {n_global_x, n_global_y, n_global_z},
                      {global_x, global_y, global_z},
                      {local_x, local_y, local_z})
                      .exec_range();

                SYCLReduction::workgroup_reduction<>(
                    item, local_mem, local_value, results_ptr,
                    device_accessible_result_ptr, reducer, false,
                    n_wgroups * wgroup_size);

                if (local_id == 0) {
                  sycl::atomic_ref<unsigned, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                      scratch_flags_ref(*scratch_flags);
                  num_teams_done[0] = ++scratch_flags_ref;
                }
                item.barrier(sycl::access::fence_space::local_space);
                if (num_teams_done[0] == n_wgroups) {
                  if (local_id >= static_cast<int>(n_wgroups))
                    reducer.init(&local_value);
                  else {
                    local_value = results_ptr[local_id];
                    for (unsigned int id = local_id + wgroup_size;
                         id < n_wgroups; id += wgroup_size) {
                      reducer.join(&local_value, &results_ptr[id]);
                    }
                  }

                  SYCLReduction::workgroup_reduction<>(
                      item, local_mem, local_value, results_ptr,
                      device_accessible_result_ptr, reducer, true,
                      std::min<int>(n_wgroups, wgroup_size));
                }
              }
            });
      });
#ifndef KOKKOS_IMPL_SYCL_USE_IN_ORDER_QUEUES
      q.ext_oneapi_submit_barrier(
          std::vector<sycl::event>{parallel_reduce_event});
#endif
      last_reduction_event = parallel_reduce_event;
    }

    // At this point, the reduced value is written to the entry in results_ptr
    // and all that is left is to copy it back to the given result pointer if
    // necessary.
    if (m_result_ptr && !m_result_ptr_device_accessible) {
      Kokkos::Impl::DeepCopy<Kokkos::Experimental::SYCLDeviceUSMSpace,
                             Kokkos::Experimental::SYCLDeviceUSMSpace>(
          m_space, m_result_ptr, results_ptr,
          sizeof(*m_result_ptr) * value_count);
    }

    return last_reduction_event;
  }

 public:
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy& policy, const Functor&) {
    return policy.space().impl_internal_space_instance()->m_maxWorkgroupSize;
  }

  void execute() const {
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *m_space.impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem = instance.get_indirect_kernel_mem();

    auto functor_reducer_wrapper =
        Experimental::Impl::make_sycl_function_wrapper(m_functor_reducer,
                                                       indirectKernelMem);

    sycl::event event = sycl_direct_launch(
        functor_reducer_wrapper, functor_reducer_wrapper.get_copy_event());
    functor_reducer_wrapper.register_event(event);
  }

 private:
  const CombinedFunctorReducerType m_functor_reducer;
  const BarePolicy m_policy;
  const Kokkos::Experimental::SYCL& m_space;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;

  // Only let one Parallel/Scan modify the shared memory. The
  // constructor acquires the mutex which is released in the destructor.
  std::scoped_lock<std::mutex> m_shared_memory_lock;
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* KOKKOS_SYCL_PARALLEL_REDUCE_HPP */
