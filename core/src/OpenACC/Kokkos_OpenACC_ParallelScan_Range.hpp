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

#ifndef KOKKOS_OPENACC_PARALLEL_SCAN_RANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_SCAN_RANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos::Experimental::Impl {
template <class IndexType, class Functor, class ValueType>
void OpenACCParallelScanRangePolicy(IndexType begin, IndexType end,
                                    IndexType chunk_size, Functor afunctor,
                                    ValueType init_value, int async_arg) {
  auto const functor(afunctor);
  const IndexType N        = end - begin;
  const IndexType n_chunks = (N + chunk_size - 1) / chunk_size;
  Kokkos::View<ValueType*, Kokkos::Experimental::OpenACC> chunk_values(
      "Kokkos::OpenACCParallelScan::chunk_values", n_chunks);
  Kokkos::View<ValueType*, Kokkos::Experimental::OpenACC> offset_values(
      "Kokkos::OpenACCParallelScan::offset_values", n_chunks);
  std::unique_ptr<ValueType[]> element_values_owner(
      new ValueType[2 * chunk_size]);
  ValueType* element_values = element_values_owner.get();

#pragma acc enter data copyin(functor) copyin(chunk_values, offset_values) \
    async(async_arg)

#pragma acc parallel loop gang vector_length(chunk_size) private( \
    element_values [0:2 * chunk_size]) present(functor, chunk_values)
  for (IndexType team_id = 0; team_id < n_chunks; ++team_id) {
#pragma acc loop vector
    for (IndexType thread_id = 0; thread_id < chunk_size; ++thread_id) {
      const IndexType local_offset = team_id * chunk_size;
      const IndexType idx          = local_offset + thread_id;
      ValueType update             = init_value;
      if ((idx > 0) && (idx < N)) functor(idx - 1, update, false);
      element_values[thread_id] = update;
    }
    IndexType current_step = 0;
    IndexType next_step    = 1;
    IndexType temp;
    for (IndexType step_size = 1; step_size < chunk_size; step_size *= 2) {
#pragma acc loop vector
      for (IndexType thread_id = 0; thread_id < chunk_size; ++thread_id) {
        if (thread_id < step_size) {
          element_values[next_step * chunk_size + thread_id] =
              element_values[current_step * chunk_size + thread_id];
        } else {
          element_values[next_step * chunk_size + thread_id] =
              element_values[current_step * chunk_size + thread_id] +
              element_values[current_step * chunk_size + thread_id - step_size];
        }
      }
      temp         = current_step;
      current_step = next_step;
      next_step    = temp;
    }
    chunk_values(team_id) =
        element_values[current_step * chunk_size + chunk_size - 1];
  }

  ValueType tempValue;
#pragma acc serial loop present(chunk_values, offset_values)
  for (IndexType team_id = 0; team_id < n_chunks; ++team_id) {
    if (team_id == 0) {
      offset_values(0) = 0;
      tempValue        = 0;
    } else {
      tempValue += chunk_values(team_id - 1);
      offset_values(team_id) = tempValue;
    }
  }

#pragma acc parallel loop gang vector_length(chunk_size) private( \
    element_values [0:2 * chunk_size]) present(functor, offset_values)
  for (IndexType team_id = 0; team_id < n_chunks; ++team_id) {
#pragma acc loop vector
    for (IndexType thread_id = 0; thread_id < chunk_size; ++thread_id) {
      const IndexType local_offset = team_id * chunk_size;
      const IndexType idx          = local_offset + thread_id;
      ValueType update             = init_value;
      if (thread_id == 0) {
        update += offset_values(team_id);
      }
      if ((idx > 0) && (idx < N)) functor(idx - 1, update, false);
      element_values[thread_id] = update;
    }
    IndexType current_step = 0;
    IndexType next_step    = 1;
    IndexType temp;
    for (IndexType step_size = 1; step_size < chunk_size; step_size *= 2) {
#pragma acc loop vector
      for (IndexType thread_id = 0; thread_id < chunk_size; ++thread_id) {
        if (thread_id < step_size) {
          element_values[next_step * chunk_size + thread_id] =
              element_values[current_step * chunk_size + thread_id];
        } else {
          element_values[next_step * chunk_size + thread_id] =
              element_values[current_step * chunk_size + thread_id] +
              element_values[current_step * chunk_size + thread_id - step_size];
        }
      }
      temp         = current_step;
      current_step = next_step;
      next_step    = temp;
    }
#pragma acc loop vector
    for (IndexType thread_id = 0; thread_id < chunk_size; ++thread_id) {
      const IndexType local_offset = team_id * chunk_size;
      const IndexType idx          = local_offset + thread_id;
      ValueType update = element_values[current_step * chunk_size + thread_id];
      if (idx < N) functor(idx, update, true);
    }
  }

#pragma acc exit data delete (functor, chunk_values, \
                              offset_values)async(async_arg)
}
}  // namespace Kokkos::Experimental::Impl

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelScan<Functor, Kokkos::RangePolicy<Traits...>,
                                 Kokkos::Experimental::OpenACC> {
  using Policy = Kokkos::RangePolicy<Traits...>;
  using Analysis =
      Kokkos::Impl::FunctorAnalysis<Kokkos::Impl::FunctorPatternInterface::SCAN,
                                    Policy, Functor>;
  using PointerType = typename Analysis::pointer_type;
  using ValueType   = typename Analysis::value_type;
  using MemberType  = typename Policy::member_type;
  Functor m_functor;
  Policy m_policy;
  PointerType m_result_ptr;
  static constexpr MemberType default_scan_chunk_size = 128;

 public:
  ParallelScan(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    auto chunk_size  = m_policy.chunk_size();

    if (end <= begin) {
      return;
    }

    if (chunk_size > 0) {
      if (!Impl::is_integral_power_of_two(chunk_size))
        Kokkos::abort(
            "RangePolicy blocking granularity must be power of two to be used "
            "with OpenACC parallel_scan()");
    } else {
      chunk_size = default_scan_chunk_size;
    }

    int const async_arg = m_policy.space().acc_async_queue();
    ValueType init_value;
    typename Analysis::Reducer final_reducer(&m_functor);
    final_reducer.init(&init_value);

    Kokkos::Experimental::Impl::OpenACCParallelScanRangePolicy(
        begin, end, chunk_size,
        Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>(m_functor),
        init_value, async_arg);
  }
};

#endif
