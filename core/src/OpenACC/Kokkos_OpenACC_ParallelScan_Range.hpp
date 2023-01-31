/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_OPENACC_PARALLEL_SCAN_RANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_SCAN_RANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

#define DEFAULT_SCAN_CHUNK_SIZE 128

namespace Kokkos::Experimental::Impl {
template <class IndexType, class Functor, class ValueType>
void OpenACCParallelScanRangePolicy(IndexType begin, IndexType end,
                                    IndexType chunk_size, Functor afunctor,
                                    ValueType init_value, int async_arg) {
  auto const functor(afunctor);
  const IndexType N        = end - begin;
  const IndexType n_chunks = (N + chunk_size - 1) / chunk_size;
  Kokkos::View<ValueType*, Kokkos::Experimental::OpenACC> chunk_values(
      "chunk_values", n_chunks);
  Kokkos::View<ValueType*, Kokkos::Experimental::OpenACC> offset_values(
      "offset_values", n_chunks);
  ValueType* element_values = new ValueType[2 * chunk_size];

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
  Functor m_functor;
  Policy m_policy;
  PointerType m_result_ptr;

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
            "for parallel_scan()");
    } else {
      chunk_size = DEFAULT_SCAN_CHUNK_SIZE;
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
