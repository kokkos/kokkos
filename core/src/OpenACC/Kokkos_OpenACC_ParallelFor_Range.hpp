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

#ifndef KOKKOS_OPENACC_PARALLEL_FOR_RANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_FOR_RANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <OpenACC/Kokkos_OpenACC_ScheduleType.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos::Experimental::Impl {
template <class IndexType, class Functor>
void OpenACCParallelForRangePolicy(Schedule<Static>, int chunk_size,
                                   IndexType begin, IndexType end,
                                   Functor functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; ++i) {
      functor(i);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; ++i) {
      functor(i);
    }
  }
}

template <class IndexType, class Functor>
void OpenACCParallelForRangePolicy(Schedule<Dynamic>, int /*chunk_size*/,
                                   IndexType begin, IndexType end,
                                   Functor functor, int async_arg) {
// clang-format off
#pragma acc parallel loop gang vector copyin(functor) async(async_arg)
  // clang-format on
  for (auto i = begin; i < end; ++i) {
    functor(i);
  }
}
}  // namespace Kokkos::Experimental::Impl

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::RangePolicy<Traits...>,
                                Kokkos::Experimental::OpenACC> {
  using Policy = Kokkos::RangePolicy<Traits...>;
  Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy> m_functor;
  Policy m_policy;
  using ScheduleType = Kokkos::Experimental::Impl::OpenACCScheduleType<Policy>;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    if (end <= begin) {
      return;
    }

    int const async_arg  = m_policy.space().acc_async_queue();
    int const chunk_size = m_policy.chunk_size();

    Kokkos::Experimental::Impl::OpenACCParallelForRangePolicy(
        ScheduleType(), chunk_size, begin, end, m_functor, async_arg);
  }
};

#endif
