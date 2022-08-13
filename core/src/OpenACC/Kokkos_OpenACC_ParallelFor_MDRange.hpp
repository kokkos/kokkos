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

#ifndef KOKKOS_OPENACC_PARALLEL_FOR_MDRANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_FOR_MDRANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::MDRangePolicy<Traits...>,
                                Kokkos::Experimental::OpenACC> {
  using Policy = Kokkos::MDRangePolicy<Traits...>;
  Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy> m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

  template <int Rank>
  std::enable_if_t<Rank == 2> execute_impl() const {
    auto const begin1 = m_policy.m_lower[1];
    auto const end1   = m_policy.m_upper[1];
    auto const begin0 = m_policy.m_lower[0];
    auto const end0   = m_policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1)) {
      return;
    }

    // avoid implicit capture of *this which would yield a memory access
    // violation when executing on the device
    auto const& functor(m_functor);

    int const async_arg = m_policy.space().acc_async_queue();

#pragma acc parallel loop gang vector collapse(2) copyin(functor) \
    async(async_arg)
    for (auto i1 = begin1; i1 < end1; ++i1) {
      for (auto i0 = begin0; i0 < end0; ++i0) {
        functor(i0, i1);
      }
    }
  }

  template <int Rank>
  std::enable_if_t<Rank == 3> execute_impl() const {
    auto const begin2 = m_policy.m_lower[2];
    auto const end2   = m_policy.m_upper[2];
    auto const begin1 = m_policy.m_lower[1];
    auto const end1   = m_policy.m_upper[1];
    auto const begin0 = m_policy.m_lower[0];
    auto const end0   = m_policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
      return;
    }

    // avoid implicit capture of *this which would yield a memory access
    // violation when executing on the device
    auto const& functor(m_functor);

    int const async_arg = m_policy.space().acc_async_queue();

#pragma acc parallel loop gang vector collapse(3) copyin(functor) \
    async(async_arg)
    for (auto i2 = begin2; i2 < end2; ++i2) {
      for (auto i1 = begin1; i1 < end1; ++i1) {
        for (auto i0 = begin0; i0 < end0; ++i0) {
          functor(i0, i1, i2);
        }
      }
    }
  }

  template <int Rank>
  std::enable_if_t<Rank == 4> execute_impl() const {
    auto const begin3 = m_policy.m_lower[3];
    auto const end3   = m_policy.m_upper[3];
    auto const begin2 = m_policy.m_lower[2];
    auto const end2   = m_policy.m_upper[2];
    auto const begin1 = m_policy.m_lower[1];
    auto const end1   = m_policy.m_upper[1];
    auto const begin0 = m_policy.m_lower[0];
    auto const end0   = m_policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3)) {
      return;
    }

    // avoid implicit capture of *this which would yield a memory access
    // violation when executing on the device
    auto const& functor(m_functor);

    int const async_arg = m_policy.space().acc_async_queue();

#pragma acc parallel loop gang vector collapse(4) copyin(functor) \
    async(async_arg)
    for (auto i3 = begin3; i3 < end3; ++i3) {
      for (auto i2 = begin2; i2 < end2; ++i2) {
        for (auto i1 = begin1; i1 < end1; ++i1) {
          for (auto i0 = begin0; i0 < end0; ++i0) {
            functor(i0, i1, i2, i3);
          }
        }
      }
    }
  }

  template <int Rank>
  std::enable_if_t<Rank == 5> execute_impl() const {
    auto const begin4 = m_policy.m_lower[4];
    auto const end4   = m_policy.m_upper[4];
    auto const begin3 = m_policy.m_lower[3];
    auto const end3   = m_policy.m_upper[3];
    auto const begin2 = m_policy.m_lower[2];
    auto const end2   = m_policy.m_upper[2];
    auto const begin1 = m_policy.m_lower[1];
    auto const end1   = m_policy.m_upper[1];
    auto const begin0 = m_policy.m_lower[0];
    auto const end0   = m_policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3) || (end4 <= begin4)) {
      return;
    }

    // avoid implicit capture of *this which would yield a memory access
    // violation when executing on the device
    auto const& functor(m_functor);

    int const async_arg = m_policy.space().acc_async_queue();

#pragma acc parallel loop gang vector collapse(5) copyin(functor) \
    async(async_arg)
    for (auto i4 = begin4; i4 < end4; ++i4) {
      for (auto i3 = begin3; i3 < end3; ++i3) {
        for (auto i2 = begin2; i2 < end2; ++i2) {
          for (auto i1 = begin1; i1 < end1; ++i1) {
            for (auto i0 = begin0; i0 < end0; ++i0) {
              functor(i0, i1, i2, i3, i4);
            }
          }
        }
      }
    }
  }

  template <int Rank>
  std::enable_if_t<Rank == 6> execute_impl() const {
    auto const begin5 = m_policy.m_lower[5];
    auto const end5   = m_policy.m_upper[5];
    auto const begin4 = m_policy.m_lower[4];
    auto const end4   = m_policy.m_upper[4];
    auto const begin3 = m_policy.m_lower[3];
    auto const end3   = m_policy.m_upper[3];
    auto const begin2 = m_policy.m_lower[2];
    auto const end2   = m_policy.m_upper[2];
    auto const begin1 = m_policy.m_lower[1];
    auto const end1   = m_policy.m_upper[1];
    auto const begin0 = m_policy.m_lower[0];
    auto const end0   = m_policy.m_upper[0];

    if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
        (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
      return;
    }

    // avoid implicit capture of *this which would yield a memory access
    // violation when executing on the device
    auto const& functor(m_functor);

    int const async_arg = m_policy.space().acc_async_queue();

#pragma acc parallel loop gang vector collapse(6) copyin(functor) \
    async(async_arg)
    for (auto i5 = begin5; i5 < end5; ++i5) {
      for (auto i4 = begin4; i4 < end4; ++i4) {
        for (auto i3 = begin3; i3 < end3; ++i3) {
          for (auto i2 = begin2; i2 < end2; ++i2) {
            for (auto i1 = begin1; i1 < end1; ++i1) {
              for (auto i0 = begin0; i0 < end0; ++i0) {
                functor(i0, i1, i2, i3, i4, i5);
              }
            }
          }
        }
      }
    }
  }

  void execute() const {
    static_assert(Policy::rank < 7,
                  "OpenACC Backend MDRangePolicy Error: Unsupported rank...");
    execute_impl<Policy::rank>();
  }
};

#endif
