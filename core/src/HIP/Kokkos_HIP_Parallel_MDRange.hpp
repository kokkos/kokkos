/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_HIP_PARALLEL_MDRANGE_HPP
#define KOKKOS_HIP_PARALLEL_MDRANGE_HPP

#include <HIP/Kokkos_HIP_KernelLaunch.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos {
namespace Impl {
// MDRangePolicy impl
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::HIP> {
 public:
  typedef Kokkos::MDRangePolicy<Traits...> Policy;

 private:
  using RP = Policy;
  typedef typename Policy::array_index_type array_index_type;
  typedef typename Policy::index_type index_type;
  typedef typename Policy::launch_bounds LaunchBounds;

  const FunctorType m_functor;
  const Policy m_rp;

 public:
  inline __device__ void operator()(void) const {
    // FIXME to implement
    //  Kokkos::Impl::Refactor::DeviceIterateTile<Policy::rank, Policy,
    //  FunctorType,
    //                                            typename Policy::work_tag>(
    //      m_rp, m_functor)
    //      .exec_range();
  }

  inline void execute() const {
    // FIXME to implement
    //  if (m_rp.m_num_tiles == 0)
    //    return;
    //  const array_index_type maxblocks = static_cast<array_index_type>(
    //      m_rp.space().impl_internal_space_instance()->m_maxBlock);
    //  if (RP::rank == 2)
    //  {
    //    const dim3 block(m_rp.m_tile[0], m_rp.m_tile[1], 1);
    //    const dim3 grid(
    //        std::min((m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1) /
    //        block.x,
    //                 maxblocks),
    //        std::min((m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1) /
    //        block.y,
    //                 maxblocks),
    //        1);
    //    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor,
    //    LaunchBounds>(
    //        *this, grid, block, 0,
    //        m_rp.space().impl_internal_space_instance(), false);
    //  }
    //  else if (RP::rank == 3)
    //  {
    //    const dim3 block(m_rp.m_tile[0], m_rp.m_tile[1], m_rp.m_tile[2]);
    //    const dim3 grid(
    //        std::min((m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1) /
    //        block.x,
    //                 maxblocks),
    //        std::min((m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1) /
    //        block.y,
    //                 maxblocks),
    //        std::min((m_rp.m_upper[2] - m_rp.m_lower[2] + block.z - 1) /
    //        block.z,
    //                 maxblocks));
    //    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor,
    //    LaunchBounds>(
    //        *this, grid, block, 0,
    //        m_rp.space().impl_internal_space_instance(), false);
    //  }
    //  else if (RP::rank == 4)
    //  {
    //    // id0,id1 encoded within threadIdx.x; id2 to threadIdx.y; id3 to
    //    // threadIdx.z
    //    const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1], m_rp.m_tile[2],
    //                     m_rp.m_tile[3]);
    //    const dim3 grid(
    //        std::min(
    //            static_cast<index_type>(m_rp.m_tile_end[0] *
    //            m_rp.m_tile_end[1]), static_cast<index_type>(maxblocks)),
    //        std::min((m_rp.m_upper[2] - m_rp.m_lower[2] + block.y - 1) /
    //        block.y,
    //                 maxblocks),
    //        std::min((m_rp.m_upper[3] - m_rp.m_lower[3] + block.z - 1) /
    //        block.z,
    //                 maxblocks));
    //    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor,
    //    LaunchBounds>(
    //        *this, grid, block, 0,
    //        m_rp.space().impl_internal_space_instance(), false);
    //  }
    //  else if (RP::rank == 5)
    //  {
    //    // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y; id4 to
    //    // threadIdx.z
    //    const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1],
    //                     m_rp.m_tile[2] * m_rp.m_tile[3], m_rp.m_tile[4]);
    //    const dim3 grid(
    //        std::min(
    //            static_cast<index_type>(m_rp.m_tile_end[0] *
    //            m_rp.m_tile_end[1]), static_cast<index_type>(maxblocks)),
    //        std::min(
    //            static_cast<index_type>(m_rp.m_tile_end[2] *
    //            m_rp.m_tile_end[3]), static_cast<index_type>(maxblocks)),
    //        std::min((m_rp.m_upper[4] - m_rp.m_lower[4] + block.z - 1) /
    //        block.z,
    //                 maxblocks));
    //    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor,
    //    LaunchBounds>(
    //        *this, grid, block, 0,
    //        m_rp.space().impl_internal_space_instance(), false);
    //  }
    //  else if (RP::rank == 6)
    //  {
    //    // id0,id1 encoded within threadIdx.x; id2,id3 to threadIdx.y; id4,id5
    //    to
    //    // threadIdx.z
    //    const dim3 block(m_rp.m_tile[0] * m_rp.m_tile[1],
    //                     m_rp.m_tile[2] * m_rp.m_tile[3],
    //                     m_rp.m_tile[4] * m_rp.m_tile[5]);
    //    const dim3 grid(std::min(static_cast<index_type>(m_rp.m_tile_end[0] *
    //                                                     m_rp.m_tile_end[1]),
    //                             static_cast<index_type>(maxblocks)),
    //                    std::min(static_cast<index_type>(m_rp.m_tile_end[2] *
    //                                                     m_rp.m_tile_end[3]),
    //                             static_cast<index_type>(maxblocks)),
    //                    std::min(static_cast<index_type>(m_rp.m_tile_end[4] *
    //                                                     m_rp.m_tile_end[5]),
    //                             static_cast<index_type>(maxblocks)));
    //    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor,
    //    LaunchBounds>(
    //        *this, grid, block, 0,
    //        m_rp.space().impl_internal_space_instance(), false);
    //  }
    //  else
    //  {
    //    printf("Kokkos::MDRange Error: Exceeded rank bounds with HIP\n");
    //    Kokkos::abort("Aborting");
    //  }

  }  // end execute

  //  inline
  ParallelFor(const FunctorType &arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_rp(arg_policy) {}
};
}  // namespace Impl
}  // namespace Kokkos

#endif
