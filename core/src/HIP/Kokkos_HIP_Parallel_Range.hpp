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

#ifndef KOKKO_HIP_PARALLEL_RANGE_HPP
#define KOKKO_HIP_PARALLEL_RANGE_HPP

#include <HIP/Kokkos_HIP_KernelLaunch.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Experimental::HIP> {
 public:
  typedef Kokkos::RangePolicy<Traits...> Policy;

 private:
  typedef typename Policy::member_type Member;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;

  const FunctorType m_functor;
  const Policy m_policy;

  ParallelFor()        = delete;
  ParallelFor &operator=(const ParallelFor &) = delete;

  template <class TagType>
  inline __device__
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const Member i) const {
    m_functor(i);
  }

  template <class TagType>
  inline __device__
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const Member i) const {
    m_functor(TagType(), i);
  }

 public:
  typedef FunctorType functor_type;

  inline __device__ void operator()(void) const {
    const Member work_stride = blockDim.y * gridDim.x;
    const Member work_end    = m_policy.end();

    for (Member iwork =
             m_policy.begin() + threadIdx.y + blockDim.y * blockIdx.x;
         iwork < work_end;
         iwork = iwork < work_end - work_stride ? iwork + work_stride
                                                : work_end) {
      this->template exec_range<WorkTag>(iwork);
    }
  }

  inline void execute() const {
    const typename Policy::index_type nwork = m_policy.end() - m_policy.begin();

    const int block_size = 256;  // Choose block_size better
    const dim3 block(1, block_size, 1);
    const dim3 grid(
        typename Policy::index_type((nwork + block.y - 1) / block.y), 1, 1);

    Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor, LaunchBounds>(
        *this, grid, block, 0, m_policy.space().impl_internal_space_instance(),
        false);
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
