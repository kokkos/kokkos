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

#ifndef KOKKOS_SYCL_PARALLEL_RANGE_HPP_
#define KOKKOS_SYCL_PARALLEL_RANGE_HPP_

#include <SYCL/Kokkos_SYCL_KernelLaunch.hpp>
//#include <algorithm>
//#include <functional>

template <class FunctorType, class ExecPolicy>
class Kokkos::Impl::ParallelFor<FunctorType, ExecPolicy,
                                Kokkos::Experimental::SYCL> {
 public:
  typedef ExecPolicy Policy;

 private:
  typedef typename Policy::member_type Member;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::launch_bounds LaunchBounds;

 public:
  const FunctorType m_functor;
  const Policy m_policy;

 private:
  ParallelFor()        = delete;
  ParallelFor& operator=(const ParallelFor&) = delete;

  template <class TagType>
  typename std::enable_if<std::is_same<TagType, void>::value>::type exec_range(
      const Member i) const {
    m_functor(i);
  }

  template <class TagType>
  typename std::enable_if<!std::is_same<TagType, void>::value>::type exec_range(
      const Member i) const {
    m_functor(TagType(), i);
  }

 public:
  typedef FunctorType functor_type;

  inline void operator()(cl::sycl::item<1> item) const {
    int id = item.get_linear_id();
    m_functor(id);
  }

  inline void execute() const {
    Kokkos::Experimental::Impl::sycl_launch(*this);
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

#endif  // KOKKOS_SYCL_PARALLEL_RANGE_HPP_
