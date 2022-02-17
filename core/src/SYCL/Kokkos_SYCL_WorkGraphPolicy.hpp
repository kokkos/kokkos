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

#ifndef KOKKOS_SYCL_WORKGRAPHPOLICY_HPP
#define KOKKOS_SYCL_WORKGRAPHPOLICY_HPP

#include <Kokkos_SYCL.hpp>

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::WorkGraphPolicy<Traits...>,
                  Kokkos::Experimental::SYCL> {
 public:
  using PolicyType = Kokkos::WorkGraphPolicy<Traits...>;
  using WorkTag    = typename PolicyType::work_tag;

 private:
  PolicyType m_policy;
  FunctorType m_functor;

  template <typename PolicyWrapper, typename FunctorWrapper>
  static void sycl_direct_launch(const PolicyWrapper& policy_wrapper,
                                 const FunctorWrapper& functor_wrapper) {
    // Convenience references
    const Kokkos::Experimental::SYCL& space =
        static_cast<const PolicyType&>(policy_wrapper.get_functor()).space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;
    //    const auto concurrency = space.concurrency();

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(999), [=](sycl::item<1>) {
        for (std::int32_t w = PolicyType::END_TOKEN;
             PolicyType::COMPLETED_TOKEN !=
             (w = static_cast<const PolicyType&>(policy_wrapper.get_functor())
                      .pop_work());) {
          if (PolicyType::END_TOKEN != w) {
            if constexpr (std::is_same<WorkTag, void>::value)
              functor_wrapper.get_functor()(w);
            else
              functor_wrapper.get_functor(WorkTag(), w);
            static_cast<const PolicyType&>(policy_wrapper.get_functor())
                .completed_work(w);
          }
        }
      });
    });
  }

 public:
  inline void execute() {
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *m_policy.space().impl_internal_space_instance();

    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
        indirectKernelMem = instance.get_indirect_kernel_mem();
    // slightly abuse m_indirectReducerMem for the policy here
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem&
        indirectPolicyMem = instance.get_indirect_kernel_mem();

    const auto functor_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_functor, indirectKernelMem);
    const auto policy_wrapper = Experimental::Impl::make_sycl_function_wrapper(
        m_policy, indirectPolicyMem);
    sycl_direct_launch(policy_wrapper, functor_wrapper);
  }

  inline ParallelFor(const FunctorType& arg_functor,
                     const PolicyType& arg_policy)
      : m_policy(arg_policy), m_functor(arg_functor) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* #define KOKKOS_SYCL_WORKGRAPHPOLICY_HPP */
