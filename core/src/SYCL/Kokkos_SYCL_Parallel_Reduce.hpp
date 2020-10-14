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

#ifndef KOKKOS_SYCL_PARALLEL_REDUCE_HPP
#define KOKKOS_SYCL_PARALLEL_REDUCE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_SYCL)

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::SYCL> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using execution_space = typename Analysis::execution_space;
  using value_type      = typename Analysis::value_type;
  using pointer_type    = typename Analysis::pointer_type;
  using reference_type  = typename Analysis::reference_type;

  using WorkTag = typename Policy::work_tag;
  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      std::conditional_t<std::is_same<InvalidType, ReducerType>::value, WorkTag,
                         void>;
  using ValueInit =
      typename Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

 public:
  // V - View
  template <typename V>
  ParallelReduce(
      const FunctorType& f, const Policy& p, const V& v,
      typename std::enable_if<Kokkos::is_view<V>::value, void*>::type = nullptr)
      : m_functor(f), m_policy(p), m_result_ptr(v.data()) {
    // FIXME_SYCL custom reducer not yet implemented
    if (m_result_ptr == nullptr) std::abort();
  }

  ParallelReduce(const FunctorType& f, const Policy& p,
                 const ReducerType& reducer)
      : m_functor(f),
        m_policy(p),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    // FIXME_SYCL custom reducer not yet implemented
    if (m_result_ptr == nullptr) std::abort();
  }

 private:
  template <typename TagType>
  std::enable_if_t<std::is_void<TagType>::value> exec(reference_type update) {
    using member_type = typename Policy::member_type;
    member_type e     = m_policy.end();
    for (member_type i = m_policy.begin(); i < e; ++i) m_functor(i, update);
  }

  template <typename TagType>
  std::enable_if_t<!std::is_void<TagType>::value> exec(reference_type update) {
    using member_type = typename Policy::member_type;
    member_type e     = m_policy.end();
    for (member_type i = m_policy.begin(); i < e; ++i)
      m_functor(TagType{}, i, update);
  }

  template <typename PolicyType, typename Functor>
  void sycl_direct_launch(const PolicyType& policy,
                          const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    cl::sycl::queue& q = *instance.m_queue;

    q.submit([this, functor, policy](cl::sycl::handler& cgh) {
      // FIXME_SYCL a local size larger than 1 doesn't work for all cases
      cl::sycl::nd_range<1> range(policy.end() - policy.begin(), 1);

      value_type identity = 0;
      ValueInit::init(functor, m_result_ptr);
      auto reduction =
          cl::sycl::intel::reduction(m_result_ptr, identity, std::plus<>());

      cgh.parallel_for(
          range, reduction, [=](cl::sycl::nd_item<1> item, auto& sum) {
            const typename Policy::index_type id = item.get_global_id(0);
            value_type partial                   = identity;
            if constexpr (std::is_same<WorkTag, void>::value)
              functor(id, partial);
            else
              functor(WorkTag(), id, partial);
            sum.combine(partial);
          });
    });

    q.wait();
  }

  template <typename Functor>
  void sycl_indirect_launch(const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMemory& kernelMem =
        *instance.m_indirectKernel;

    // Allocate USM shared memory for the functor
    kernelMem.resize(std::max(kernelMem.size(), sizeof(functor)));

    // Placement new a copy of functor into USM shared memory
    //
    // Store it in a unique_ptr to call its destructor on scope exit
    std::unique_ptr<ReducerTypeFwd, Kokkos::Impl::destruct_delete>
        kernelFunctorPtr(new (kernelMem.data()) ReducerTypeFwd(functor));

    auto kernelFunctor = std::reference_wrapper(*kernelFunctorPtr);
    sycl_direct_launch(m_policy, kernelFunctor);
  }

 public:
  void execute() const {
    ReducerTypeFwd functor = ReducerConditional::select(m_functor, m_reducer);

    if constexpr (std::is_trivially_copyable_v<decltype(functor)>)
      sycl_direct_launch(m_policy, functor);
    else
      sycl_indirect_launch(functor);
  }

 private:
  FunctorType m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* KOKKOS_SYCL_PARALLEL_REDUCE_HPP */
