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
      Kokkos::Impl::if_c<std::is_same_v<InvalidType, ReducerType>, FunctorType,
                         ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      std::conditional_t<std::is_same_v<InvalidType, ReducerType>, WorkTag,
                         void>;
  using ValueInit =
      typename Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

 public:
  // V - View
  template <typename V>
  ParallelReduce(const FunctorType& f, const Policy& p, const V& v)
      : m_functor(f), m_policy(p), m_result_ptr(v.data()) {
  }

 private:
  template <typename TagType>
  std::enable_if_t<std::is_void_v<TagType>> exec(reference_type update) {
    using member_type = typename Policy::member_type;
    member_type e     = m_policy.end();
    for (member_type i = m_policy.begin(); i < e; ++i) m_functor(i, update);
  }

  template <typename TagType>
  std::enable_if_t<!std::is_void_v<TagType>> exec(reference_type update) {
    using member_type = typename Policy::member_type;
    member_type e     = m_policy.end();
    for (member_type i = m_policy.begin(); i < e; ++i)
      m_functor(TagType{}, i, update);
  }

  template <typename Tag>
  static auto tagged_reducer(const ReducerTypeFwd& r, Tag*) {
    return [r](int i, reference_type acc) { return r(Tag{}, i, acc); };
  };

  static ReducerTypeFwd tagged_reducer(const ReducerTypeFwd& r, void*) {
    return r;
  }

 public:
  void execute() {

    cl::sycl::queue& q =
        *execution_space().impl_internal_space_instance()->m_queue;

    q.submit([this](cl::sycl::handler& cgh) {
      cl::sycl::nd_range<1> range(m_policy.end() - m_policy.begin(), 8);

      ReducerTypeFwd functor = ReducerConditional::select(m_functor, m_reducer);

      value_type identity = ValueInit::init(functor, m_result_ptr);
      auto reduction =
          cl::sycl::intel::reduction(m_result_ptr, identity, std::plus<>());

      auto taggedFunctor =
          tagged_reducer(functor, static_cast<WorkTag*>(nullptr));

      cgh.parallel_for(range, reduction,
                       [=](cl::sycl::nd_item<1> item, auto& sum) {
                         int i              = item.get_global_id(0);
                         value_type partial = identity;
                         taggedFunctor(i, partial);
                         sum.combine(partial);
                       });
    });

    q.wait();
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
