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

#ifndef KOKKOS_OPENMPTARGET_PARALLEL_MDRANGE_HPP
#define KOKKOS_OPENMPTARGET_PARALLEL_MDRANGE_HPP

#include <omp.h>
#include <iostream>
#include <Kokkos_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Exec.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::OpenMPTarget> {
 private:
  typedef Kokkos::MDRangePolicy<Traits...> Policy;
  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const { execute_impl<WorkTag>(); }
/*
  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type
  execute_impl() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const typename Policy::member_type begin = m_policy.begin();
    const typename Policy::member_type end   = m_policy.end();

#pragma omp target teams distribute parallel for map(to: this->m_functor)
    for (int i = begin; i < end; i++) m_functor(i);
  }
*/
  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type
  execute_impl() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const typename Policy::member_type begin = m_policy.begin();
    const typename Policy::member_type end   = m_policy.end();
    FunctorType a_functor(m_functor);
#pragma omp target teams distribute parallel for map(to : a_functor)
    for (int i = begin; i < end; i++) a_functor(i);
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  execute_impl() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const typename Policy::member_type begin = m_policy.begin();
    const typename Policy::member_type end   = m_policy.end();

    FunctorType a_functor(m_functor);
#pragma omp target teams distribute parallel for num_threads(128) \
    map(to                                                        \
        : a_functor)
    for (int i = begin; i < end; i++) a_functor(TagType(), i);
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class PointerType,
          class ValueType, class... PolicyArgs>
struct ParallelReduceSpecialize<FunctorType, Kokkos::MDRangePolicy<PolicyArgs...>,
                                ReducerType, PointerType, ValueType, 0, 0> {
  typedef Kokkos::RangePolicy<PolicyArgs...> PolicyType;
  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_impl(const FunctorType& f, const PolicyType& p,
                   PointerType result_ptr) {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const typename PolicyType::member_type begin = p.begin();
    const typename PolicyType::member_type end   = p.end();

    ValueType result = ValueType();
#pragma omp target teams distribute parallel for num_teams(512) map(to:f) map(tofrom:result) reduction(+: result)
    for (int i = begin; i < end; i++) f(i, result);

    *result_ptr = result;
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_impl(const FunctorType& f, const PolicyType& p,
                   PointerType result_ptr) {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const typename PolicyType::member_type begin = p.begin();
    const typename PolicyType::member_type end   = p.end();

    ValueType result = ValueType();
#pragma omp target teams distribute parallel for num_teams(512) map(to:f) map(tofrom: result) reduction(+: result)
    for (int i = begin; i < end; i++) f(TagType(), i, result);

    *result_ptr = result;
  }

  inline static void execute(const FunctorType& f, const PolicyType& p,
                             PointerType ptr) {
    execute_impl<typename PolicyType::work_tag>(f, p, ptr);
  }
};
/*
template<class FunctorType, class PolicyType, class ReducerType, class
PointerType, class ValueType> struct ParallelReduceSpecialize<FunctorType,
PolicyType, ReducerType, PointerType, ValueType, 0,1> {

  #pragma omp declare reduction(custom: ValueType : ReducerType::join(omp_out,
omp_in)) initializer ( ReducerType::init(omp_priv) )

  template< class TagType >
  inline static
  typename std::enable_if< std::is_same< TagType , void >::value >::type
  execute_impl(const FunctorType& f, const PolicyType& p, PointerType
result_ptr)
    {
      OpenMPTargetExec::verify_is_process("Kokkos::Experimental::OpenMPTarget
parallel_for");
      OpenMPTargetExec::verify_initialized("Kokkos::Experimental::OpenMPTarget
parallel_for"); const typename PolicyType::member_type begin = p.begin(); const
typename PolicyType::member_type end = p.end();

      ValueType result = ValueType();
      #pragma omp target teams distribute parallel for num_teams(512) map(to:f)
map(tofrom:result) reduction(custom: result) for(int i=begin; i<end; i++)
        f(i,result);

      *result_ptr=result;
    }


  template< class TagType >
  inline static
  typename std::enable_if< ! std::is_same< TagType , void >::value >::type
  execute_impl(const FunctorType& f, const PolicyType& p, PointerType
result_ptr)
    {
      OpenMPTargetExec::verify_is_process("Kokkos::Experimental::OpenMPTarget
parallel_for");
      OpenMPTargetExec::verify_initialized("Kokkos::Experimental::OpenMPTarget
parallel_for"); const typename PolicyType::member_type begin = p.begin(); const
typename PolicyType::member_type end = p.end();

      ValueType result = ValueType();
      #pragma omp target teams distribute parallel for num_teams(512) map(to:f)
map(tofrom: result) reduction(custom: result) for(int i=begin; i<end; i++)
        f(TagType(),i,result);

      *result_ptr=result;
    }


    inline static
    void execute(const FunctorType& f, const PolicyType& p, PointerType ptr) {
      execute_impl<typename PolicyType::work_tag>(f,p,ptr);
    }
};
*/

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::OpenMPTarget> {
 private:
  typedef Kokkos::MDRangePolicy<Traits...> Policy;

  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;

  // Static Assert WorkTag void if ReducerType not InvalidType

  typedef Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>
      ValueTraits;
  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd> ValueJoin;

  enum { HasJoin = ReduceFunctorHasJoin<FunctorType>::value };
  enum { UseReducer = is_reducer_type<ReducerType>::value };

  typedef typename ValueTraits::pointer_type pointer_type;
  typedef typename ValueTraits::reference_type reference_type;

  typedef ParallelReduceSpecialize<
      FunctorType, Policy, ReducerType, pointer_type,
      typename ValueTraits::value_type, HasJoin, UseReducer>
      ParForSpecialize;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {
    ParForSpecialize::execute(m_functor, m_policy, m_result_ptr);
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_result_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = NULL)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::Experimental::OpenMPTarget must be a
      Kokkos::View in HostSpace" );*/
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::Experimental::OpenMPTarget must be a
      Kokkos::View in HostSpace" );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos


}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* KOKKOS_OPENMPTARGET_PARALLEL_HPP */
