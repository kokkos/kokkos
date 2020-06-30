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

#include <SYCL/Kokkos_SYCL_KernelLaunch.hpp>

// NLIBER
#include "pretty_name.h"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::SYCL> {
  // NLIBER replace with SYCL implementation
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
  // TODO Add constraints
  // F - Functor
  // P - Policy
  // R - View
  template <typename F, typename P, typename R>
  ParallelReduce(F const& f, P const& p, R const& r)
      : m_functor(f), m_policy(p), m_result_ptr(r.data()) {
    std::cout << "ParallelReduce::ParallelReduce\n";
    std::cout << "FunctorType:\t" << cool::pretty_type<FunctorType>() << '\n';
    std::cout << "F:          \t" << cool::pretty_type<F>() << '\n';

    std::cout << "Policy:\t" << cool::pretty_type<Policy>() << '\n';
    std::cout << "P:     \t" << cool::pretty_type<P>() << '\n';

    std::cout << "ReducerType:\t" << cool::pretty_type<ReducerType>() << '\n';
    std::cout << "R:          \t" << cool::pretty_type<R>() << '\n';

    std::cout << "execution_space:\t" << cool::pretty_type<execution_space>()
              << '\n';
    std::cout << "value_type:\t" << cool::pretty_type<value_type>() << '\n';
    std::cout << "pointer_type:\t" << cool::pretty_type<pointer_type>() << '\n';
    std::cout << "reference_type:\t" << cool::pretty_type<reference_type>()
              << '\n';

    std::cout << "WorkTag:\t" << cool::pretty_type<WorkTag>() << '\n';
    std::cout << "ReducerTypeFwd:\t" << cool::pretty_type<ReducerTypeFwd>()
              << '\n';
    std::cout << "WorkTagFwd:\t" << cool::pretty_type<WorkTagFwd>() << '\n';
    std::cout << "ValueInit:\t" << cool::pretty_type<ValueInit>() << '\n';

    std::cout << std::endl;
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

 public:
  void execute() {
    std::cout << "execute()\n";

    reference_type update = ValueInit::init(
        ReducerConditional::select(m_functor, m_reducer), m_result_ptr);

    this->exec<WorkTag>(update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), m_result_ptr);
  }

 private:
  FunctorType m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;

#if 0
 private:
  typedef Kokkos::RangePolicy<Traits...> Policy;

  typedef typename Policy::work_tag WorkTag;
  typedef typename Policy::WorkRange WorkRange;
  typedef typename Policy::member_type Member;

  typedef FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>
      Analysis;

  typedef Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                             FunctorType, ReducerType>
      ReducerConditional;
  typedef typename ReducerConditional::type ReducerTypeFwd;
  typedef
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type WorkTagFwd;

  // Static Assert WorkTag void if ReducerType not InvalidType

  typedef Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd> ValueInit;
  typedef Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd> ValueJoin;

  typedef typename Analysis::pointer_type pointer_type;
  typedef typename Analysis::reference_type reference_type;

  OpenMPExec* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(iwork, update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType& functor, const Member ibeg,
                 const Member iend, reference_type update) {
    const TagType t{};
    for (Member iwork = ibeg; iwork < iend; ++iwork) {
      functor(t, iwork, update);
    }
  }

 public:
  inline void execute() const {
    enum {
      is_dynamic = std::is_same<typename Policy::schedule_type::type,
                                Kokkos::Dynamic>::value
    };

    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

    const size_t pool_reduce_bytes =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    m_instance->resize_thread_data(pool_reduce_bytes, 0  // team_reduce_bytes
                                   ,
                                   0  // team_shared_bytes
                                   ,
                                   0  // thread_local_bytes
    );

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    const int pool_size = OpenMP::thread_pool_size();
#else
    const int pool_size = OpenMP::impl_thread_pool_size();
#endif
#pragma omp parallel num_threads(pool_size)
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_policy.end() - m_policy.begin(),
                              m_policy.chunk_size());

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      reference_type update =
          ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                          data.pool_reduce_local());

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        ParallelReduce::template exec_range<WorkTag>(
            m_functor, range.first + m_policy.begin(),
            range.second + m_policy.begin(), update);

      } while (is_dynamic && 0 <= range.first);
    }

    // Reduction:

    const pointer_type ptr =
        pointer_type(m_instance->get_thread_data(0)->pool_reduce_local());

    for (int i = 1; i < pool_size; ++i) {
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                      m_instance->get_thread_data(i)->pool_reduce_local());
    }

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);

    if (m_result_ptr) {
      const int n = Analysis::value_count(
          ReducerConditional::select(m_functor, m_reducer));

      for (int j = 0; j < n; ++j) {
        m_result_ptr[j] = ptr[j];
      }
    }
  }

  //----------------------------------------

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, Policy arg_policy,
      const ViewType& arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = NULL)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_instance(t_openmp_instance),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
#endif
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* KOKKOS_SYCL_PARALLEL_REDUCE_HPP */
