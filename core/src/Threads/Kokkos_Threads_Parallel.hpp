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

#ifndef KOKKOS_THREADS_PARALLEL_HPP
#define KOKKOS_THREADS_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_THREADS)

#include <Kokkos_Parallel.hpp>

#include <impl/Kokkos_FunctorAdapter.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelFor Kokkos::Threads with RangePolicy */

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Threads> {
 private:
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member ibeg,
                 const Member iend) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(i);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member ibeg,
                 const Member iend) {
    const TagType t{};
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(t, i);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelFor::template exec_range<WorkTag>(self.m_functor, range.begin(),
                                              range.end());

    exec.fan_in();
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin() - self.m_policy.begin(),
                        range.end() - self.m_policy.begin(),
                        self.m_policy.chunk_size());
    exec.reset_steal_target();
    exec.barrier();

    long work_index = exec.get_work_index();

    while (work_index != -1) {
      const Member begin =
          static_cast<Member>(work_index) * self.m_policy.chunk_size() +
          self.m_policy.begin();
      const Member end =
          begin + self.m_policy.chunk_size() < self.m_policy.end()
              ? begin + self.m_policy.chunk_size()
              : self.m_policy.end();
      ParallelFor::template exec_range<WorkTag>(self.m_functor, begin, end);
      work_index = exec.get_work_index();
    }

    exec.fan_in();
  }

 public:
  inline void execute() const {
    ThreadsExec::start(&ParallelFor::exec, this);
    ThreadsExec::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

// MDRangePolicy impl
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Threads> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag = typename MDRangePolicy::work_tag;

  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor

  inline static void exec_range(const MDRangePolicy &mdr_policy,
                                const FunctorType &functor, const Member ibeg,
                                const Member iend) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      iterate_type(mdr_policy, functor)(i);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelFor::exec_range(self.m_mdr_policy, self.m_functor, range.begin(),
                            range.end());

    exec.fan_in();
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin(), range.end(), self.m_policy.chunk_size());
    exec.reset_steal_target();
    exec.barrier();

    long work_index = exec.get_work_index();

    while (work_index != -1) {
      const Member begin =
          static_cast<Member>(work_index) * self.m_policy.chunk_size();
      const Member end =
          begin + self.m_policy.chunk_size() < self.m_policy.end()
              ? begin + self.m_policy.chunk_size()
              : self.m_policy.end();

      ParallelFor::exec_range(self.m_mdr_policy, self.m_functor, begin, end);
      work_index = exec.get_work_index();
    }

    exec.fan_in();
  }

 public:
  inline void execute() const {
    ThreadsExec::start(&ParallelFor::exec, this);
    ThreadsExec::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const MDRangePolicy &arg_policy)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};

//----------------------------------------------------------------------------
/* ParallelFor Kokkos::Threads with TeamPolicy */

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Threads> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Threads, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const int m_shared;

  template <class TagType, class Schedule>
  inline static typename std::enable_if<
      std::is_same<TagType, void>::value &&
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_static(); member.next_static()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static typename std::enable_if<
      !std::is_same<TagType, void>::value &&
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_static(); member.next_static()) {
      functor(t, member);
    }
  }

  template <class TagType, class Schedule>
  inline static typename std::enable_if<
      std::is_same<TagType, void>::value &&
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_team(const FunctorType &functor, Member member) {
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(member);
    }
  }

  template <class TagType, class Schedule>
  inline static typename std::enable_if<
      !std::is_same<TagType, void>::value &&
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_team(const FunctorType &functor, Member member) {
    const TagType t{};
    for (; member.valid_dynamic(); member.next_dynamic()) {
      functor(t, member);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    ParallelFor::exec_team<WorkTag, typename Policy::schedule_type::type>(
        self.m_functor, Member(&exec, self.m_policy, self.m_shared));

    exec.barrier();
    exec.fan_in();
  }
  template <typename Policy>
  Policy fix_policy(Policy policy) {
    if (policy.impl_vector_length() < 0) {
      policy.impl_set_vector_length(1);
    }
    if (policy.team_size() < 0) {
      policy.impl_set_team_size(
          policy.team_size_recommended(m_functor, ParallelForTag{}));
    }
    return policy;
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(
        0, Policy::member_type::team_reduce_size() + m_shared);

    ThreadsExec::start(&ParallelFor::exec, this);

    ThreadsExec::fence();
  }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor),
        m_policy(fix_policy(arg_policy)),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, m_policy.team_size())) {}
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::Threads and RangePolicy */

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Threads> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit  = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueFinal = Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(i, update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update) {
    const TagType t{};
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(t, i, update);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelReduce::template exec_range<WorkTag>(
        self.m_functor, range.begin(), range.end(),
        ValueInit::init(
            ReducerConditional::select(self.m_functor, self.m_reducer),
            exec.reduce_memory()));

    exec.template fan_in_reduce<ReducerTypeFwd, WorkTagFwd>(
        ReducerConditional::select(self.m_functor, self.m_reducer));
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin() - self.m_policy.begin(),
                        range.end() - self.m_policy.begin(),
                        self.m_policy.chunk_size());
    exec.reset_steal_target();
    exec.barrier();

    long work_index       = exec.get_work_index();
    reference_type update = ValueInit::init(
        ReducerConditional::select(self.m_functor, self.m_reducer),
        exec.reduce_memory());
    while (work_index != -1) {
      const Member begin =
          static_cast<Member>(work_index) * self.m_policy.chunk_size() +
          self.m_policy.begin();
      const Member end =
          begin + self.m_policy.chunk_size() < self.m_policy.end()
              ? begin + self.m_policy.chunk_size()
              : self.m_policy.end();
      ParallelReduce::template exec_range<WorkTag>(self.m_functor, begin, end,
                                                   update);
      work_index = exec.get_work_index();
    }

    exec.template fan_in_reduce<ReducerTypeFwd, WorkTagFwd>(
        ReducerConditional::select(self.m_functor, self.m_reducer));
  }

 public:
  inline void execute() const {
    if (m_policy.end() <= m_policy.begin()) {
      if (m_result_ptr) {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        m_result_ptr);
        ValueFinal::final(ReducerConditional::select(m_functor, m_reducer),
                          m_result_ptr);
      }
    } else {
      ThreadsExec::resize_scratch(
          ValueTraits::value_size(
              ReducerConditional::select(m_functor, m_reducer)),
          0);

      ThreadsExec::start(&ParallelReduce::exec, this);

      ThreadsExec::fence();

      if (m_result_ptr) {
        const pointer_type data =
            (pointer_type)ThreadsExec::root_reduce_scratch();

        const unsigned n = ValueTraits::value_count(
            ReducerConditional::select(m_functor, m_reducer));
        for (unsigned i = 0; i < n; ++i) {
          m_result_ptr[i] = data[i];
        }
      }
    }
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const HostViewType &arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::Threads reduce result must be a View");

    static_assert(
        std::is_same<typename HostViewType::memory_space, HostSpace>::value,
        "Kokkos::Threads reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

// MDRangePolicy impl
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Threads> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag   = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using value_type     = typename ValueTraits::value_type;
  using reference_type = typename ValueTraits::reference_type;

  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  inline static void exec_range(const MDRangePolicy &mdr_policy,
                                const FunctorType &functor, const Member &ibeg,
                                const Member &iend, reference_type update) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      iterate_type(mdr_policy, functor, update)(i);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    exec_schedule<typename Policy::schedule_type::type>(exec, arg);
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Static>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelReduce::exec_range(
        self.m_mdr_policy, self.m_functor, range.begin(), range.end(),
        ValueInit::init(
            ReducerConditional::select(self.m_functor, self.m_reducer),
            exec.reduce_memory()));

    exec.template fan_in_reduce<ReducerTypeFwd, WorkTagFwd>(
        ReducerConditional::select(self.m_functor, self.m_reducer));
  }

  template <class Schedule>
  static typename std::enable_if<
      std::is_same<Schedule, Kokkos::Dynamic>::value>::type
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin(), range.end(), self.m_policy.chunk_size());
    exec.reset_steal_target();
    exec.barrier();

    long work_index       = exec.get_work_index();
    reference_type update = ValueInit::init(
        ReducerConditional::select(self.m_functor, self.m_reducer),
        exec.reduce_memory());
    while (work_index != -1) {
      const Member begin =
          static_cast<Member>(work_index) * self.m_policy.chunk_size();
      const Member end =
          begin + self.m_policy.chunk_size() < self.m_policy.end()
              ? begin + self.m_policy.chunk_size()
              : self.m_policy.end();
      ParallelReduce::exec_range(self.m_mdr_policy, self.m_functor, begin, end,
                                 update);
      work_index = exec.get_work_index();
    }

    exec.template fan_in_reduce<ReducerTypeFwd, WorkTagFwd>(
        ReducerConditional::select(self.m_functor, self.m_reducer));
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(
        ValueTraits::value_size(
            ReducerConditional::select(m_functor, m_reducer)),
        0);

    ThreadsExec::start(&ParallelReduce::exec, this);

    ThreadsExec::fence();

    if (m_result_ptr) {
      const pointer_type data =
          (pointer_type)ThreadsExec::root_reduce_scratch();

      const unsigned n = ValueTraits::value_count(
          ReducerConditional::select(m_functor, m_reducer));
      for (unsigned i = 0; i < n; ++i) {
        m_result_ptr[i] = data[i];
      }
    }
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const MDRangePolicy &arg_policy,
      const HostViewType &arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = nullptr)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::Threads reduce result must be a View");

    static_assert(
        std::is_same<typename HostViewType::memory_space, HostSpace>::value,
        "Kokkos::Threads reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType &arg_functor,
                        MDRangePolicy arg_policy, const ReducerType &reducer)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

//----------------------------------------------------------------------------
/* ParallelReduce with Kokkos::Threads and TeamPolicy */

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Threads> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Threads, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueTraits =
      Kokkos::Impl::FunctorValueTraits<ReducerTypeFwd, WorkTagFwd>;
  using ValueInit  = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueFinal = Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const int m_shared;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_team(const FunctorType &functor, Member member,
                reference_type update) {
    for (; member.valid_static(); member.next_static()) {
      functor(member, update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_team(const FunctorType &functor, Member member,
                reference_type update) {
    const TagType t{};
    for (; member.valid_static(); member.next_static()) {
      functor(t, member, update);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);

    ParallelReduce::template exec_team<WorkTag>(
        self.m_functor, Member(&exec, self.m_policy, self.m_shared),
        ValueInit::init(
            ReducerConditional::select(self.m_functor, self.m_reducer),
            exec.reduce_memory()));

    exec.template fan_in_reduce<ReducerTypeFwd, WorkTagFwd>(
        ReducerConditional::select(self.m_functor, self.m_reducer));
  }

 public:
  inline void execute() const {
    if (m_policy.league_size() * m_policy.team_size() == 0) {
      if (m_result_ptr) {
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                        m_result_ptr);
        ValueFinal::final(ReducerConditional::select(m_functor, m_reducer),
                          m_result_ptr);
      }
    } else {
      ThreadsExec::resize_scratch(
          ValueTraits::value_size(
              ReducerConditional::select(m_functor, m_reducer)),
          Policy::member_type::team_reduce_size() + m_shared);

      ThreadsExec::start(&ParallelReduce::exec, this);

      ThreadsExec::fence();

      if (m_result_ptr) {
        const pointer_type data =
            (pointer_type)ThreadsExec::root_reduce_scratch();

        const unsigned n = ValueTraits::value_count(
            ReducerConditional::select(m_functor, m_reducer));
        for (unsigned i = 0; i < n; ++i) {
          m_result_ptr[i] = data[i];
        }
      }
    }
  }

  template <typename Policy>
  Policy fix_policy(Policy policy) {
    if (policy.impl_vector_length() < 0) {
      policy.impl_set_vector_length(1);
    }
    if (policy.team_size() < 0) {
      policy.impl_set_team_size(policy.team_size_recommended(
          m_functor, m_reducer, ParallelReduceTag{}));
    }
    return policy;
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const ViewType &arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(fix_policy(arg_policy)),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, m_policy.team_size())) {}

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor),
        m_policy(fix_policy(arg_policy)),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shared(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, m_policy.team_size())) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                            , Kokkos::HostSpace >::value
    , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
    );*/
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
/* ParallelScan with Kokkos::Threads and RangePolicy */

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Threads> {
 private:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using WorkRange   = typename Policy::WorkRange;
  using WorkTag     = typename Policy::work_tag;
  using Member      = typename Policy::member_type;
  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
  using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update, const bool final) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(i, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update, const bool final) {
    const TagType t{};
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(t, i, update, final);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelScan &self = *((const ParallelScan *)arg);

    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    reference_type update =
        ValueInit::init(self.m_functor, exec.reduce_memory());

    ParallelScan::template exec_range<WorkTag>(self.m_functor, range.begin(),
                                               range.end(), update, false);

    //  exec.template scan_large<FunctorType,WorkTag>( self.m_functor );
    exec.template scan_small<FunctorType, WorkTag>(self.m_functor);

    ParallelScan::template exec_range<WorkTag>(self.m_functor, range.begin(),
                                               range.end(), update, true);

    exec.fan_in();
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(2 * ValueTraits::value_size(m_functor), 0);
    ThreadsExec::start(&ParallelScan::exec, this);
    ThreadsExec::fence();
  }

  ParallelScan(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Threads> {
 private:
  using Policy      = Kokkos::RangePolicy<Traits...>;
  using WorkRange   = typename Policy::WorkRange;
  using WorkTag     = typename Policy::work_tag;
  using Member      = typename Policy::member_type;
  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, WorkTag>;
  using ValueInit   = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename ValueTraits::pointer_type;
  using reference_type = typename ValueTraits::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType &m_returnvalue;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update, const bool final) {
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(i, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_range(const FunctorType &functor, const Member &ibeg,
                 const Member &iend, reference_type update, const bool final) {
    const TagType t{};
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#pragma ivdep
#endif
    for (Member i = ibeg; i < iend; ++i) {
      functor(t, i, update, final);
    }
  }

  static void exec(ThreadsExec &exec, const void *arg) {
    const ParallelScanWithTotal &self = *((const ParallelScanWithTotal *)arg);

    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    reference_type update =
        ValueInit::init(self.m_functor, exec.reduce_memory());

    ParallelScanWithTotal::template exec_range<WorkTag>(
        self.m_functor, range.begin(), range.end(), update, false);

    //  exec.template scan_large<FunctorType,WorkTag>( self.m_functor );
    exec.template scan_small<FunctorType, WorkTag>(self.m_functor);

    ParallelScanWithTotal::template exec_range<WorkTag>(
        self.m_functor, range.begin(), range.end(), update, true);

    exec.fan_in();

    if (exec.pool_rank() == exec.pool_size() - 1) {
      self.m_returnvalue = update;
    }
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(2 * ValueTraits::value_size(m_functor), 0);
    ThreadsExec::start(&ParallelScanWithTotal::exec, this);
    ThreadsExec::fence();
  }

  ParallelScanWithTotal(const FunctorType &arg_functor,
                        const Policy &arg_policy, ReturnType &arg_returnvalue)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif
#endif /* #define KOKKOS_THREADS_PARALLEL_HPP */
