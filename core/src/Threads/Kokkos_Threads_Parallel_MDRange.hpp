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

#ifndef KOKKOS_THREADS_PARALLEL_MDRANGE_HPP
#define KOKKOS_THREADS_PARALLEL_MDRANGE_HPP

#include <Kokkos_Parallel.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {
namespace Impl {

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
  static std::enable_if_t<std::is_same<Schedule, Kokkos::Static>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelFor &self = *((const ParallelFor *)arg);

    WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelFor::exec_range(self.m_mdr_policy, self.m_functor, range.begin(),
                            range.end());

    exec.fan_in();
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, Kokkos::Dynamic>::value>
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

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy &, const Functor &) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType,
                     Kokkos::MDRangePolicy<Traits...>, Kokkos::Threads> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using WorkTag   = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using pointer_type   = typename ReducerType::pointer_type;
  using value_type     = typename ReducerType::value_type;
  using reference_type = typename ReducerType::reference_type;

  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  const CombinedFunctorReducerType m_functor_reducer;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;  // construct as RangePolicy( 0, num_tiles
                          // ).set_chunk_size(1) in ctor
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
  static std::enable_if_t<std::is_same<Schedule, Kokkos::Static>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    ParallelReduce::exec_range(
        self.m_mdr_policy, self.m_functor_reducer.get_functor(), range.begin(),
        range.end(),
        self.m_functor_reducer.get_reducer().init(
            static_cast<pointer_type>(exec.reduce_memory())));

    exec.fan_in_reduce(self.m_functor_reducer.get_reducer());
  }

  template <class Schedule>
  static std::enable_if_t<std::is_same<Schedule, Kokkos::Dynamic>::value>
  exec_schedule(ThreadsExec &exec, const void *arg) {
    const ParallelReduce &self = *((const ParallelReduce *)arg);
    const WorkRange range(self.m_policy, exec.pool_rank(), exec.pool_size());

    exec.set_work_range(range.begin(), range.end(), self.m_policy.chunk_size());
    exec.reset_steal_target();
    exec.barrier();

    long work_index = exec.get_work_index();

    reference_type update = self.m_functor_reducer.get_reducer().init(
        static_cast<pointer_type>(exec.reduce_memory()));
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

    exec.fan_in_reduce(self.m_functor_reducer.get_reducer());
  }

 public:
  inline void execute() const {
    ThreadsExec::resize_scratch(m_functor_reducer.get_reducer().value_size(),
                                0);

    ThreadsExec::start(&ParallelReduce::exec, this);

    ThreadsExec::fence();

    if (m_result_ptr) {
      const pointer_type data =
          (pointer_type)ThreadsExec::root_reduce_scratch();

      const unsigned n = m_functor_reducer.get_reducer().value_count();
      for (unsigned i = 0; i < n; ++i) {
        m_result_ptr[i] = data[i];
      }
    }
  }

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType &arg_functor_reducer,
                 const MDRangePolicy &arg_policy,
                 const ViewType &arg_result_view)
      : m_functor_reducer(arg_functor_reducer),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Kokkos::Threads reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::Threads reduce result must be accessible from HostSpace");
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy &, const Functor &) {
    /**
     * 1024 here is just our guess for a reasonable max tile size,
     * it isn't a hardware constraint. If people see a use for larger
     * tile size products, we're happy to change this.
     */
    return 1024;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
