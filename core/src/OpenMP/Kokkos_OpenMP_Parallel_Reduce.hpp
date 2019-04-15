/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
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

#ifndef KOKKOS_OPENMP_KOKKOS_OPENMP_PARALLEL_REDUCE_HPP
#define KOKKOS_OPENMP_KOKKOS_OPENMP_PARALLEL_REDUCE_HPP

#include <Kokkos_Macros.hpp>
#if defined( KOKKOS_ENABLE_OPENMP )

#include <Concepts/Reducer/Kokkos_Reducer_Concept.hpp>
#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>

#include <Patterns/ParallelReduce/impl/Kokkos_ReducerStorage.hpp>

#include <OpenMP/Kokkos_OpenMP_Exec.hpp>

#include <Kokkos_ExecPolicy.hpp>

#include <omp.h>

namespace Kokkos {
namespace Impl {

//==============================================================================

template <class Derived>
class OpenMPParallelReduceCommon
{
private:

  // CRTP boilerplate
  constexpr inline
  Derived const& self() const noexcept
  {
    return *static_cast<Derived const*>(this);
  }

protected:

  inline void
  resize_thread_data() const
  {
    size_t pool_reduce_bytes =
      Concepts::reducer_value_size(self().m_reducer_storage.get_reducer());

    self().m_instance->resize_thread_data(
      pool_reduce_bytes,
      /* team_reduce_bytes = */ 0,
      /* team_shared_bytes = */ 0,
      /* thread_local_bytes = */ 0
    );
  }

  inline constexpr bool
  organize_team(HostThreadTeamData&) const
  {
    // in the absense of teams, all workers are active
    return true;
  }

  inline void
  set_work_partition(HostThreadTeamData& data) const
  {
    // TODO using begin and end customization points
    // TODO use Kokkos::query to get chunk size property
    data.set_work_partition(
      self().m_policy.end() - self().m_policy.begin(),
      self().m_policy.chunk_size()
    );
  }

  inline void
  disband_team(HostThreadTeamData&) const
  {
    // in the absense of teams, nothing to do here
  }

public:

  inline void execute()
  {
    using Policy = typename Derived::Policy;
    using ReducerType = typename Derived::reducer;

    // TODO make this a Kokkos::query()
    static constexpr auto is_dynamic =
      std::is_same<typename Policy::schedule_type::type, Kokkos::Dynamic>::value;

    OpenMPExec::verify_is_master("Kokkos::OpenMP parallel_reduce");

    self().resize_thread_data();

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
    const int pool_size = OpenMP::thread_pool_size();
#else
    const int pool_size = OpenMP::impl_thread_pool_size();
#endif

#pragma omp parallel num_threads(pool_size)
    {
      HostThreadTeamData& data = *(self().m_instance->get_thread_data());

      auto active = self().organize_team(data);

      if(active) {
        self().set_work_partition(data);
      }

      if /* constexpr */ (is_dynamic) {
        // Make sure work partition is set before stealing
        if(data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      Concepts::reducer_reference_type_t<ReducerType> update =
        Concepts::reducer_bind_reference(
          self().m_reducer_storage.get_reducer(),
          data.pool_reduce_local()
        );
      Concepts::reducer_init(self().m_reducer_storage.get_reducer(), update);

      if(active) {
        // Why is this a pair and not an array??
        std::pair<int64_t,int64_t> range(0,0);

        do {

          range = is_dynamic ? data.get_work_stealing_chunk()
            : data.get_work_partition();

          self().exec_range(data, range.first, range.second, update);

        } while ( is_dynamic && 0 <= range.first );
      }

      self().disband_team(data);

    } // end pragma omp parallel num_threads(pool_size)

    // Reduction:

    Concepts::reducer_reference_type_t<ReducerType> dst =
      Concepts::reducer_bind_reference(
        self().m_reducer_storage.get_reducer(),
        self().m_instance->get_thread_data(0)->pool_reduce_local()
      );

    for (int i = 1; i < pool_size; ++i) {

      Concepts::reducer_reference_type_t<ReducerType> i_src =
        Concepts::reducer_bind_reference(
          self().m_reducer_storage.get_reducer(),
          self().m_instance->get_thread_data(i)->pool_reduce_local()
        );

      Concepts::reducer_join(self().m_reducer_storage.get_reducer(), dst, i_src);

    }

    Concepts::reducer_final(self().m_reducer_storage.get_reducer(), dst);

    Concepts::reducer_assign_result(self().m_reducer_storage.get_reducer(), dst);
  }


};

//==============================================================================


template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<
  FunctorType,
  Kokkos::RangePolicy<Traits...>,
  ReducerType,
  Kokkos::OpenMP
> : public OpenMPParallelReduceCommon<
      ParallelReduce<
        FunctorType,
        Kokkos::RangePolicy<Traits...>,
        ReducerType,
        Kokkos::OpenMP
      >
    >
{
private:

  template <class Derived>
  friend class OpenMPParallelReduceCommon;

  using Policy = Kokkos::RangePolicy<Traits...>;
  using Member = typename Policy::member_type;
  using reducer = ReducerType;

  OpenMPExec* m_instance;
  FunctorType const m_functor;
  Policy const m_policy;
  ReducerStorage<ReducerType> const m_reducer_storage;

  inline
  void exec_range(
    HostThreadTeamData&,
    Member ibeg,
    Member iend,
    Concepts::reducer_reference_type_t<ReducerType> update
  ) const
  {
    auto begin = ibeg + m_policy.begin();
    auto end = iend + m_policy.begin();
    for(Member iwork = ibeg; iwork < iend; ++iwork) {
      Concepts::functor_invoke_with_policy(
        m_policy, m_functor, iwork, update
      );
    }
  }

  //----------------------------------------
public:

  template <class ViewType>
  inline
  ParallelReduce(
    FunctorType arg_functor,
    Policy arg_policy,
    ViewType const& arg_view,
    typename std::enable_if<
      Kokkos::is_view<ViewType>::value
        && !Kokkos::is_reducer_type<ViewType>::value,
      void const**
    >::type = nullptr
  )
    : m_instance(t_openmp_instance),
      m_functor(arg_functor),
      m_policy(arg_policy),
      m_reducer_storage(m_functor, m_policy, arg_view)
  { }

  inline
  ParallelReduce(
    FunctorType arg_functor,
    Policy arg_policy,
    ReducerType const& reducer
  ) : m_instance(t_openmp_instance),
      m_functor(std::move(arg_functor)),
      m_policy(std::move(arg_policy)),
      m_reducer_storage(reducer)
  { }

};


// MDRangePolicy impl
template<class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<
  FunctorType,
  Kokkos::MDRangePolicy<Traits...>,
  ReducerType,
  Kokkos::OpenMP
> : public OpenMPParallelReduceCommon<
      ParallelReduce<
        FunctorType,
        Kokkos::MDRangePolicy<Traits...>,
        ReducerType,
        Kokkos::OpenMP
      >
    >
{
private:

  template <class Derived>
  friend class OpenMPParallelReduceCommon;

  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy = typename MDRangePolicy::impl_range_policy;
  using Member = typename Policy::member_type;
  using reducer = ReducerType;

  using iterate_type =
    Kokkos::Impl::HostIterateTile<
      MDRangePolicy,
      FunctorType,
      Concepts::execution_policy_work_tag_t<MDRangePolicy>,
      Concepts::reducer_reference_type_t<ReducerType>
    >;

  OpenMPExec* m_instance;
  FunctorType const m_functor;
  MDRangePolicy const m_mdr_policy;
  Policy const m_policy;     // construct as RangePolicy( 0, num_tiles ).set_chunk_size(1) in ctor
  ReducerStorage<ReducerType> const m_reducer_storage;

  inline void
  exec_range(
    HostThreadTeamData&,
    Member ibeg,
    Member iend,
    Concepts::reducer_reference_type_t<ReducerType> update
  ) const
  {
    auto begin = ibeg + m_policy.begin();
    auto end = iend + m_policy.begin();
    for(Member iwork = ibeg; iwork < end; ++iwork) {
      iterate_type(m_mdr_policy, m_functor, update)(iwork);
    }
  }

public:

  template <class ViewType>
  inline
  ParallelReduce(
    FunctorType arg_functor,
    MDRangePolicy arg_policy,
    ViewType const& arg_view,
    typename std::enable_if<
      Kokkos::is_view<ViewType>::value
        && !Kokkos::is_reducer_type<ViewType>::value,
      void const**
    >::type = nullptr
  ) : m_instance(t_openmp_instance),
      m_functor(arg_functor),
      m_mdr_policy(arg_policy),
      m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
      m_reducer_storage(m_functor, m_mdr_policy, arg_view)
  { }

  inline
  ParallelReduce(
    FunctorType arg_functor,
    MDRangePolicy arg_policy,
    ReducerType const& reducer
  ) : m_instance(t_openmp_instance),
      m_functor(std::move(arg_functor)),
      m_mdr_policy(arg_policy),
      m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
      m_reducer_storage(reducer)
  { }

};

//==============================================================================

// TODO unify this with the Range implementation
template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<
  FunctorType,
  Kokkos::TeamPolicy<Properties...>,
  ReducerType,
  Kokkos::OpenMP
> : public OpenMPParallelReduceCommon<
      ParallelReduce<
        FunctorType,
        Kokkos::TeamPolicy<Properties...>,
        ReducerType,
        Kokkos::OpenMP
      >
    >
{
private:

  template <class Derived>
  friend class OpenMPParallelReduceCommon;

  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy = Kokkos::TeamPolicy<Properties...>;
  using Member = typename Policy::member_type;
  using reducer = ReducerType;

  OpenMPExec* m_instance;
  FunctorType const m_functor;
  Policy const m_policy;
  ReducerStorage<ReducerType> const m_reducer_storage;
  int m_shmem_size;

  inline void
  exec_range(
    HostThreadTeamData& data,
    int league_rank_begin,
    int league_rank_end,
    Concepts::reducer_reference_type_t<ReducerType>& update
  ) const
  {
    const int league_size = m_policy.league_size();
    for(int r = league_rank_begin; r < league_rank_end; /* intentionally empty */) {

      Concepts::functor_invoke_with_policy(
        m_policy, m_functor, Member{data, r, league_size}, update
      );

      if(++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if(data.team_rendezvous()) { data.team_rendezvous_release(); }
      }
    }
  }

  inline void resize_thread_data() const
  {
    const size_t pool_reduce_size =
      Concepts::reducer_value_size(m_reducer_storage.get_reducer());

    auto shmem_size =
      m_policy.scratch_size(0)
        + m_policy.scratch_size(1)
        + Concepts::functor_team_shmem_size(m_functor, m_policy.team_size());

    const size_t team_reduce_size = TEAM_REDUCE_SIZE * m_policy.team_size();
    const size_t team_shared_size = shmem_size + m_policy.scratch_size(1);
    const size_t thread_local_size = 0; // Never shrinks

    m_instance->resize_thread_data(
      pool_reduce_size,
      team_reduce_size,
      team_shared_size,
      thread_local_size
    );
  }

  inline bool
  organize_team(HostThreadTeamData& data) const
  {
    return static_cast<bool>(
      data.organize_team(m_policy.team_size())
    );
  }

  inline void
  set_work_partition(HostThreadTeamData& data) const
  {
    data.set_work_partition(
      m_policy.league_size(),
      ( 0 < m_policy.chunk_size()
        ? m_policy.chunk_size()
        : m_policy.team_iter()
      )
    );
  }

  inline void
  disband_team(HostThreadTeamData& data) const
  {
    data.disband_team();

    // TODO move this to a separate function?

    //  This thread has updated 'pool_reduce_local()' with its
    //  contributions to the reduction.  The parallel region is
    //  about to terminate and the master thread will load and
    //  reduce each 'pool_reduce_local()' contribution.
    //  Must 'memory_fence()' to guarantee that storing the update to
    //  'pool_reduce_local()' will complete before this thread
    //  exits the parallel region.

    memory_fence();
  }

public:

  //----------------------------------------

  template <class ViewType>
  inline
  ParallelReduce(
    FunctorType arg_functor,
    Policy arg_policy,
    ViewType arg_result,
    typename std::enable_if<
      Kokkos::is_view<ViewType>::value &&
        !Kokkos::is_reducer_type<ViewType>::value,
      void const**
    >::type = nullptr
  ) : m_instance(t_openmp_instance),
      m_functor(std::move(arg_functor)),
      m_policy(std::move(arg_policy)),
      m_reducer_storage(m_functor, m_policy, arg_result)
  { }

  inline
  ParallelReduce(
    FunctorType arg_functor,
    Policy arg_policy,
    ReducerType arg_reducer
  ) : m_instance(t_openmp_instance),
      m_functor(std::move(arg_functor)),
      m_policy(std::move(arg_policy)),
      m_reducer_storage(std::move(arg_reducer))
  { }

};

//==============================================================================

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif // KOKKOS_ENABLE_OPENMP
#endif //KOKKOS_OPENMP_KOKKOS_OPENMP_PARALLEL_REDUCE_HPP
