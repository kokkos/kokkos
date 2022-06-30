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

#ifndef KOKKOS_OPENACCEXEC_HPP
#define KOKKOS_OPENACCEXEC_HPP

#include <openacc.h>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Spinwait.hpp>
#include <Kokkos_Atomic.hpp>
//#include "Kokkos_OpenACC_Abort.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

//----------------------------------------------------------------------------
/** \brief  Data for OpenMP thread execution */

class OpenACCExec {
 public:
  // FIXME_OPENACC - Currently the maximum number of
  // teams possible is calculated based on NVIDIA's Volta GPU. In
  // future this value should be based on the chosen architecture for the
  // OpenACC backend.
  constexpr static int MAX_ACTIVE_THREADS = 2080 * 80;
  constexpr static int MAX_ACTIVE_TEAMS   = MAX_ACTIVE_THREADS / 32;

 private:
  static void* scratch_ptr;

 public:
  static void verify_is_process(const char* const);
  static void verify_initialized(const char* const);

  static int* get_lock_array(int num_teams);
  static void* get_scratch_ptr();
  static void clear_scratch();
  static void clear_lock_array();
  static void resize_scratch(int64_t team_reduce_bytes,
                             int64_t team_shared_bytes,
                             int64_t thread_local_bytes);

  static void* m_scratch_ptr;
  static int64_t m_scratch_size;
  static int* m_lock_array;
  static int64_t m_lock_size;
  static uint32_t* m_uniquetoken_ptr;
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class OpenACCExecTeamMember {
 public:
  constexpr static int TEAM_REDUCE_SIZE = 512;

  /** \brief  Thread states for team synchronization */
  enum { Active = 0, Rendezvous = 1 };

  using execution_space      = Kokkos::Experimental::OpenACC;
  using scratch_memory_space = execution_space::scratch_memory_space;

  scratch_memory_space m_team_shared;
  int m_team_scratch_size[2];
  int m_team_rank;
  int m_team_size;
  int m_league_rank;
  int m_league_size;
  int m_vector_length;
  int m_vector_lane;
  int m_shmem_block_index;
  void* m_glb_scratch;
  void* m_reduce_scratch;

  /*
  // Fan-in team threads, root of the fan-in which does not block returns true
  bool team_fan_in() const
    {
      memory_fence();
      for ( int n = 1 , j ; ( ( j = m_team_rank_rev + n ) < m_team_size ) && ! (
  m_team_rank_rev & n ) ; n <<= 1 ) {

        m_exec.pool_rev( m_team_base_rev + j )->state_wait( Active );
      }

      if ( m_team_rank_rev ) {
        m_exec.state_set( Rendezvous );
        memory_fence();
        m_exec.state_wait( Rendezvous );
      }

      return 0 == m_team_rank_rev ;
    }

  void team_fan_out() const
    {
      memory_fence();
      for ( int n = 1 , j ; ( ( j = m_team_rank_rev + n ) < m_team_size ) && ! (
  m_team_rank_rev & n ) ; n <<= 1 ) { m_exec.pool_rev( m_team_base_rev + j
  )->state_set( Active ); memory_fence();
      }
    }
  */
 public:
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& team_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, 1,
                                              m_team_scratch_size[level]);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& thread_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const {
#ifdef __USE_NVHPC__
    if (m_team_size <= 1) {
      return m_team_rank;
    } else {
      return __pgi_vectoridx();
    }
#else
    return m_team_rank;
#endif
  }
  KOKKOS_INLINE_FUNCTION int vector_length() const { return m_vector_length; }
  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }
  KOKKOS_INLINE_FUNCTION void* impl_reduce_scratch() const {
    return m_reduce_scratch;
  }

  KOKKOS_INLINE_FUNCTION void team_barrier() const {
    //[FIXME-SL] OpenACC does not provide any explicit barrier constructs for
    // device kernels.
    std::cerr
        << "Kokkos::Experimental::OpenACC ERROR: OpenACC does not provide any "
           "explicit barrier constructs for device kernels; exit!"
        << std::endl;
    exit(0);
  }

  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType& value,
                                             int thread_id) const {
    // Make sure there is enough scratch space:
    using type =
        typename std::conditional<(sizeof(ValueType) < TEAM_REDUCE_SIZE),
                                  ValueType, void>::type;
    type* team_scratch = reinterpret_cast<type*>(
        ((char*)(m_glb_scratch) + TEAM_REDUCE_SIZE * league_rank()));
    // FIXME_OPENACC
    //#pragma acc barrier
    if (team_rank() == thread_id) *team_scratch = value;
    // FIXME_OPENACC
    //#pragma acc barrier
    value = *team_scratch;
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(const Closure& f, ValueType& value,
                                             const int& thread_id) const {
    f(value);
    team_broadcast(value, thread_id);
  }

  template <class ValueType, class JoinOp>
  KOKKOS_INLINE_FUNCTION ValueType team_reduce(const ValueType& value,
                                               const JoinOp& op_in) const {
    // FIXME_OPENACC
    //#pragma acc barrier

    using value_type = ValueType;
    const JoinLambdaAdapter<value_type, JoinOp> op(op_in);

    // Make sure there is enough scratch space:
    using type = std::conditional_t<(sizeof(value_type) < TEAM_REDUCE_SIZE),
                                    value_type, void>;

    const int n_values = TEAM_REDUCE_SIZE / sizeof(value_type);
    type* team_scratch =
        (type*)((char*)m_glb_scratch + TEAM_REDUCE_SIZE * league_rank());
    for (int i = m_team_rank; i < n_values; i += m_team_size) {
      team_scratch[i] = value_type();
    }

    // FIXME_OPENACC
    //#pragma acc barrier

    for (int k = 0; k < m_team_size; k += n_values) {
      if ((k <= m_team_rank) && (k + n_values > m_team_rank))
        team_scratch[m_team_rank % n_values] += value;
      // FIXME_OPENACC
      //#pragma acc barrier
    }

    for (int d = 1; d < n_values; d *= 2) {
      if ((m_team_rank + d < n_values) && (m_team_rank % (2 * d) == 0)) {
        team_scratch[m_team_rank] += team_scratch[m_team_rank + d];
      }
      // FIXME_OPENACC
      //#pragma acc barrier
    }
    return team_scratch[0];
  }
  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
   *          with intra-team non-deterministic ordering accumulation.
   *
   *  The global inter-team accumulation value will, at the end of the
   *  league's parallel execution, be the scan's total.
   *  Parallel execution ordering of the league's teams is non-deterministic.
   *  As such the base value for each team's scan operation is similarly
   *  non-deterministic.
   */
  template <typename ArgType>
  KOKKOS_INLINE_FUNCTION ArgType
  team_scan(const ArgType& /*value*/, ArgType* const /*global_accum*/) const {
    // FIXME_OPENACC
    /*  // Make sure there is enough scratch space:
      using type =
        std::conditional_t<(sizeof(ArgType) < TEAM_REDUCE_SIZE), ArgType, void>;

      volatile type * const work_value  = ((type*) m_exec.scratch_thread());

      *work_value = value ;

      memory_fence();

      if ( team_fan_in() ) {
        // The last thread to synchronize returns true, all other threads wait
      for team_fan_out()
        // m_team_base[0]                 == highest ranking team member
        // m_team_base[ m_team_size - 1 ] == lowest ranking team member
        //
        // 1) copy from lower to higher rank, initialize lowest rank to zero
        // 2) prefix sum from lowest to highest rank, skipping lowest rank

        type accum = 0 ;

        if ( global_accum ) {
          for ( int i = m_team_size ; i-- ; ) {
            type & val = *((type*) m_exec.pool_rev( m_team_base_rev + i
      )->scratch_thread()); accum += val ;
          }
          accum = atomic_fetch_add( global_accum , accum );
        }

        for ( int i = m_team_size ; i-- ; ) {
          type & val = *((type*) m_exec.pool_rev( m_team_base_rev + i
      )->scratch_thread()); const type offset = accum ; accum += val ; val =
      offset ;
        }

        memory_fence();
      }

      team_fan_out();

      return *work_value ;*/
    return ArgType();
  }

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
   *
   *  The highest rank thread can compute the reduction total as
   *    reduction_total = dev.team_scan( value ) + value ;
   */
  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type team_scan(const Type& value) const {
    return this->template team_scan<Type>(value, 0);
  }

  //----------------------------------------
  // Private for the driver

 private:
  using space = execution_space::scratch_memory_space;

 public:
  // FIXME_OPENACC - 512(16*32) bytes at the begining of the scratch space
  // for each league is saved for reduction. It should actually be based on the
  // ValueType of the reduction variable.
  OpenACCExecTeamMember(
      const int league_rank, const int league_size, const int team_size,
      const int vector_length)  // const TeamPolicyInternal< OpenACC,
                                // Properties ...> & team
      : m_team_rank(0),
        m_team_size(team_size),
        m_league_rank(league_rank),
        m_league_size(league_size),
        m_vector_length(vector_length) {
    m_league_rank = league_rank;
    // m_team_rank   = omp_tid;
    // m_vector_lane = 0;
  }

  static int team_reduce_size() { return TEAM_REDUCE_SIZE; }
};

template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::OpenACC, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  //----------------------------------------

  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelForTag&) {
    return 256;
  }

  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelReduceTag&) {
    return 256;
  }

  template <class FunctorType, class ReducerType>
  static int team_size_max(const FunctorType&, const ReducerType&,
                           const ParallelReduceTag&) {
    return 256;
  }

  template <class FunctorType>
  static int team_size_recommended(const FunctorType&, const ParallelForTag&) {
    return 128;
  }

  template <class FunctorType>
  static int team_size_recommended(const FunctorType&,
                                   const ParallelReduceTag&) {
    return 128;
  }

  template <class FunctorType, class ReducerType>
  static int team_size_recommended(const FunctorType&, const ReducerType&,
                                   const ParallelReduceTag&) {
    return 128;
  }

  //----------------------------------------

 private:
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  int m_team_alloc;
  int m_team_iter;
  std::array<size_t, 2> m_team_scratch_size;
  std::array<size_t, 2> m_thread_scratch_size;
  bool m_tune_team_size;
  bool m_tune_vector_length;
  constexpr const static size_t default_team_size = 128;
  int m_chunk_size;

  void init(const int league_size_request, const int team_size_request,
            const int vector_length_request) {
    m_league_size = league_size_request;

    // Minimum team size should be 32 for OpenACC backend.
    if (team_size_request < 32) {
      fprintf(stderr, "%s.\n",
              "OpenACC backend requires a minimum of 32 threads per team.\n");
      std::abort();

    } else
      m_team_size = team_size_request;

    m_vector_length = vector_length_request;
    set_auto_chunk_size();
  }

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 public:
  bool impl_auto_team_size() const { return m_tune_team_size; }
  bool impl_auto_vector_length() const { return m_tune_vector_length; }
  void impl_set_team_size(const size_t size) { m_team_size = size; }
  void impl_set_vector_length(const size_t length) {
    m_tune_vector_length = length;
  }
  int impl_vector_length() const { return m_vector_length; }
  KOKKOS_DEPRECATED int vector_length() const { return impl_vector_length(); }
  int team_size() const { return m_team_size; }
  int league_size() const { return m_league_size; }
  size_t scratch_size(const int& level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  Kokkos::Experimental::OpenACC space() const {
    return Kokkos::Experimental::OpenACC();
  }

  template <class... OtherProperties>
  TeamPolicyInternal(const TeamPolicyInternal<OtherProperties...>& p)
      : m_league_size(p.m_league_size),
        m_team_size(p.m_team_size),
        m_vector_length(p.m_vector_length),
        m_team_alloc(p.m_team_alloc),
        m_team_iter(p.m_team_iter),
        m_team_scratch_size(p.m_team_scratch_size),
        m_thread_scratch_size(p.m_thread_scratch_size),
        m_tune_team_size(p.m_tune_team_size),
        m_tune_vector_length(p.m_tune_vector_length),
        m_chunk_size(p.m_chunk_size) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(false),
        m_tune_vector_length(false),
        m_chunk_size(0) {
    init(league_size_request, team_size_request, vector_length_request);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(true),
        m_tune_vector_length(false),
        m_chunk_size(0) {
    init(league_size_request, default_team_size / vector_length_request,
         vector_length_request);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(true),
        m_tune_vector_length(true),
        m_chunk_size(0) {
    init(league_size_request, default_team_size, 1);
  }
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(false),
        m_tune_vector_length(true),
        m_chunk_size(0) {
    init(league_size_request, team_size_request, 1);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(false),
        m_tune_vector_length(false),
        m_chunk_size(0) {
    init(league_size_request, team_size_request, vector_length_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(true),
        m_tune_vector_length(false),
        m_chunk_size(0) {
    init(league_size_request, default_team_size / vector_length_request,
         vector_length_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(true),
        m_tune_vector_length(true),
        m_chunk_size(0) {
    init(league_size_request, default_team_size, 1);
  }
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_tune_team_size(false),
        m_tune_vector_length(true),
        m_chunk_size(0) {
    init(league_size_request, team_size_request, 1);
  }
  static size_t vector_length_max() {
    return 32; /* TODO: this is bad. Need logic that is compiler and backend
                  aware */
  }
  int team_alloc() const { return m_team_alloc; }
  int team_iter() const { return m_team_iter; }

  int chunk_size() const { return m_chunk_size; }

  /** \brief set chunk_size to a discrete value*/
  TeamPolicyInternal& set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerTeamValue& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerThreadValue& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerTeamValue& per_team,
                                       const PerThreadValue& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

 private:
  /** \brief finalize chunk_size if it was set to AUTO*/
  void set_auto_chunk_size() {
    int concurrency = 2048 * 128;

    if (concurrency == 0) concurrency = 1;

    if (m_chunk_size > 0) {
      if (!Impl::is_integral_power_of_two(m_chunk_size))
        Kokkos::abort("TeamPolicy blocking granularity must be power of two");
    }

    int new_chunk_size = 1;
    while (new_chunk_size * 100 * concurrency < m_league_size)
      new_chunk_size *= 2;
    if (new_chunk_size < 128) {
      new_chunk_size = 1;
      while ((new_chunk_size * 40 * concurrency < m_league_size) &&
             (new_chunk_size < 128))
        new_chunk_size *= 2;
    }
    m_chunk_size = new_chunk_size;
  }

 public:
  using member_type = Impl::OpenACCExecTeamMember;
};
}  // namespace Impl

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

// Single Range reducer operations
template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapper {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<Sum<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) { dest += src; }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest += src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(+ : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(+ : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<Prod<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) { dest *= src; }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest *= src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(* : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(* : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<Min<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src < dest) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src < dest) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(min : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(min : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<Max<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src > dest) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src > dest) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(max : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(max : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<LAnd<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    dest = dest && src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest = dest && src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(&& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(&& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<LOr<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

  using result_view_type = Kokkos::View<value_type, Space>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    dest = dest || src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest = dest || src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(|| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(|| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<BAnd<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    dest = dest & src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest = dest & src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<BOr<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    dest = dest | src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    dest = dest | src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Index, class Space, class FunctorType,
          class TagType, class... Traits>
struct OpenACCReducerWrapper<MinLoc<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 private:
  using scalar_type = typename std::remove_cv<Scalar>::type;
  using index_type  = typename std::remove_cv<Index>::type;

 public:
  using value_type = ValLocScalar<scalar_type, index_type>;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src.val < dest.val) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src.val < dest.val) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val.val = reduction_identity<scalar_type>::min();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    Kokkos::abort(
        "[ERROR in reduce()] MinLoc reduce type is not supported in the "
        "OpenACC backend implementation.\n");
  }
};

template <class Scalar, class Index, class Space, class FunctorType,
          class TagType, class... Traits>
struct OpenACCReducerWrapper<MaxLoc<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 private:
  using scalar_type = typename std::remove_cv<Scalar>::type;
  using index_type  = typename std::remove_cv<Index>::type;

 public:
  using value_type = ValLocScalar<scalar_type, index_type>;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src.val > dest.val) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src.val > dest.val) dest = src;
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val.val = reduction_identity<scalar_type>::max();
    val.loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    Kokkos::abort(
        "[ERROR in reduce()] MaxLoc reduce type is not supported in the "
        "OpenACC backend implementation.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapper<MinMax<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 private:
  using scalar_type = typename std::remove_cv<Scalar>::type;

 public:
  using value_type = MinMaxScalar<scalar_type>;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
    }
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
    }
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    Kokkos::abort(
        "[ERROR in reduce()] MinMax reduce type is not supported in the "
        "OpenACC backend implementation.\n");
  }
};

template <class Scalar, class Index, class Space, class FunctorType,
          class TagType, class... Traits>
struct OpenACCReducerWrapper<MinMaxLoc<Scalar, Space>, FunctorType,
                             Kokkos::RangePolicy<Traits...>, TagType> {
 private:
  using scalar_type = typename std::remove_cv<Scalar>::type;
  using index_type  = typename std::remove_cv<Index>::type;

 public:
  using value_type = MinMaxLocScalar<scalar_type, index_type>;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type& dest, const value_type& src) {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    }
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type& dest, const volatile value_type& src) {
    if (src.min_val < dest.min_val) {
      dest.min_val = src.min_val;
      dest.min_loc = src.min_loc;
    }
    if (src.max_val > dest.max_val) {
      dest.max_val = src.max_val;
      dest.max_loc = src.max_loc;
    }
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val.max_val = reduction_identity<scalar_type>::max();
    val.min_val = reduction_identity<scalar_type>::min();
    val.max_loc = reduction_identity<index_type>::min();
    val.min_loc = reduction_identity<index_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    Kokkos::abort(
        "[ERROR in reduce()] MinMaxLocScalar reduce type is not supported in "
        "the OpenACC backend implementation.\n");
  }
};

// Multi-Dimensional Range reducer operations - Rank=2

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank2 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank2<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[1];
    int end1   = m_policy.m_upper[1];
    int begin2 = m_policy.m_lower[0];
    int end2   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(2) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(i1, i0, ltmp);
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(2) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          a_functor(TagType(), i1, i0, ltmp);
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=3

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank3 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  // Required
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank3<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[2];
    int end1   = m_policy.m_upper[2];
    int begin2 = m_policy.m_lower[1];
    int end2   = m_policy.m_upper[1];
    int begin3 = m_policy.m_lower[0];
    int end3   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(3) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(i2, i1, i0, ltmp);
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(3) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            a_functor(TagType(), i2, i1, i0, ltmp);
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=4

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank4 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i2, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[1];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  // Required
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank4<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[3];
    int end1   = m_policy.m_upper[3];
    int begin2 = m_policy.m_lower[2];
    int end2   = m_policy.m_upper[2];
    int begin3 = m_policy.m_lower[1];
    int end3   = m_policy.m_upper[1];
    int begin4 = m_policy.m_lower[0];
    int end4   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(4) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(4) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              a_functor(TagType(), i3, i2, i1, i0, ltmp);
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=5

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank5 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank5<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[4];
    int end1   = m_policy.m_upper[4];
    int begin2 = m_policy.m_lower[3];
    int end2   = m_policy.m_upper[3];
    int begin3 = m_policy.m_lower[2];
    int end3   = m_policy.m_upper[2];
    int begin4 = m_policy.m_lower[1];
    int end4   = m_policy.m_upper[1];
    int begin5 = m_policy.m_lower[0];
    int end5   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(5) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(5) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                a_functor(TagType(), i4, i3, i2, i1, i0, ltmp);
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=6

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank6 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank6<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[5];
    int end1   = m_policy.m_upper[5];
    int begin2 = m_policy.m_lower[4];
    int end2   = m_policy.m_upper[4];
    int begin3 = m_policy.m_lower[3];
    int end3   = m_policy.m_upper[3];
    int begin4 = m_policy.m_lower[2];
    int end4   = m_policy.m_upper[2];
    int begin5 = m_policy.m_lower[1];
    int end5   = m_policy.m_upper[1];
    int begin6 = m_policy.m_lower[0];
    int end6   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(6) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(6) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  a_functor(TagType(), i5, i4, i3, i2, i1, i0, ltmp);
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=7

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank7 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank7<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[6];
    int end1   = m_policy.m_upper[6];
    int begin2 = m_policy.m_lower[5];
    int end2   = m_policy.m_upper[5];
    int begin3 = m_policy.m_lower[4];
    int end3   = m_policy.m_upper[4];
    int begin4 = m_policy.m_lower[3];
    int end4   = m_policy.m_upper[3];
    int begin5 = m_policy.m_lower[2];
    int end5   = m_policy.m_upper[2];
    int begin6 = m_policy.m_lower[1];
    int end6   = m_policy.m_upper[1];
    int begin7 = m_policy.m_lower[0];
    int end7   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(7) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(7) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    a_functor(TagType(), i6, i5, i4, i3, i2, i1, i0, ltmp);
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Multi-Dimensional Range reducer operations - Rank=8

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperMD_Rank8 {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<Sum<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(+:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<Prod<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(*:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<Min<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(min     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<Max<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(max     \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<LAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(&&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<LOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(||      \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<BAnd<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(&:ltmp) copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperMD_Rank8<BOr<Scalar, Space>, FunctorType,
                                     Kokkos::MDRangePolicy<Traits...>,
                                     TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::MDRangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    value_type ltmp;
    init(ltmp);
    int begin1 = m_policy.m_lower[7];
    int end1   = m_policy.m_upper[7];
    int begin2 = m_policy.m_lower[6];
    int end2   = m_policy.m_upper[6];
    int begin3 = m_policy.m_lower[5];
    int end3   = m_policy.m_upper[5];
    int begin4 = m_policy.m_lower[4];
    int end4   = m_policy.m_upper[4];
    int begin5 = m_policy.m_lower[3];
    int end5   = m_policy.m_upper[3];
    int begin6 = m_policy.m_lower[2];
    int end6   = m_policy.m_upper[2];
    int begin7 = m_policy.m_lower[1];
    int end7   = m_policy.m_upper[1];
    int begin8 = m_policy.m_lower[0];
    int end8   = m_policy.m_upper[0];
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector collapse(8) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(i7, i6, i5, i4, i4, i2, i1, i0, ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
#pragma acc parallel loop gang vector collapse(8) reduction(|       \
                                                            : ltmp) \
    copyin(a_functor)
      for (auto i0 = begin1; i0 < end1; i0++) {
        for (auto i1 = begin2; i1 < end2; i1++) {
          for (auto i2 = begin3; i2 < end3; i2++) {
            for (auto i3 = begin4; i3 < end4; i3++) {
              for (auto i4 = begin5; i4 < end5; i4++) {
                for (auto i5 = begin6; i5 < end6; i5++) {
                  for (auto i6 = begin7; i6 < end7; i6++) {
                    for (auto i7 = begin8; i7 < end8; i7++) {
                      a_functor(TagType(), i7, i6, i5, i4, i3, i2, i1, i0,
                                ltmp);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    tmp = ltmp;
  }
};

// Hierarchical Parallelism Range reducer operations

template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReducerWrapperTeams {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(value_type&, const value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void join(volatile value_type&, const volatile value_type&) {
    Kokkos::abort(
        "[ERROR in join()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in init()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] Using a generic unknown Reducer for the OpenACC "
        "backend is not "
        "implemented.\n");
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<Sum<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(+ : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<Prod<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(* : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<Min<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(min : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<Max<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(max : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<LAnd<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(&& : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<LOr<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(|| : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<BAnd<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(& : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReducerWrapperTeams<BOr<Scalar, Space>, FunctorType,
                                  Kokkos::Impl::TeamPolicyInternal<Traits...>,
                                  TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::Impl::TeamPolicyInternal<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type ftmp;
    init(ftmp);

#pragma acc parallel loop gang reduction(| : ftmp) copyin(a_functor)
    //#pragma acc loop seq
    for (int i = 0; i < league_size; i++) {
      int league_id = i;
      typename Policy::member_type team(league_id, league_size, team_size,
                                        vector_length);
      if constexpr (std::is_same<TagType, void>::value)
        a_functor(team, ftmp);
      else
        a_functor(TagType(), team, ftmp);
    }
    tmp = ftmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

// Hierarchical Parallelism -> Team thread level implementation
namespace Kokkos {

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine worker
#endif
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
  iType j_start =
      loop_boundaries.team.team_rank() / loop_boundaries.team.vector_length();
  iType j_end  = loop_boundaries.end;
  iType j_step = loop_boundaries.team.team_size();
  if (j_start >= loop_boundaries.start) {
#pragma acc loop seq
    for (iType j = j_start; j < j_end; j += j_step) {
      lambda(j);
    }
  }
#else
//#pragma acc loop seq
#pragma acc loop worker
  for (iType j = loop_boundaries.start; j < loop_boundaries.end; j++) {
    lambda(j);
  }
#endif
}

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine worker
#endif
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<!Kokkos::is_reducer_type<ValueType>::value>
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
                    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc loop seq
#else
#pragma acc loop worker reduction(+ : tmp)
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine worker
#endif
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<Kokkos::is_reducer_type<ReducerType>::value>
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
                    const Lambda& lambda, ReducerType& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp = ValueType();
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc loop seq
#else
#pragma acc loop worker reduction(+ : tmp)
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  /*
  ValueType* TeamThread_scratch =
      static_cast<ValueType*>(loop_boundaries.team.impl_reduce_scratch());

  // FIXME_OPENACC - Make sure that if its an array reduction, number of
  // elements in the array <= 32. For reduction we allocate, 16 bytes per
  // element in the scratch space, hence, 16*32 = 512.
  static_assert(sizeof(ValueType) <=
                Impl::OpenACCExecTeamMember::TEAM_REDUCE_SIZE);

//FIXME_OPENACC
#pragma acc barrier
  TeamThread_scratch[0] = init_result;
//FIXME_OPENACC
#pragma acc barrier

  if constexpr (std::is_arithmetic<ValueType>::value) {
#pragma omp for reduction(+ : TeamThread_scratch[:1])
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      ValueType tmp = ValueType();
      lambda(i, tmp);
      TeamThread_scratch[0] += tmp;
    }
  } else {
#pragma omp declare reduction(custom:ValueType : omp_out += omp_in)

#pragma omp for reduction(custom : TeamThread_scratch[:1])
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      ValueType tmp = ValueType();
      lambda(i, tmp);
      join(TeamThread_scratch[0], tmp);
    }
  }

  init_result = TeamThread_scratch[0];
  */
}

// This is largely the same code as in HIP and CUDA except for the member name
template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_bounds,
    const FunctorType& lambda) {
  /*
  // Extract value_type from lambda
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void,
      FunctorType>::value_type;

  const auto start = loop_bounds.start;
  const auto end   = loop_bounds.end;
  // Note this thing is called .member in the CUDA specialization of
  // TeamThreadRangeBoundariesStruct
  auto& member         = loop_bounds.team;
  const auto team_size = member.team_size();
  const auto team_rank = member.team_rank();
  const auto nchunk    = (end - start + team_size - 1) / team_size;
  value_type accum     = 0;
  // each team has to process one or more chunks of the prefix scan
  for (iType i = 0; i < nchunk; ++i) {
    auto ii = start + i * team_size + team_rank;
    // local accumulation for this chunk
    value_type local_accum = 0;
    // user updates value with prefix value
    if (ii < loop_bounds.end) lambda(ii, local_accum, false);
    // perform team scan
    local_accum = member.team_scan(local_accum);
    // add this blocks accum to total accumulation
    auto val = accum + local_accum;
    // user updates their data with total accumulation
    if (ii < loop_bounds.end) lambda(ii, val, true);
    // the last value needs to be propogated to next chunk
    if (team_rank == team_size - 1) accum = val;
    // broadcast last value to rest of the team
    member.team_broadcast(accum, team_size - 1);
  }
  */
}

}  // namespace Kokkos

// Hierarchical Parallelism -> Vector thread level implementation
namespace Kokkos {

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine vector
#endif
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  iType j_end  = loop_boundaries.end;
  iType j_step = loop_boundaries.team.vector_length();
  if (j_start >= loop_boundaries.start) {
#pragma acc loop seq
    for (iType j = j_start; j < j_end; j += j_step) {
      lambda(j);
    }
  }
#else
//#pragma acc loop seq
#pragma acc loop vector
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i);
  }
#endif
}

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine vector
#endif
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc loop seq
#else
#pragma acc loop vector reduction(+ : tmp)
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine vector
#endif
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<Kokkos::is_reducer_type<ReducerType>::value>
    parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
                    const Lambda& lambda, ReducerType const& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType vector_reduce;
  Impl::OpenACCReducerWrapper<ReducerType>::init(vector_reduce);

#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc loop seq
#else
#pragma acc loop vector reduction(+ : tmp)
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, vector_reduce);
  }
  result.reference() = vector_reduce;
}

template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  ValueType result = init_result;

  /*
  // FIXME_OPENACC: incorrect implementation
  //think about omp simd
  // join does not work with omp reduction clause
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    ValueType tmp = ValueType();
    lambda(i, tmp);
    join(result, tmp);
  }

  init_result = result;
  */
}

template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const FunctorType& lambda) {
  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, void>;
  using value_type  = typename ValueTraits::value_type;

  /*
    value_type scan_val = value_type();

  #ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
  #pragma ivdep
  #endif
    for (iType i = loop_boundaries.start; i < loop_boundaries.end;
         i += loop_boundaries.increment) {
      lambda(i, scan_val, true);
    }
  */
}
}  // namespace Kokkos

// Hierarchical Parallelism -> Team vector level implementation
namespace Kokkos {
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
#pragma acc routine seq
#else
#pragma acc routine vector
#endif
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#ifdef KOKKOS_ENABLE_COLLAPSE_HIERARCHICAL_CONSTRUCTS
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  iType j_end  = loop_boundaries.end;
  iType j_step = loop_boundaries.team.vector_length();
  if (j_start >= loop_boundaries.start) {
#pragma acc loop seq
    for (iType j = j_start; j < j_end; j += j_step) {
      lambda(j);
    }
  }
#else
//#pragma acc loop seq
#pragma acc loop vector
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
#endif
}

template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenACCExecTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

//#pragma acc loop vector reduction(+:tmp)
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::OpenACCExecTeamMember>
    TeamThreadRange(const Impl::OpenACCExecTeamMember& thread,
                    const iType& count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::OpenACCExecTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::OpenACCExecTeamMember>
TeamThreadRange(const Impl::OpenACCExecTeamMember& thread, const iType1& begin,
                const iType2& end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::OpenACCExecTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::OpenACCExecTeamMember>
    ThreadVectorRange(const Impl::OpenACCExecTeamMember& thread,
                      const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::OpenACCExecTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::ThreadVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::OpenACCExecTeamMember>
ThreadVectorRange(const Impl::OpenACCExecTeamMember& thread,
                  const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::OpenACCExecTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamVectorRangeBoundariesStruct<iType, Impl::OpenACCExecTeamMember>
    TeamVectorRange(const Impl::OpenACCExecTeamMember& thread,
                    const iType& count) {
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::OpenACCExecTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::OpenACCExecTeamMember>
TeamVectorRange(const Impl::OpenACCExecTeamMember& thread,
                const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::OpenACCExecTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::OpenACCExecTeamMember> PerTeam(
    const Impl::OpenACCExecTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::OpenACCExecTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::OpenACCExecTeamMember> PerThread(
    const Impl::OpenACCExecTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::OpenACCExecTeamMember>(thread);
}
}  // namespace Kokkos

namespace Kokkos {

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::OpenACCExecTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda) {
  lambda();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::OpenACCExecTeamMember>& single_struct,
    const FunctorType& lambda) {
  if (single_struct.team_member.team_rank() == 0) lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::OpenACCExecTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda, ValueType& val) {
  lambda(val);
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::OpenACCExecTeamMember>& single_struct,
    const FunctorType& lambda, ValueType& val) {
  if (single_struct.team_member.team_rank() == 0) {
    lambda(val);
  }
  single_struct.team_member.team_broadcast(val, 0);
}
}  // namespace Kokkos

#endif /* #ifndef KOKKOS_OPENACCEXEC_HPP */
