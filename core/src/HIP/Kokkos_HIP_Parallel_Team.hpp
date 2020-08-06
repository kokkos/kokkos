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

#ifndef KOKKO_HIP_PARALLEL_TEAM_HPP
#define KOKKO_HIP_PARALLEL_TEAM_HPP

#include <Kokkos_Parallel.hpp>

#if defined(__HIPCC__)

#include <HIP/Kokkos_HIP_KernelLaunch.hpp>
#include <HIP/Kokkos_HIP_Locks.hpp>
#include <HIP/Kokkos_HIP_Team.hpp>
#include <HIP/Kokkos_HIP_Instance.hpp>

namespace Kokkos {
namespace Impl {
template <typename... Properties>
class TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 private:
  static int constexpr MAX_WARP = 8;

  typename traits::execution_space m_space;
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  int m_team_scratch_size[2];
  int m_thread_scratch_size[2];
  int m_chunk_size;

 public:
  using execution_space = Kokkos::Experimental::HIP;

  template <class... OtherProperties>
  TeamPolicyInternal(TeamPolicyInternal<OtherProperties...> const& p) {
    m_league_size            = p.m_league_size;
    m_team_size              = p.m_team_size;
    m_vector_length          = p.m_vector_length;
    m_team_scratch_size[0]   = p.m_team_scratch_size[0];
    m_team_scratch_size[1]   = p.m_team_scratch_size[1];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size             = p.m_chunk_size;
    m_space                  = p.m_space;
  }

  template <typename FunctorType>
  int team_size_max(FunctorType const& f, ParallelForTag const&) const {
    using closure_type =
        Impl::ParallelFor<FunctorType, TeamPolicy<Properties...> >;
    hipFuncAttributes attr = ::Kokkos::Experimental::Impl::HIPParallelLaunch<
        closure_type,
        typename traits::launch_bounds>::get_hip_func_attributes();
    int const block_size = ::Kokkos::Experimental::Impl::hip_get_max_block_size<
        FunctorType, typename traits::launch_bounds>(
        space().impl_internal_space_instance(), attr, f,
        static_cast<size_t>(vector_length()),
        static_cast<size_t>(team_scratch_size(0)) + 2 * sizeof(double),
        static_cast<size_t>(thread_scratch_size(0)) + sizeof(double));
    return block_size / vector_length();
  }

  template <class FunctorType>
  inline int team_size_max(const FunctorType& f,
                           const ParallelReduceTag&) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, FunctorType>;
    using reducer_type = typename Impl::ParallelReduceReturnValue<
        void, typename functor_analysis_type::value_type,
        FunctorType>::reducer_type;
    using closure_type =
        Impl::ParallelReduce<FunctorType, TeamPolicy<Properties...>,
                             reducer_type>;
    return internal_team_size_max<closure_type>(f);
  }

  template <class FunctorType, class ReducerType>
  inline int team_size_max(const FunctorType& f, const ReducerType& /*r*/,
                           const ParallelReduceTag&) const {
    using closure_type =
        Impl::ParallelReduce<FunctorType, TeamPolicy<Properties...>,
                             ReducerType>;
    return internal_team_size_max<closure_type>(f);
  }

  template <typename FunctorType>
  int team_size_recommended(FunctorType const& f, ParallelForTag const&) const {
    using closure_type =
        Impl::ParallelFor<FunctorType, TeamPolicy<Properties...> >;
    hipFuncAttributes attr = ::Kokkos::Experimental::Impl::HIPParallelLaunch<
        closure_type,
        typename traits::launch_bounds>::get_hip_func_attributes();
    int const block_size = ::Kokkos::Experimental::Impl::hip_get_opt_block_size<
        FunctorType, typename traits::launch_bounds>(
        space().impl_internal_space_instance(), attr, f,
        static_cast<size_t>(vector_length()),
        static_cast<size_t>(team_scratch_size(0)) + 2 * sizeof(double),
        static_cast<size_t>(thread_scratch_size(0)) + sizeof(double));
    return block_size / vector_length();
  }

  template <typename FunctorType>
  inline int team_size_recommended(FunctorType const& f,
                                   ParallelReduceTag const&) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, FunctorType>;
    using reducer_type = typename Impl::ParallelReduceReturnValue<
        void, typename functor_analysis_type::value_type,
        FunctorType>::reducer_type;
    using closure_type =
        Impl::ParallelReduce<FunctorType, TeamPolicy<Properties...>,
                             reducer_type>;
    return internal_team_size_recommended<closure_type>(f);
  }

  template <class FunctorType, class ReducerType>
  int team_size_recommended(FunctorType const& f, ReducerType const&,
                            ParallelReduceTag const&) const {
    using closure_type =
        Impl::ParallelReduce<FunctorType, TeamPolicy<Properties...>,
                             ReducerType>;
    return internal_team_size_recommended<closure_type>(f);
  }

  static int vector_length_max() {
    return ::Kokkos::Experimental::Impl::HIPTraits::WarpSize;
  }

  static int verify_requested_vector_length(int requested_vector_length) {
    int test_vector_length =
        std::min(requested_vector_length, vector_length_max());

    // Allow only power-of-two vector_length
    if (!(is_integral_power_of_two(test_vector_length))) {
      int test_pow2           = 1;
      int constexpr warp_size = Experimental::Impl::HIPTraits::WarpSize;
      while (test_pow2 < warp_size) {
        test_pow2 <<= 1;
        if (test_pow2 > test_vector_length) {
          break;
        }
      }
      test_vector_length = test_pow2 >> 1;
    }

    return test_vector_length;
  }

  static int scratch_size_max(int level) {
    return (
        level == 0 ? 1024 * 40 :  // FIXME_HIP arbitrarily setting this to 48kB
            20 * 1024 * 1024);    // FIXME_HIP arbitrarily setting this to 20MB
  }

  int vector_length() const { return m_vector_length; }

  int team_size() const { return m_team_size; }

  int league_size() const { return m_league_size; }

  int scratch_size(int level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  int team_scratch_size(int level) const { return m_team_scratch_size[level]; }

  int thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  typename traits::execution_space space() const { return m_space; }

  TeamPolicyInternal()
      : m_space(typename traits::execution_space()),
        m_league_size(0),
        m_team_size(-1),
        m_vector_length(0),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     int team_size_request, int vector_length_request = 1)
      : m_space(space_),
        m_league_size(league_size_),
        m_team_size(team_size_request),
        m_vector_length(verify_requested_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {
    // Make sure league size is permissable
    if (league_size_ >=
        static_cast<int>(
            ::Kokkos::Experimental::Impl::hip_internal_maximum_grid_count()))
      Impl::throw_runtime_exception(
          "Requested too large league_size for TeamPolicy on HIP execution "
          "space.");

    // Make sure total block size is permissable
    if (m_team_size * m_vector_length > 1024) {
      Impl::throw_runtime_exception(
          std::string("Kokkos::TeamPolicy< HIP > the team size is too large. "
                      "Team size x vector length must be smaller than 1024."));
    }
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : m_space(space_),
        m_league_size(league_size_),
        m_team_size(-1),
        m_vector_length(verify_requested_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {
    // Make sure league size is permissable
    if (league_size_ >=
        static_cast<int>(
            ::Kokkos::Experimental::Impl::hip_internal_maximum_grid_count()))
      Impl::throw_runtime_exception(
          "Requested too large league_size for TeamPolicy on HIP execution "
          "space.");
  }

  TeamPolicyInternal(int league_size_, int team_size_request,
                     int vector_length_request = 1)
      : m_space(typename traits::execution_space()),
        m_league_size(league_size_),
        m_team_size(team_size_request),
        m_vector_length(verify_requested_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {
    // Make sure league size is permissable
    if (league_size_ >=
        static_cast<int>(
            ::Kokkos::Experimental::Impl::hip_internal_maximum_grid_count()))
      Impl::throw_runtime_exception(
          "Requested too large league_size for TeamPolicy on HIP execution "
          "space.");

    // Make sure total block size is permissable
    if (m_team_size * m_vector_length > 1024) {
      Impl::throw_runtime_exception(
          std::string("Kokkos::TeamPolicy< HIP > the team size is too large. "
                      "Team size x vector length must be smaller than 1024."));
    }
  }

  TeamPolicyInternal(int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : m_space(typename traits::execution_space()),
        m_league_size(league_size_),
        m_team_size(-1),
        m_vector_length(verify_requested_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(::Kokkos::Experimental::Impl::HIPTraits::WarpSize) {
    // Make sure league size is permissable
    if (league_size_ >=
        static_cast<int>(
            ::Kokkos::Experimental::Impl::hip_internal_maximum_grid_count()))
      Impl::throw_runtime_exception(
          "Requested too large league_size for TeamPolicy on HIP execution "
          "space.");
  }

  int chunk_size() const { return m_chunk_size; }

  TeamPolicyInternal& set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(int level,
                                       PerTeamValue const& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(int level,
                                       PerThreadValue const& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  TeamPolicyInternal& set_scratch_size(int level, PerTeamValue const& per_team,
                                       PerThreadValue const& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  using member_type = Kokkos::Impl::HIPTeamMember;

 protected:
  template <class ClosureType, class FunctorType, class BlockSizeCallable>
  int internal_team_size_common(const FunctorType& f,
                                BlockSizeCallable&& block_size_callable) const {
    using closure_type = ClosureType;
    using functor_value_traits =
        Impl::FunctorValueTraits<FunctorType, typename traits::work_tag>;

    hipFuncAttributes attr = ::Kokkos::Experimental::Impl::HIPParallelLaunch<
        closure_type,
        typename traits::launch_bounds>::get_hip_func_attributes();
    const int block_size = std::forward<BlockSizeCallable>(block_size_callable)(
        space().impl_internal_space_instance(), attr, f,
        static_cast<size_t>(vector_length()),
        static_cast<size_t>(team_scratch_size(0)) + 2 * sizeof(double),
        static_cast<size_t>(thread_scratch_size(0)) + sizeof(double) +
            ((functor_value_traits::StaticValueSize != 0)
                 ? 0
                 : functor_value_traits::value_size(f)));
    KOKKOS_ASSERT(block_size > 0);

    // Currently we require Power-of-2 team size for reductions.
    int p2 = 1;
    while (p2 <= block_size) p2 *= 2;
    p2 /= 2;
    return p2 / vector_length();
  }

  template <class ClosureType, class FunctorType>
  int internal_team_size_max(const FunctorType& f) const {
    return internal_team_size_common<ClosureType>(
        f, ::Kokkos::Experimental::Impl::hip_get_max_block_size<
               FunctorType, typename traits::launch_bounds>);
  }

  template <class ClosureType, class FunctorType>
  int internal_team_size_recommended(const FunctorType& f) const {
    return internal_team_size_common<ClosureType>(
        f, ::Kokkos::Experimental::Impl::hip_get_opt_block_size<
               FunctorType, typename traits::launch_bounds>);
  }
};

struct HIPLockArrays {
  std::int32_t* atomic  = nullptr;
  std::int32_t* scratch = nullptr;
  std::int32_t n        = 0;
};

template <typename FunctorType, typename... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::HIP> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>;
  using functor_type = FunctorType;
  using size_type    = ::Kokkos::Experimental::HIP::size_type;

 private:
  using member_type   = typename Policy::member_type;
  using work_tag      = typename Policy::work_tag;
  using launch_bounds = typename Policy::launch_bounds;

  // Algorithmic constraints: blockDim.y is a power of two AND
  // blockDim.y  == blockDim.z == 1 shared memory utilization:
  //
  //  [ team   reduce space ]
  //  [ team   shared space ]

  FunctorType const m_functor;
  Policy const m_policy;
  size_type const m_league_size;
  int m_team_size;
  size_type const m_vector_size;
  int m_shmem_begin;
  int m_shmem_size;
  void* m_scratch_ptr[2];
  int m_scratch_size[2];
  mutable HIPLockArrays hip_lock_arrays;

  template <typename TagType>
  __device__ inline
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_team(const member_type& member) const {
    m_functor(member);
  }

  template <typename TagType>
  __device__ inline
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_team(const member_type& member) const {
    m_functor(TagType(), member);
  }

 public:
  __device__ inline void operator()(void) const {
    // Iterate this block through the league
    int64_t threadid = 0;
    if (m_scratch_size[1] > 0) {
      __shared__ int64_t base_thread_id;
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        threadid = (blockIdx.x * blockDim.z + threadIdx.z) %
                   (hip_lock_arrays.n / (blockDim.x * blockDim.y));
        threadid *= blockDim.x * blockDim.y;
        int done = 0;
        while (!done) {
          done = (0 == atomicCAS(&hip_lock_arrays.scratch[threadid], 0, 1));
          if (!done) {
            threadid += blockDim.x * blockDim.y;
            if (int64_t(threadid + blockDim.x * blockDim.y) >=
                int64_t(hip_lock_arrays.n))
              threadid = 0;
          }
        }
        base_thread_id = threadid;
      }
      __syncthreads();
      threadid = base_thread_id;
    }

    int const int_league_size = static_cast<int>(m_league_size);
    for (int league_rank = blockIdx.x; league_rank < int_league_size;
         league_rank += gridDim.x) {
      this->template exec_team<work_tag>(typename Policy::member_type(
          ::Kokkos::Experimental::kokkos_impl_hip_shared_memory<void>(),
          m_shmem_begin, m_shmem_size,
          static_cast<void*>(static_cast<char*>(m_scratch_ptr[1]) +
                             ptrdiff_t(threadid / (blockDim.x * blockDim.y)) *
                                 m_scratch_size[1]),
          m_scratch_size[1], league_rank, m_league_size));
    }
    if (m_scratch_size[1] > 0) {
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0)
        hip_lock_arrays.scratch[threadid] = 0;
    }
  }

  inline void execute() const {
    HIP_SAFE_CALL(hipMalloc(
        &hip_lock_arrays.atomic,
        sizeof(std::int32_t) * (KOKKOS_IMPL_HIP_SPACE_ATOMIC_MASK + 1)));
    HIP_SAFE_CALL(hipMalloc(
        &hip_lock_arrays.scratch,
        sizeof(std::int32_t) * (::Kokkos::Experimental::HIP::concurrency())));
    HIP_SAFE_CALL(hipMemset(
        hip_lock_arrays.scratch, 0,
        sizeof(std::int32_t) * (::Kokkos::Experimental::HIP::concurrency())));
    hip_lock_arrays.n = ::Kokkos::Experimental::HIP::concurrency();

    int64_t const shmem_size_total = m_shmem_begin + m_shmem_size;
    dim3 const grid(static_cast<int>(m_league_size), 1, 1);
    dim3 const block(static_cast<int>(m_vector_size),
                     static_cast<int>(m_team_size), 1);

    ::Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelFor, launch_bounds>(
        *this, grid, block, shmem_size_total,
        m_policy.space().impl_internal_space_instance(),
        true);  // copy to device and execute

    if (hip_lock_arrays.atomic) {
      HIP_SAFE_CALL(hipFree(hip_lock_arrays.atomic));
      hip_lock_arrays.atomic = nullptr;
    }
    if (hip_lock_arrays.scratch) {
      HIP_SAFE_CALL(hipFree(hip_lock_arrays.scratch));
      hip_lock_arrays.scratch = nullptr;
    }
    hip_lock_arrays.n = 0;
  }

  ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.vector_length()) {
    hipFuncAttributes attr = ::Kokkos::Experimental::Impl::HIPParallelLaunch<
        ParallelFor, launch_bounds>::get_hip_func_attributes();
    m_team_size =
        m_team_size >= 0
            ? m_team_size
            : ::Kokkos::Experimental::Impl::hip_get_opt_block_size<
                  FunctorType, launch_bounds>(
                  m_policy.space().impl_internal_space_instance(), attr,
                  m_functor, m_vector_size, m_policy.team_scratch_size(0),
                  m_policy.thread_scratch_size(0)) /
                  m_vector_size;

    m_shmem_begin = (sizeof(double) * (m_team_size + 2));
    m_shmem_size =
        (m_policy.scratch_size(0, m_team_size) +
         FunctorTeamShmemSize<FunctorType>::value(m_functor, m_team_size));
    m_scratch_size[0] = m_policy.scratch_size(0, m_team_size);
    m_scratch_size[1] = m_policy.scratch_size(1, m_team_size);

    // Functor's reduce memory, team scan memory, and team shared memory depend
    // upon team size.
    m_scratch_ptr[0] = nullptr;
    m_scratch_ptr[1] =
        m_team_size <= 0
            ? nullptr
            : ::Kokkos::Experimental::Impl::hip_resize_scratch_space(
                  static_cast<ptrdiff_t>(m_scratch_size[1]) *
                  static_cast<ptrdiff_t>(
                      ::Kokkos::Experimental::HIP::concurrency() /
                      (m_team_size * m_vector_size)));

    int const shmem_size_total = m_shmem_begin + m_shmem_size;
    if (m_policy.space().impl_internal_space_instance()->m_maxShmemPerBlock <
        shmem_size_total) {
      printf(
          "%i %i\n",
          m_policy.space().impl_internal_space_instance()->m_maxShmemPerBlock,
          shmem_size_total);
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::ParallelFor< HIP > insufficient shared memory"));
    }

    if (static_cast<int>(m_team_size) >
        static_cast<int>(
            ::Kokkos::Experimental::Impl::hip_get_max_block_size<FunctorType,
                                                                 launch_bounds>(
                m_policy.space().impl_internal_space_instance(), attr,
                arg_functor, arg_policy.vector_length(),
                arg_policy.team_scratch_size(0),
                arg_policy.thread_scratch_size(0)) /
            arg_policy.vector_length())) {
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::ParallelFor< HIP > requested too large team size."));
    }
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::HIP> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::HIP, Properties...>;

 private:
  using member_type   = typename Policy::member_type;
  using work_tag      = typename Policy::work_tag;
  using launch_bounds = typename Policy::launch_bounds;

  using reducer_conditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using reducer_type_fwd = typename reducer_conditional::type;
  using work_tag_fwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  work_tag, void>::type;

  using value_traits =
      Kokkos::Impl::FunctorValueTraits<reducer_type_fwd, work_tag_fwd>;
  using value_init =
      Kokkos::Impl::FunctorValueInit<reducer_type_fwd, work_tag_fwd>;
  using value_join =
      Kokkos::Impl::FunctorValueJoin<reducer_type_fwd, work_tag_fwd>;

  using pointer_type   = typename value_traits::pointer_type;
  using reference_type = typename value_traits::reference_type;
  using value_type     = typename value_traits::value_type;

 public:
  using functor_type = FunctorType;
  using size_type    = Kokkos::Experimental::HIP::size_type;

  static int constexpr UseShflReduction = (value_traits::StaticValueSize != 0);

 private:
  using DummyShflReductionType  = double;
  using DummySHMEMReductionType = int;

  // Algorithmic constraints: blockDim.y is a power of two AND
  // blockDim.y == blockDim.z == 1 shared memory utilization:
  //
  //  [ global reduce space ]
  //  [ team   reduce space ]
  //  [ team   shared space ]
  //

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;
  size_type* m_scratch_space;
  size_type* m_scratch_flags;
  size_type m_team_begin;
  size_type m_shmem_begin;
  size_type m_shmem_size;
  void* m_scratch_ptr[2];
  int m_scratch_size[2];
  const size_type m_league_size;
  int m_team_size;
  const size_type m_vector_size;

  template <class TagType>
  __device__ inline
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      exec_team(member_type const& member, reference_type update) const {
    m_functor(member, update);
  }

  template <class TagType>
  __device__ inline
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      exec_team(member_type const& member, reference_type update) const {
    m_functor(TagType(), member, update);
  }

 public:
  __device__ inline void operator()() const {
    int64_t threadid = 0;
    if (m_scratch_size[1] > 0) {
      __shared__ int64_t base_thread_id;
      // FIXME_HIP This uses g_device_hip_lock_arrays which is not working
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        Impl::hip_abort("Error should not be here (not implemented yet)\n");
        threadid = (blockIdx.x * blockDim.z + threadIdx.z) %
                   (g_device_hip_lock_arrays.n / (blockDim.x * blockDim.y));
        threadid *= blockDim.x * blockDim.y;
        int done = 0;
        while (!done) {
          done = (0 ==
                  atomicCAS(&g_device_hip_lock_arrays.scratch[threadid], 0, 1));
          if (!done) {
            threadid += blockDim.x * blockDim.y;
            if (static_cast<int64_t>(threadid + blockDim.x * blockDim.y) >=
                static_cast<int64_t>(g_device_hip_lock_arrays.n))
              threadid = 0;
          }
        }
        base_thread_id = threadid;
      }
      __syncthreads();
      threadid = base_thread_id;
    }

    run(Kokkos::Impl::if_c<UseShflReduction, DummyShflReductionType,
                           DummySHMEMReductionType>::select(1, 1.0),
        threadid);
    if (m_scratch_size[1] > 0) {
      __syncthreads();
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        Impl::hip_abort("Error should not be here (not implemented yet)\n");
        g_device_hip_lock_arrays.scratch[threadid] = 0;
      }
    }
  }

  __device__ inline void run(DummySHMEMReductionType const&,
                             int const& threadid) const {
    integral_nonzero_constant<size_type, value_traits::StaticValueSize /
                                             sizeof(size_type)> const
        word_count(value_traits::value_size(
                       reducer_conditional::select(m_functor, m_reducer)) /
                   sizeof(size_type));

    reference_type value = value_init::init(
        reducer_conditional::select(m_functor, m_reducer),
        Kokkos::Experimental::kokkos_impl_hip_shared_memory<size_type>() +
            threadIdx.y * word_count.value);

    // Iterate this block through the league
    int const int_league_size = static_cast<int>(m_league_size);
    for (int league_rank = blockIdx.x; league_rank < int_league_size;
         league_rank += gridDim.x) {
      this->template exec_team<work_tag>(
          member_type(
              Kokkos::Experimental::kokkos_impl_hip_shared_memory<char>() +
                  m_team_begin,
              m_shmem_begin, m_shmem_size,
              reinterpret_cast<void*>(
                  reinterpret_cast<char*>(m_scratch_ptr[1]) +
                  static_cast<ptrdiff_t>(threadid / (blockDim.x * blockDim.y)) *
                      m_scratch_size[1]),
              m_scratch_size[1], league_rank, m_league_size),
          value);
    }

    // Reduce with final value at blockDim.y - 1 location.
    if (hip_single_inter_block_reduce_scan<false, FunctorType, work_tag>(
            reducer_conditional::select(m_functor, m_reducer), blockIdx.x,
            gridDim.x,
            Kokkos::Experimental::kokkos_impl_hip_shared_memory<size_type>(),
            m_scratch_space, m_scratch_flags)) {
      // This is the final block with the final result at the final threads'
      // location

      size_type* const shared =
          Kokkos::Experimental::kokkos_impl_hip_shared_memory<size_type>() +
          (blockDim.y - 1) * word_count.value;
      size_type* const global = m_result_ptr_device_accessible
                                    ? reinterpret_cast<size_type*>(m_result_ptr)
                                    : m_scratch_space;

      if (threadIdx.y == 0) {
        Kokkos::Impl::FunctorFinal<reducer_type_fwd, work_tag_fwd>::final(
            reducer_conditional::select(m_functor, m_reducer), shared);
      }

      if (Kokkos::Experimental::Impl::HIPTraits::WarpSize < word_count.value) {
        __syncthreads();
      }

      for (unsigned i = threadIdx.y; i < word_count.value; i += blockDim.y) {
        global[i] = shared[i];
      }
    }
  }

  __device__ inline void run(DummyShflReductionType const&,
                             int const& threadid) const {
    // FIXME_HIP implementation close to the function above
    value_type value;
    value_init::init(reducer_conditional::select(m_functor, m_reducer), &value);

    // Iterate this block through the league
    int const int_league_size = static_cast<int>(m_league_size);
    for (int league_rank = blockIdx.x; league_rank < int_league_size;
         league_rank += gridDim.x) {
      this->template exec_team<work_tag>(
          member_type(
              Kokkos::Experimental::kokkos_impl_hip_shared_memory<char>() +
                  m_team_begin,
              m_shmem_begin, m_shmem_size,
              reinterpret_cast<void*>(
                  reinterpret_cast<char*>(m_scratch_ptr[1]) +
                  static_cast<ptrdiff_t>(threadid / (blockDim.x * blockDim.y)) *
                      m_scratch_size[1]),
              m_scratch_size[1], league_rank, m_league_size),
          value);
    }

    pointer_type const result =
        m_result_ptr_device_accessible
            ? m_result_ptr
            : reinterpret_cast<pointer_type>(m_scratch_space);

    value_type init;
    value_init::init(reducer_conditional::select(m_functor, m_reducer), &init);
    if (Impl::hip_inter_block_shuffle_reduction<FunctorType, value_join,
                                                work_tag>(
            value, init,
            value_join(reducer_conditional::select(m_functor, m_reducer)),
            m_scratch_space, result, m_scratch_flags, blockDim.y)) {
      unsigned int const id = threadIdx.y * blockDim.x + threadIdx.x;
      if (id == 0) {
        Kokkos::Impl::FunctorFinal<reducer_type_fwd, work_tag_fwd>::final(
            reducer_conditional::select(m_functor, m_reducer),
            reinterpret_cast<void*>(&value));
        *result = value;
      }
    }
  }

  inline void execute() {
    const int nwork = m_league_size * m_team_size;
    if (nwork) {
      const int block_count =
          UseShflReduction
              ? std::min(
                    m_league_size,
                    size_type(1024 *
                              Kokkos::Experimental::Impl::HIPTraits::WarpSize))
              : std::min(static_cast<int>(m_league_size), m_team_size);

      m_scratch_space = Kokkos::Experimental::Impl::hip_internal_scratch_space(
          value_traits::value_size(
              reducer_conditional::select(m_functor, m_reducer)) *
          block_count);
      m_scratch_flags = Kokkos::Experimental::Impl::hip_internal_scratch_flags(
          sizeof(size_type));

      dim3 block(m_vector_size, m_team_size, 1);
      dim3 grid(block_count, 1, 1);
      const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size;

      Kokkos::Experimental::Impl::HIPParallelLaunch<ParallelReduce,
                                                    launch_bounds>(
          *this, grid, block, shmem_size_total,
          m_policy.space().impl_internal_space_instance(),
          true);  // copy to device and execute

      if (!m_result_ptr_device_accessible) {
        m_policy.space().impl_internal_space_instance()->fence();

        if (m_result_ptr) {
          const int size = value_traits::value_size(
              reducer_conditional::select(m_functor, m_reducer));
          DeepCopy<HostSpace, Kokkos::Experimental::HIPSpace>(
              m_result_ptr, m_scratch_space, size);
        }
      }
    } else {
      if (m_result_ptr) {
        value_init::init(reducer_conditional::select(m_functor, m_reducer),
                         m_result_ptr);
      }
    }
  }

  template <class ViewType>
  ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
                 ViewType const& arg_result,
                 typename std::enable_if<Kokkos::is_view<ViewType>::value,
                                         void*>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::HIPSpace,
                              typename ViewType::memory_space>::accessible),
        m_scratch_space(0),
        m_scratch_flags(0),
        m_team_begin(0),
        m_shmem_begin(0),
        m_shmem_size(0),
        m_scratch_ptr{nullptr, nullptr},
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.vector_length()) {
    hipFuncAttributes attr = Kokkos::Experimental::Impl::HIPParallelLaunch<
        ParallelReduce, launch_bounds>::get_hip_func_attributes();
    m_team_size =
        m_team_size >= 0
            ? m_team_size
            : Kokkos::Experimental::Impl::hip_get_opt_block_size<FunctorType,
                                                                 launch_bounds>(
                  m_policy.space().impl_internal_space_instance(), attr,
                  m_functor, m_vector_size, m_policy.team_scratch_size(0),
                  m_policy.thread_scratch_size(0)) /
                  m_vector_size;

    // Return Init value if the number of worksets is zero
    if (m_league_size * m_team_size == 0) {
      value_init::init(reducer_conditional::select(m_functor, m_reducer),
                       arg_result.data());
      return;
    }

    m_team_begin =
        UseShflReduction
            ? 0
            : hip_single_inter_block_reduce_scan_shmem<false, FunctorType,
                                                       work_tag>(arg_functor,
                                                                 m_team_size);
    m_shmem_begin = sizeof(double) * (m_team_size + 2);
    m_shmem_size =
        m_policy.scratch_size(0, m_team_size) +
        FunctorTeamShmemSize<FunctorType>::value(arg_functor, m_team_size);
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = m_policy.scratch_size(1, m_team_size);
    m_scratch_ptr[1] =
        m_team_size <= 0 ? nullptr
                         : Kokkos::Experimental::Impl::hip_resize_scratch_space(
                               static_cast<std::int64_t>(m_scratch_size[1]) *
                               (static_cast<std::int64_t>(
                                   Kokkos::Experimental::HIP::concurrency() /
                                   (m_team_size * m_vector_size))));

    // The global parallel_reduce does not support vector_length other than 1 at
    // the moment
    if ((arg_policy.vector_length() > 1) && !UseShflReduction)
      Impl::throw_runtime_exception(
          "Kokkos::parallel_reduce with a TeamPolicy using a vector length of "
          "greater than 1 is not currently supported for HIP for dynamic "
          "sized reduction types.");

    if ((m_team_size < Kokkos::Experimental::Impl::HIPTraits::WarpSize) &&
        !UseShflReduction)
      Impl::throw_runtime_exception(
          "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller "
          "than 64 is not currently supported with HIP for dynamic sized "
          "reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend
    // upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size;

    if (!Kokkos::Impl::is_integral_power_of_two(m_team_size) &&
        !UseShflReduction) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelReduce< HIP > bad team size"));
    }

    if (m_policy.space().impl_internal_space_instance()->m_maxShmemPerBlock <
        shmem_size_total) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelReduce< HIP > requested too much "
                      "L0 scratch memory"));
    }

    if (static_cast<int>(m_team_size) >
        arg_policy.team_size_max(m_functor, m_reducer, ParallelReduceTag())) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelReduce< HIP > requested too "
                      "large team size."));
    }
  }

  ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
                 ReducerType const& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::HIPSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_scratch_space(0),
        m_scratch_flags(0),
        m_team_begin(0),
        m_shmem_begin(0),
        m_shmem_size(0),
        m_scratch_ptr{nullptr, nullptr},
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.vector_length()) {
    hipFuncAttributes attr = Kokkos::Experimental::Impl::HIPParallelLaunch<
        ParallelReduce, launch_bounds>::get_hip_func_attributes();
    m_team_size =
        m_team_size >= 0
            ? m_team_size
            : Kokkos::Experimental::Impl::hip_get_opt_block_size<FunctorType,
                                                                 launch_bounds>(
                  m_policy.space().impl_internal_space_instance(), attr,
                  m_functor, m_vector_size, m_policy.team_scratch_size(0),
                  m_policy.thread_scratch_size(0)) /
                  m_vector_size;

    // Return Init value if the number of worksets is zero
    if (arg_policy.league_size() == 0) {
      value_init::init(reducer_conditional::select(m_functor, m_reducer),
                       m_result_ptr);
      return;
    }

    m_team_begin =
        UseShflReduction
            ? 0
            : hip_single_inter_block_reduce_scan_shmem<false, FunctorType,
                                                       work_tag>(arg_functor,
                                                                 m_team_size);
    m_shmem_begin = sizeof(double) * (m_team_size + 2);
    m_shmem_size =
        m_policy.scratch_size(0, m_team_size) +
        FunctorTeamShmemSize<FunctorType>::value(arg_functor, m_team_size);
    m_scratch_size[0] = m_shmem_size;
    m_scratch_size[1] = m_policy.scratch_size(1, m_team_size);
    m_scratch_ptr[1] =
        m_team_size <= 0 ? nullptr
                         : Kokkos::Experimental::Impl::hip_resize_scratch_space(
                               static_cast<ptrdiff_t>(m_scratch_size[1]) *
                               static_cast<ptrdiff_t>(
                                   Kokkos::Experimental::HIP::concurrency() /
                                   (m_team_size * m_vector_size)));

    // The global parallel_reduce does not support vector_length other than 1 at
    // the moment
    if ((arg_policy.vector_length() > 1) && !UseShflReduction)
      Impl::throw_runtime_exception(
          "Kokkos::parallel_reduce with a TeamPolicy using a vector length of "
          "greater than 1 is not currently supported for HIP for dynamic "
          "sized reduction types.");

    if ((m_team_size < Kokkos::Experimental::Impl::HIPTraits::WarpSize) &&
        !UseShflReduction)
      Impl::throw_runtime_exception(
          "Kokkos::parallel_reduce with a TeamPolicy using a team_size smaller "
          "than 64 is not currently supported with HIP for dynamic sized "
          "reduction types.");

    // Functor's reduce memory, team scan memory, and team shared memory depend
    // upon team size.

    const int shmem_size_total = m_team_begin + m_shmem_begin + m_shmem_size;

    if ((!Kokkos::Impl::is_integral_power_of_two(m_team_size) &&
         !UseShflReduction) ||
        m_policy.space().impl_internal_space_instance()->m_maxShmemPerBlock <
            shmem_size_total) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelReduce< HIP > bad team size"));
    }
    if (static_cast<int>(m_team_size) >
        arg_policy.team_size_max(m_functor, m_reducer, ParallelReduceTag())) {
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelReduce< HIP > requested too "
                      "large team size."));
    }
  }
};
}  // namespace Impl
}  // namespace Kokkos

#endif

#endif
