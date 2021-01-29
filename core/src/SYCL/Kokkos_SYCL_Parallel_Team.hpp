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

#ifndef KOKKOS_SYCL_PARALLEL_TEAM_HPP
#define KOKKOS_SYCL_PARALLEL_TEAM_HPP

#include <Kokkos_Parallel.hpp>

#include <SYCL/Kokkos_SYCL_Team.hpp>

namespace Kokkos {
namespace Impl {
template <typename... Properties>
class TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>
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
  bool m_tune_team_size;
  bool m_tune_vector_length;

 public:
  using execution_space = Kokkos::Experimental::SYCL;

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
    m_tune_team_size         = p.m_tune_team_size;
    m_tune_vector_length     = p.m_tune_vector_length;
  }

  template <typename FunctorType>
  int team_size_max(FunctorType const& f, ParallelForTag const&) const {
    using closure_type =
        Impl::ParallelFor<FunctorType, TeamPolicy<Properties...>>;
    return internal_team_size_max<closure_type>(f);
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
  int team_size_recommended(FunctorType const& /*f*/,
                            ParallelForTag const&) const {
    // FIXME_SYCL optimize
    return m_space.impl_internal_space_instance()->m_maxThreadsPerSM;
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
  inline bool impl_auto_vector_length() const { return m_tune_vector_length; }
  inline bool impl_auto_team_size() const { return m_tune_team_size; }
  static int vector_length_max() {
    // FIXME_SYCL provide a reasonable value
    return 8;
  }

  static int verify_requested_vector_length(int requested_vector_length) {
    int test_vector_length =
        std::min(requested_vector_length, vector_length_max());

    // Allow only power-of-two vector_length
    if (!(is_integral_power_of_two(test_vector_length))) {
      int test_pow2 = 1;
      for (int i = 0; i < 5; i++) {
        test_pow2 = test_pow2 << 1;
        if (test_pow2 > test_vector_length) {
          break;
        }
      }
      test_vector_length = test_pow2 >> 1;
    }

    return test_vector_length;
  }

  static int scratch_size_max(int level) {
    return level == 0 ? 1024 * 32
                      :           // FIXME_SYCL arbitrarily setting this to 32kB
               20 * 1024 * 1024;  // FIXME_SYCL arbitrarily setting this to 20MB
  }
  inline void impl_set_vector_length(size_t size) { m_vector_length = size; }
  inline void impl_set_team_size(size_t size) { m_team_size = size; }
  int impl_vector_length() const { return m_vector_length; }
  KOKKOS_DEPRECATED int vector_length() const { return impl_vector_length(); }

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
        m_chunk_size(0),
        m_tune_team_size(false),
        m_tune_vector_length(false) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     int team_size_request, int vector_length_request = 1)
      : m_space(space_),
        m_league_size(league_size_),
        m_team_size(team_size_request),
        m_vector_length(
            (vector_length_request > 0)
                ? verify_requested_vector_length(vector_length_request)
                : (verify_requested_vector_length(1))),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0),
        m_tune_team_size(bool(team_size_request <= 0)),
        m_tune_vector_length(bool(vector_length_request <= 0)) {
    // FIXME_SYCL check paramters
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : TeamPolicyInternal(space_, league_size_, -1, vector_length_request) {}
  // FLAG
  /** \brief  Specify league size and team size, request vector length*/
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space_, league_size_, team_size_request, -1)

  {}

  /** \brief  Specify league size, request team size and vector length*/
  TeamPolicyInternal(const execution_space space_, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(space_, league_size_, -1, -1)

  {}

  TeamPolicyInternal(int league_size_, int team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                           team_size_request, vector_length_request) {}

  TeamPolicyInternal(int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(), league_size_, -1,
                           vector_length_request) {}

  /** \brief  Specify league size and team size, request vector length*/
  TeamPolicyInternal(int league_size_, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(typename traits::execution_space(), league_size_,
                           team_size_request, -1)

  {}

  /** \brief  Specify league size, request team size and vector length*/
  TeamPolicyInternal(int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(typename traits::execution_space(), league_size_, -1,
                           -1) {}

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

  using member_type = Kokkos::Impl::SYCLTeamMember;

 protected:
  template <class ClosureType, class FunctorType, class BlockSizeCallable>
  int internal_team_size_common(
      const FunctorType& /*f*/,
      BlockSizeCallable&& /*block_size_callable*/) const {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
    return 0;
  }

  template <class ClosureType, class FunctorType>
  int internal_team_size_max(const FunctorType& /*f*/) const {
    return m_space.impl_internal_space_instance()->m_maxThreadsPerSM;
  }

  template <class ClosureType, class FunctorType>
  int internal_team_size_recommended(const FunctorType& /*f*/) const {
    // FIXME_SYCL improve
    return m_space.impl_internal_space_instance()->m_maxThreadsPerSM;
  }
};

template <typename FunctorType, typename... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::SYCL> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>;
  using functor_type = FunctorType;
  using size_type    = ::Kokkos::Experimental::SYCL::size_type;

 private:
  using member_type   = typename Policy::member_type;
  using work_tag      = typename Policy::work_tag;
  using launch_bounds = typename Policy::launch_bounds;

  FunctorType const m_functor;
  Policy const m_policy;
  size_type const m_league_size;
  int m_team_size;
  size_type const m_vector_size;
  int m_shmem_begin;
  int m_shmem_size;
  void* m_scratch_ptr[2];
  int m_scratch_size[2];

  template <typename Functor>
  void sycl_direct_launch(const Policy& policy, const Functor& functor) const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    sycl::queue& q = *instance.m_queue;

    q.submit([&](sycl::handler& cgh) {
      const auto shmem_begin  = m_shmem_begin;
      const auto shmem_size   = m_shmem_size;
      const auto scratch_size = m_scratch_size[1];
      auto team_lambda        = [=](sycl::nd_item<1> item) {
        // FIXME_SYCL Add scratch memory
        const member_type team_member(nullptr, shmem_begin, shmem_size, nullptr,
                                      scratch_size, item);

        if constexpr (std::is_same<work_tag, void>::value)
          functor(team_member);
        else
          functor(work_tag(), team_member);
      };
      cgh.parallel_for(
          sycl::nd_range<1>(m_league_size * m_team_size, m_team_size),
          team_lambda);
    });
    space.fence();
  }

  // Indirectly launch a functor by explicitly creating it in USM shared memory
  void sycl_indirect_launch() const {
    // Convenience references
    const Kokkos::Experimental::SYCL& space = m_policy.space();
    Kokkos::Experimental::Impl::SYCLInternal& instance =
        *space.impl_internal_space_instance();
    using IndirectKernelMem =
        Kokkos::Experimental::Impl::SYCLInternal::IndirectKernelMem;
    IndirectKernelMem& indirectKernelMem = instance.m_indirectKernelMem;

    // Copy the functor into USM Shared Memory
    using KernelFunctorPtr =
        std::unique_ptr<FunctorType, IndirectKernelMem::Deleter>;
    KernelFunctorPtr kernelFunctorPtr = indirectKernelMem.copy_from(m_functor);

    // Use reference_wrapper (because it is both trivially copyable and
    // invocable) and launch it
    sycl_direct_launch(m_policy, std::reference_wrapper(*kernelFunctorPtr));
  }

 public:
  inline void execute() const {
    if (m_league_size == 0) return;

    // if the functor is trivially copyable, we can launch it directly;
    // otherwise, we will launch it indirectly via explicitly creating
    // it in USM shared memory.
    if constexpr (std::is_trivially_copyable_v<decltype(m_functor)>)
      sycl_direct_launch(m_policy, m_functor);
    else
      sycl_indirect_launch();
  }

  ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()) {
    // FIXME_SYCL check_parameters
    // FIXME_SYCL optimize
    if (m_team_size < 0) m_team_size = 32;
  }
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::SYCL> {
 public:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::SYCL, Properties...>;

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
  using size_type    = Kokkos::Experimental::SYCL::size_type;

 private:
  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;
  const bool m_result_ptr_device_accessible;
  const bool m_result_ptr_host_accessible;
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

 public:
  inline void execute() {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
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
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename ViewType::memory_space>::accessible),
        m_result_ptr_host_accessible(
            MemorySpaceAccess<Kokkos::HostSpace,
                              typename ViewType::memory_space>::accessible),
        m_scratch_space(nullptr),
        m_scratch_flags(nullptr),
        m_team_begin(0),
        m_shmem_begin(0),
        m_shmem_size(0),
        m_scratch_ptr{nullptr, nullptr},
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()) {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }

  ParallelReduce(FunctorType const& arg_functor, Policy const& arg_policy,
                 ReducerType const& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_result_ptr_device_accessible(
            MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_result_ptr_host_accessible(
            MemorySpaceAccess<Kokkos::HostSpace,
                              typename ReducerType::result_view_type::
                                  memory_space>::accessible),
        m_scratch_space(nullptr),
        m_scratch_flags(nullptr),
        m_team_begin(0),
        m_shmem_begin(0),
        m_shmem_size(0),
        m_scratch_ptr{nullptr, nullptr},
        m_league_size(arg_policy.league_size()),
        m_team_size(arg_policy.team_size()),
        m_vector_size(arg_policy.impl_vector_length()) {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }
};
}  // namespace Impl
}  // namespace Kokkos

#endif
