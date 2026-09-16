// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SYCL_TEAM_POLICY_HPP
#define KOKKOS_SYCL_TEAM_POLICY_HPP

#include <SYCL/Kokkos_SYCL_Team.hpp>
#include <Kokkos_BitManipulation.hpp>
#include <SYCL/Kokkos_SYCL_Instance.hpp>

#include <vector>

template <typename... Properties>
class Kokkos::Impl::TeamPolicyInternal<Kokkos::SYCL, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 private:
  typename traits::execution_space m_space;
  int m_league_size;
  int m_team_size;
  int m_vector_length;
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];
  int m_chunk_size;
  bool m_tune_team_size;
  bool m_tune_vector_length;

 public:
  using execution_space = Kokkos::SYCL;

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
    return internal_team_size_max_for(f);
  }

  template <class FunctorType>
  inline int team_size_max(const FunctorType& f,
                           const ParallelReduceTag& tag) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, FunctorType, void>;
    typename functor_analysis_type::Reducer reducer(f);
    return team_size_max_internal(f, reducer, tag);
  }

  template <class FunctorType, class ReducerType>
  inline int team_size_max(const FunctorType& f, const ReducerType& r,
                           const ParallelReduceTag& tag) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, ReducerType, void>;
    typename functor_analysis_type::Reducer reducer(r);
    return team_size_max_internal(f, reducer, tag);
  }

  template <class FunctorType, class ReducerType>
  int team_size_max_internal(FunctorType const& f, ReducerType const& r,
                             ParallelReduceTag const&) const {
    return internal_team_size_max_reduce<typename ReducerType::value_type>(
        CombinedFunctorReducer<FunctorType, ReducerType>(f, r));
  }

  template <typename FunctorType>
  int team_size_recommended(FunctorType const& f, ParallelForTag const&) const {
    return internal_team_size_recommended_for(f);
  }

  template <typename FunctorType>
  inline int team_size_recommended(FunctorType const& f,
                                   ParallelReduceTag const& tag) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, FunctorType, void>;
    typename functor_analysis_type::Reducer reducer(f);
    return team_size_recommended_internal(f, reducer, tag);
  }

  template <class FunctorType, class ReducerType>
  int team_size_recommended(FunctorType const& f, ReducerType const& r,
                            ParallelReduceTag const& tag) const {
    using functor_analysis_type =
        Impl::FunctorAnalysis<Impl::FunctorPatternInterface::REDUCE,
                              TeamPolicyInternal, ReducerType, void>;
    typename functor_analysis_type::Reducer reducer(r);
    return team_size_recommended_internal(f, reducer, tag);
  }

  template <class FunctorType, class ReducerType>
  int team_size_recommended_internal(FunctorType const& f, ReducerType const& r,
                                     ParallelReduceTag const&) const {
    return internal_team_size_recommended_reduce<
        typename ReducerType::value_type>(
        CombinedFunctorReducer<FunctorType, ReducerType>(f, r));
  }

  inline bool impl_auto_vector_length() const { return m_tune_vector_length; }
  inline bool impl_auto_team_size() const { return m_tune_team_size; }
  // FIXME_SYCL This is correct in most cases, but not necessarily in case a
  // custom sycl::queue is used to initialize the execution space.
  static int vector_length_max() {
    std::vector<size_t> sub_group_sizes =
        execution_space{}
            .sycl_queue()
            .get_device()
            .template get_info<sycl::info::device::sub_group_sizes>();
    return *std::max_element(sub_group_sizes.begin(), sub_group_sizes.end());
  }

 private:
  static int determine_vector_length(int requested) {
    // restrict requested between 1 and max
    unsigned vector_length = std::clamp(requested, 1, vector_length_max());
    // return the largest integral power of 2 not greater than requested
    return Kokkos::bit_floor(vector_length);
  }

 public:
  static int scratch_size_max(int level) {
    const auto& sycl_instance = *SYCL{}.impl_internal_space_instance();
    // FIXME_SYCL Avoid requesting too many registers on NVIDIA GPUs.
#if defined(KOKKOS_IMPL_ARCH_NVIDIA_GPU)
    const size_t max_possible_team_size = 256;
#else
    const size_t max_possible_team_size = sycl_instance.m_maxWorkgroupSize;
#endif
    const size_t max_reserved_shared_mem_per_team =
        (max_possible_team_size + 2) * sizeof(double);
    // arbitrarily setting level 1 scratch limit to 20MB
    constexpr size_t max_l1_scratch_size =
        static_cast<size_t>(80) * 1024 * 1024;

    size_t max_shmem = sycl_instance.m_maxShmemPerBlock;
    return (level == 0 ? max_shmem - max_reserved_shared_mem_per_team
                       : max_l1_scratch_size);
  }
  inline void impl_set_vector_length(size_t size) { m_vector_length = size; }
  inline void impl_set_team_size(size_t size) { m_team_size = size; }
  int impl_vector_length() const { return m_vector_length; }

  int team_size() const { return m_team_size; }

  int league_size() const { return m_league_size; }

  size_t scratch_size(int level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = m_team_size;
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  size_t team_scratch_size(int level) const {
    return m_team_scratch_size[level];
  }

  size_t thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  typename traits::execution_space space() const { return m_space; }

  TeamPolicyInternal()
      : m_space(),
        m_league_size(0),
        m_team_size(-1),
        m_vector_length(0),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(vector_length_max()),
        m_tune_team_size(false),
        m_tune_vector_length(false) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(execution_space space, int league_size_,
                     int team_size_request, int vector_length_request = 1)
      : m_space(std::move(space)),
        m_league_size(league_size_),
        m_team_size(team_size_request),
        m_vector_length(determine_vector_length(vector_length_request)),
        m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(vector_length_max()),
        m_tune_team_size(bool(team_size_request <= 0)),
        m_tune_vector_length(bool(vector_length_request <= 0)) {
    // FIXME_SYCL Check that league size is permissible,
    // https://github.com/intel/llvm/pull/4064

    // Make sure total block size is permissible
    if (m_team_size * m_vector_length >
        static_cast<int>(
            m_space.impl_internal_space_instance()->m_maxWorkgroupSize)) {
      std::stringstream error;
      error << "Kokkos::TeamPolicy<SYCL>: Requested too large team size. "
               "Requested: "
            << m_team_size << ", Maximum: "
            << m_space.impl_internal_space_instance()->m_maxWorkgroupSize /
                   m_vector_length;
      Kokkos::Impl::throw_runtime_exception(error.str().c_str());
    }
  }

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(execution_space space, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     int vector_length_request = 1)
      : TeamPolicyInternal(std::move(space), league_size_, -1,
                           vector_length_request) {}
  // FLAG
  /** \brief  Specify league size and team size, request vector length*/
  TeamPolicyInternal(execution_space space, int league_size_,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(std::move(space), league_size_, team_size_request,
                           -1) {}

  /** \brief  Specify league size, request team size and vector length*/
  TeamPolicyInternal(execution_space space, int league_size_,
                     const Kokkos::AUTO_t& /* team_size_request */,
                     const Kokkos::AUTO_t& /* vector_length_request */

                     )
      : TeamPolicyInternal(std::move(space), league_size_, -1, -1)

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

  TeamPolicyInternal(const PolicyUpdate, const TeamPolicyInternal& other,
                     typename traits::execution_space space)
      : TeamPolicyInternal(other) {
    this->m_space = std::move(space);
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

  using member_type = Kokkos::Impl::SYCLTeamMember;

 protected:
  template <class FunctorType>
  int internal_team_size_max_for(const FunctorType& f) const {
    // nested_reducer_memsize = (sizeof(double) * (m_team_size + 2)
    // custom: m_team_scratch_size[0] + m_thread_scratch_size[0] * m_team_size
    // total:
    // 2*sizeof(double)+m_team_scratch_size[0]
    // + m_team_size(sizeof(double)+m_thread_scratch_size[0])
    const int max_threads_for_memory =
        (space().impl_internal_space_instance()->m_maxShmemPerBlock -
         2 * sizeof(double) - m_team_scratch_size[0]) /
        (sizeof(double) + m_thread_scratch_size[0]);

    auto& instance          = *m_space.impl_internal_space_instance();
    auto& indirectKernelMem = instance.get_indirect_kernel_mem();
    auto functor_wrapper =
        Impl::make_sycl_function_wrapper(f, indirectKernelMem);
    sycl::queue& q = m_space.sycl_queue();

    int max_threads_kernel = 0;
    auto event             = q.submit([&](sycl::handler& cgh) {
      // minimal local accessor to create the same lambda type used at
      // launch-time
      sycl::local_accessor<char, 1> team_scratch_memory_L0(sycl::range<1>(1),
                                                                       cgh);

      const auto shmem_begin       = 0u;
      const size_t scratch_size[2] = {0, 0};

      // Create a dummy kernel mirroring the TeamPolicy parallel_for
      // implementation for introspection. The kernel call has an empty range
      // and hence the kernel isn't actually executed but the kernel still needs
      // to be submitted for it to be introspected.
      using ParallelForImpl =
          Impl::ParallelFor<FunctorType, TeamPolicy<Properties...>,
                            Kokkos::SYCL>;
      auto lambda = ParallelForImpl::create_team_for_lambda(
          functor_wrapper, team_scratch_memory_L0, scratch_size, shmem_begin,
          /*global_scratch_ptr*/ nullptr);

      sycl::kernel_id functor_kernel_id =
          sycl::get_kernel_id<decltype(lambda)>();
      auto kernel_bundle =
          sycl::get_kernel_bundle<sycl::bundle_state::executable>(
              q.get_context(), std::vector{functor_kernel_id});
      auto kernel = kernel_bundle.get_kernel(functor_kernel_id);
      max_threads_kernel =
          kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
              q.get_device());

      cgh.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(0, 0), sycl::range<2>(1, 1)),
          lambda);
    });
    functor_wrapper.register_event(event);

    return std::min({max_threads_kernel, max_threads_for_memory}) /
           impl_vector_length();
  }

  template <class ValueType, class CombinedFunctorReducerType>
  int internal_team_size_max_reduce(const CombinedFunctorReducerType& f) const {
    using Analysis =
        FunctorAnalysis<FunctorPatternInterface::REDUCE, TeamPolicyInternal,
                        CombinedFunctorReducerType, ValueType>;
    using value_type      = typename Analysis::value_type;
    const int value_count = Analysis::value_count(f);

    // nested_reducer_memsize = (sizeof(double) * (m_team_size + 2)
    // reducer_memsize = sizeof(value_type) * m_team_size * value_count
    // custom: m_team_scratch_size[0] + m_thread_scratch_size[0] * m_team_size
    // total:
    // 2*sizeof(double)+m_team_scratch_size[0]
    // + m_team_size(sizeof(double)+sizeof(value_type)*value_count
    //               +m_thread_scratch_size[0])
    const int max_threads_for_memory =
        (space().impl_internal_space_instance()->m_maxShmemPerBlock -
         2 * sizeof(double) - m_team_scratch_size[0]) /
        (sizeof(double) + sizeof(value_type) * value_count +
         m_thread_scratch_size[0]);

    auto& instance          = *m_space.impl_internal_space_instance();
    auto& indirectKernelMem = instance.get_indirect_kernel_mem();
    auto functor_wrapper =
        Impl::make_sycl_function_wrapper(f, indirectKernelMem);
    sycl::queue& q = m_space.sycl_queue();

    int max_threads_kernel = 0;
    auto event             = q.submit([&](sycl::handler& cgh) {
      // minimal local accessor to form the expected lambda type
      sycl::local_accessor<char, 1> team_scratch_memory_L0(sycl::range<1>(1),
                                                                       cgh);

      const auto shmem_begin       = 0u;
      const size_t scratch_size[2] = {0, 0};

      // Create a dummy kernel using the actual TeamPolicy parallel_reduce
      // implementation for introspection. The kernel call has an empty range
      // and hence the kernel isn't actually executed but the kernel still needs
      // to be submitted for it to be introspected.
      auto lambda =
          Impl::ParallelReduce<CombinedFunctorReducerType,
                               TeamPolicy<Properties...>, Kokkos::SYCL>::
              create_team_reduction_lambda(
                  /* local_mem */ {1, cgh},
                  /* results_ptr */ nullptr,
                  /* device_accessible_result_ptr */ nullptr, functor_wrapper,
                  value_count,
                  /*league_size*/ 0, team_scratch_memory_L0, scratch_size,
                  shmem_begin,
                  /*global_scratch_ptr*/ nullptr,
                  /*num_teams_done*/ {1, cgh},
                  /*scratch_flags*/ nullptr);

      sycl::kernel_id functor_kernel_id =
          sycl::get_kernel_id<decltype(lambda)>();
      auto kernel_bundle =
          sycl::get_kernel_bundle<sycl::bundle_state::executable>(
              q.get_context(), std::vector{functor_kernel_id});
      auto kernel = kernel_bundle.get_kernel(functor_kernel_id);
      max_threads_kernel =
          kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
              q.get_device());

      cgh.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(0, 0), sycl::range<2>(1, 1)),
          lambda);
    });
    functor_wrapper.register_event(event);

    return std::min({max_threads_kernel, max_threads_for_memory}) /
           impl_vector_length();
  }

  template <class FunctorType>
  int internal_team_size_recommended_for(const FunctorType& f) const {
    // FIXME_SYCL improve
    return Kokkos::Experimental::bit_floor_builtin<unsigned>(
        internal_team_size_max_for(f));
  }

  template <class ValueType, class FunctorType>
  int internal_team_size_recommended_reduce(const FunctorType& f) const {
    // FIXME_SYCL improve
    return Kokkos::Experimental::bit_floor_builtin<unsigned>(
        internal_team_size_max_reduce<ValueType>(f));
  }
};

#endif
