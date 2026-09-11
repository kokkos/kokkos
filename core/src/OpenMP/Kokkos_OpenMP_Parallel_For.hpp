// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_OPENMP_PARALLEL_FOR_HPP
#define KOKKOS_OPENMP_PARALLEL_FOR_HPP

#include <omp.h>
#include <OpenMP/Kokkos_OpenMP_Instance.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <MDRange/Kokkos_FlatIterate.hpp>
#include <sstream>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#if defined(KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION) && \
    defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#undef KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#define KOKKOS_PRAGMA_IVDEP_IF_ENABLED _Pragma("ivdep")
#endif

#ifndef KOKKOS_COMPILER_NVHPC
#define KOKKOS_OPENMP_OPTIONAL_CHUNK_SIZE , m_policy.chunk_size()
#else
#define KOKKOS_OPENMP_OPTIONAL_CHUNK_SIZE
#endif

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::OpenMP> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;

  inline static void exec_range(const FunctorType& functor, const Member ibeg,
                                const Member iend) {
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = ibeg; iwork < iend; ++iwork) {
      exec_work(functor, iwork);
    }
  }

  inline static void exec_work(const FunctorType& functor, const Member iwork) {
    if constexpr (std::is_void_v<WorkTag>) {
      functor(iwork);
    } else {
      functor(WorkTag{}, iwork);
    }
  }

  template <class Policy>
  std::enable_if_t<
      std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
    // prevent bug in NVHPC 21.9/CUDA 11.4 (entering zero iterations loop)
    if (m_policy.begin() >= m_policy.end()) return;
#pragma omp parallel for schedule(dynamic KOKKOS_OPENMP_OPTIONAL_CHUNK_SIZE) \
    num_threads(m_instance->thread_pool_size())
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = m_policy.begin(); iwork < m_policy.end(); ++iwork) {
      exec_work(m_functor, iwork);
    }
  }

  template <class Policy>
  std::enable_if_t<
      !std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
// Specifying an chunksize with GCC compiler leads to performance regression
// with static schedule.
#ifdef KOKKOS_COMPILER_GNU
#pragma omp parallel for schedule(static) \
    num_threads(m_instance->thread_pool_size())
#else
#pragma omp parallel for schedule(static KOKKOS_OPENMP_OPTIONAL_CHUNK_SIZE) \
    num_threads(m_instance->thread_pool_size())
#endif
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (auto iwork = m_policy.begin(); iwork < m_policy.end(); ++iwork) {
      exec_work(m_functor, iwork);
    }
  }

 public:
  inline void execute() const {
    // Serialize kernels on the same execution space instance
    std::lock_guard<std::mutex> lock(m_instance->m_instance_mutex);
    if (execute_in_serial(m_policy.space())) {
      exec_range(m_functor, m_policy.begin(), m_policy.end());
      return;
    }

#ifndef KOKKOS_INTERNAL_DISABLE_NATIVE_OPENMP
    execute_parallel<Policy>();
#else
    constexpr bool is_dynamic =
        std::is_same<typename Policy::schedule_type::type,
                     Kokkos::Dynamic>::value;
#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      data.set_work_partition(m_policy.end() - m_policy.begin(),
                              m_policy.chunk_size());

      if (is_dynamic) {
        // Make sure work partition is set before stealing
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      std::pair<int64_t, int64_t> range(0, 0);

      do {
        range = is_dynamic ? data.get_work_stealing_chunk()
                           : data.get_work_partition();

        exec_range(m_functor, range.first + m_policy.begin(),
                   range.second + m_policy.begin());

      } while (is_dynamic && 0 <= range.first);
    }
#endif
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_instance(nullptr),
        m_functor(arg_functor),
        m_policy(std::move(arg_policy)) {
    m_instance = m_policy.space().impl_internal_space_instance();
  }
};

// MDRangePolicy impl
template <class MDRP, class Functor, class Tag>
  requires std::same_as<typename MDRP::execution_space, Kokkos::OpenMP>
class FlatIterate<MDRP, Functor, Tag> {
 public:
  using range_policy         = typename MDRP::impl_range_policy;
  using index_type           = typename range_policy::index_type;
  using iteration_pattern    = typename MDRP::iteration_pattern;
  using point_type           = typename MDRP::point_type;
  static constexpr auto rank = MDRP::rank;

  FlatIterate(const MDRP& mdrp, const Functor& fun)
      : m_md_range_policy(mdrp), m_functor(fun) {}

  void exec(Static) const {
    if constexpr (iteration_pattern::inner_direction == Iterate::Right) {
      exec_rank(std::make_integer_sequence<int, rank>{}, m_tag, Static{});
    } else {
      exec_rank(make_reverse_integer_sequence<int, rank>{}, m_tag, Static{});
    }
  }

  const MDRP& policy() const noexcept { return m_md_range_policy; }

 private:
  struct NoTag {};

  void apply_to_functor(const point_type& point, NoTag) const {
    apply(m_functor, point);
  }

  template<typename AnyTag>
  void apply_to_functor(const point_type& point, AnyTag tag) const {
    apply(
        [tag, this]<typename... Args>(Args&&... args) {
          m_functor(tag, std::forward<Args>(args)...);
        },
        point);
  }


  template <int R1, class TagOrNoTag>
  void exec_rank(std::integer_sequence< int, R1>, TagOrNoTag tag, Static) const {
    point_type p;
    using array_index_type = typename MDRP::array_index_type;
    #pragma omp parallel for schedule(static, 1) firstprivate(p)
    for ( array_index_type i = m_md_range_policy.m_lower[R1];
          i < m_md_range_policy.m_upper[R1]; ++i) {
      p[R1] = i;
      apply_to_functor(p, tag);
    }
  }


  template <int R1, int R2, class TagOrNoTag>
  void exec_rank(std::integer_sequence< int, R1, R2>, TagOrNoTag tag, Static) const {
    point_type p;
    using array_index_type = typename MDRP::array_index_type;
    #pragma omp parallel for schedule(static, 1) collapse(2) firstprivate(p)
    for (array_index_type i = m_md_range_policy.m_lower[R1];
         i < m_md_range_policy.m_upper[R1]; ++i) {
      for (array_index_type j = m_md_range_policy.m_lower[R2];
           j < m_md_range_policy.m_upper[R2]; ++j) {
        p[R1] = i;
        p[R2] = j;
        apply_to_functor(p, tag);
      }
    }
  }

  template <int R1, int R2, int R3, class TagOrNoTag, int... Rs>
  void exec_rank(std::integer_sequence< int, R1, R2, R3, Rs...>, TagOrNoTag tag, Static) const {
    point_type p;
    using array_index_type = typename MDRP::array_index_type;
    #pragma omp parallel for schedule(static, 1) collapse(3) firstprivate(p)
    for (array_index_type i = m_md_range_policy.m_lower[R1];
         i < m_md_range_policy.m_upper[R1]; ++i) {
      for (array_index_type j = m_md_range_policy.m_lower[R2];
           j < m_md_range_policy.m_upper[R2]; ++j) {
        for (array_index_type k = m_md_range_policy.m_lower[R3];
             k < m_md_range_policy.m_upper[R3]; ++k) {
          p[R1] = i;
          p[R2] = j;
          p[R3] = k;
          exec_rank_nested(std::integer_sequence< int, Rs... >{}, p, tag);
        }
      }
    }
  }

  template <class TagOrNoTag>
  void exec_rank_nested(std::integer_sequence<int>, point_type& point, TagOrNoTag tag) const {
    apply_to_functor(point, tag);
  }

  template <int Rank, int... RemRanks, class TagOrNoTag>
  void exec_rank_nested(std::integer_sequence<int, Rank, RemRanks...>,
                 point_type& point, TagOrNoTag tag) const {
    using array_index_type = typename MDRP::array_index_type;
    for (array_index_type i = m_md_range_policy.m_lower[Rank];
         i < m_md_range_policy.m_upper[Rank]; ++i) {
      point[Rank] = i;
      exec_rank_nested(std::integer_sequence<int, RemRanks...>{}, point,
                tag);
    }
  }

  const MDRP m_md_range_policy;
  const Functor m_functor;
  static constexpr std::conditional_t<std::is_void_v<Tag>, NoTag, Tag> m_tag{};
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::OpenMP> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;
  using WorkTag       = typename MDRangePolicy::work_tag;

  using Member = typename Policy::member_type;

  using index_type   = typename Policy::index_type;
  //using iterate_type = typename Kokkos::Impl::HostIterateTile<
  //    MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;
  using iterate_type = FlatIterate<MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag>;

  OpenMPInternal* m_instance;
  const iterate_type m_iter;

  //inline void exec_range(const Member ibeg, const Member iend) const {
  //  KOKKOS_PRAGMA_IVDEP_IF_ENABLED
  //  for (Member iwork = ibeg; iwork < iend; ++iwork) {
  //    m_iter(iwork);
  //  }
  //}

  template <class Policy>
  typename std::enable_if_t<
      std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(m_instance->thread_pool_size())
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (index_type iwork = 0; iwork < m_iter.m_rp.m_num_tiles; ++iwork) {
      m_iter(iwork);
    }
  }

  template <class Policy>
  std::enable_if_t<
      !std::is_same_v<typename Policy::schedule_type::type, Kokkos::Dynamic>>
  execute_parallel() const {
#pragma omp parallel for schedule(static, 1) \
    num_threads(m_instance->thread_pool_size())
    KOKKOS_PRAGMA_IVDEP_IF_ENABLED
    for (index_type iwork = 0; iwork < m_iter.m_rp.m_num_tiles; ++iwork) {
      m_iter(iwork);
    }
  }

 public:
  inline void execute() const {
    // Serialize kernels on the same execution space instance
    std::lock_guard<std::mutex> lock(m_instance->m_instance_mutex);

    if (execute_in_serial(m_iter.policy().space())) {
      m_iter.exec(Static{});
      //exec_range(0, m_iter.m_rp.m_num_tiles);
      return;
    }

    m_iter.exec(Static{});
    //execute_parallel<Policy>();
  }

  inline ParallelFor(const FunctorType& arg_functor, MDRangePolicy arg_policy)
      : m_instance(nullptr), m_iter(arg_policy, arg_functor) {
    m_instance = arg_policy.space().impl_internal_space_instance();
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
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

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::OpenMP> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::OpenMP, Properties...>;
  using WorkTag  = typename Policy::work_tag;
  using SchedTag = typename Policy::schedule_type::type;
  using Member   = typename Policy::member_type;

  OpenMPInternal* m_instance;
  const FunctorType m_functor;
  const Policy m_policy;
  const size_t m_shmem_size;

  template <class TagType>
  inline static std::enable_if_t<(std::is_void_v<TagType>)> exec_team(
      const FunctorType& functor, HostThreadTeamData& data,
      const int league_rank_begin, const int league_rank_end,
      const int league_size) {
    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

  template <class TagType>
  inline static std::enable_if_t<(!std::is_void_v<TagType>)> exec_team(
      const FunctorType& functor, HostThreadTeamData& data,
      const int league_rank_begin, const int league_rank_end,
      const int league_size) {
    const TagType t{};

    for (int r = league_rank_begin; r < league_rank_end;) {
      functor(t, Member(data, r, league_size));

      if (++r < league_rank_end) {
        // Don't allow team members to lap one another
        // so that they don't overwrite shared memory.
        if (data.team_rendezvous()) {
          data.team_rendezvous_release();
        }
      }
    }
  }

 public:
  inline void execute() const {
    enum { is_dynamic = std::is_same_v<SchedTag, Kokkos::Dynamic> };

    const size_t pool_reduce_size  = 0;  // Never shrinks
    const size_t team_reduce_size  = TEAM_REDUCE_SIZE * m_policy.team_size();
    const size_t team_shared_size  = m_shmem_size;
    const size_t thread_local_size = 0;  // Never shrinks

    // Serialize kernels on the same execution space instance
    std::lock_guard<std::mutex> lock(m_instance->m_instance_mutex);

    m_instance->resize_thread_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    if (execute_in_serial(m_policy.space())) {
      ParallelFor::template exec_team<WorkTag>(
          m_functor, *(m_instance->get_thread_data()), 0,
          m_policy.league_size(), m_policy.league_size());

      return;
    }

#pragma omp parallel num_threads(m_instance->thread_pool_size())
    {
      HostThreadTeamData& data = *(m_instance->get_thread_data());

      const int active = data.organize_team(m_policy.team_size());

      if (active) {
        data.set_work_partition(
            m_policy.league_size(),
            (0 < m_policy.chunk_size() ? m_policy.chunk_size()
                                       : m_policy.team_iter()));
      }

      if (is_dynamic) {
        // Must synchronize to make sure each team has set its
        // partition before beginning the work stealing loop.
        if (data.pool_rendezvous()) data.pool_rendezvous_release();
      }

      if (active) {
        std::pair<int64_t, int64_t> range(0, 0);

        do {
          range = is_dynamic ? data.get_work_stealing_chunk()
                             : data.get_work_partition();

          ParallelFor::template exec_team<WorkTag>(m_functor, data, range.first,
                                                   range.second,
                                                   m_policy.league_size());

        } while (is_dynamic && 0 <= range.first);
      }

      data.disband_team();
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_instance(nullptr),
        m_functor(arg_functor),
        m_policy(arg_policy),
        m_shmem_size(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         m_functor, m_policy.team_size())) {
    m_instance = m_policy.space().impl_internal_space_instance();

    if ((m_policy.scratch_size(0) + FunctorTeamShmemSize<FunctorType>::value(
                                        m_functor, m_policy.team_size())) >
        static_cast<size_t>(TeamPolicy<Kokkos::OpenMP>::scratch_size_max(0))) {
      std::stringstream error;
      error << "Kokkos::parallel_for<OpenMP>: Requested too much scratch "
               "memory on level 0. Requested: "
            << m_policy.scratch_size(0) +
                   FunctorTeamShmemSize<FunctorType>::value(
                       m_functor, m_policy.team_size())
            << ", Maximum: " << TeamPolicy<Kokkos::OpenMP>::scratch_size_max(0);
      Kokkos::Impl::throw_runtime_exception(error.str().c_str());
    }
    if (m_policy.scratch_size(1) >
        static_cast<size_t>(TeamPolicy<Kokkos::OpenMP>::scratch_size_max(1))) {
      std::stringstream error;
      error << "Kokkos::parallel_for<OpenMP>: Requested too much scratch "
               "memory on level 1. Requested: "
            << m_policy.scratch_size(1)
            << ", Maximum: " << TeamPolicy<Kokkos::OpenMP>::scratch_size_max(1);
      Kokkos::Impl::throw_runtime_exception(error.str().c_str());
    }
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#undef KOKKOS_PRAGMA_IVDEP_IF_ENABLED
#undef KOKKOS_OPENMP_OPTIONAL_CHUNK_SIZE

#endif /* KOKKOS_OPENMP_PARALLEL_FOR_HPP */
