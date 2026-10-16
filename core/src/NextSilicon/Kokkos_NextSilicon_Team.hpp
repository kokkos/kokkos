// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_TEAM_HPP
#define KOKKOS_NEXTSILICON_TEAM_HPP

#include <Kokkos_Assert.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <impl/Kokkos_Error.hpp>

#include <iostream>
#include <sstream>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class NextSiliconTeamMember {
 public:
  using execution_space      = Kokkos::Experimental::NextSilicon;
  using scratch_memory_space = execution_space::scratch_memory_space;
  using team_handle          = NextSiliconTeamMember;

 private:
  scratch_memory_space m_team_shared;
  int m_league_rank;
  int m_league_size;
  // FIXME_NEXTSILICON: team size always 1
  static constexpr int m_team_rank = 0;
  static constexpr int m_team_size = 1;

 public:
  KOKKOS_FUNCTION
  const scratch_memory_space& team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_FUNCTION
  const scratch_memory_space& team_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, 1, 0);
  }

  KOKKOS_FUNCTION
  const scratch_memory_space& thread_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
  }

  KOKKOS_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_FUNCTION int league_size() const { return m_league_size; }
  // FIXME_NEXTSILICON: make these non-constexpr when larger team sizes are
  // supported
  KOKKOS_FUNCTION constexpr int team_rank() const { return m_team_rank; }
  KOKKOS_FUNCTION constexpr int team_size() const { return m_team_size; }

  KOKKOS_FUNCTION void team_barrier() const {
    // barrier is a no-op with team size = 1
  }

  template <class ValueType>
  KOKKOS_FUNCTION void team_broadcast(ValueType& value, int thread_id) const {
    (void)value;
    (void)thread_id;
    // broadcast is a no-op with team size = 1
  }

  template <class Closure, class ValueType>
  KOKKOS_FUNCTION void team_broadcast(const Closure& f, ValueType& value,
                                      int thread_id) const {
    f(value);
    team_broadcast(value, thread_id);
  }

  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<is_reducer_v<ReducerType>>
  team_reduce(ReducerType const& reducer) const noexcept {
    team_reduce(reducer, reducer.reference());
  }

  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION std::enable_if_t<is_reducer_v<ReducerType>>
  team_reduce(ReducerType const& reducer,
              typename ReducerType::value_type& value) const noexcept {
    // Team size = 1, value is already the result
    reducer.reference() = value;
  }

  template <typename ArgType>
  KOKKOS_FUNCTION ArgType team_scan(const ArgType& value,
                                    ArgType* const global_accum) const {
    if (global_accum) {
      ArgType accum = atomic_fetch_add(global_accum, value);
      return accum;
    }
    return ArgType(0);  // exclusive scan
  }

  template <typename Type>
  KOKKOS_FUNCTION Type team_scan(const Type& value) const {
    return this->template team_scan<Type>(value, 0);
  }

  //----------------------------------------
  // Constructed by the ParallelFor/ParallelReduce team drivers

  NextSiliconTeamMember(const int league_rank, const int league_size,
                        [[maybe_unused]] const int team_rank,
                        [[maybe_unused]] const int team_size,
                        const scratch_memory_space& team_shared)
      : m_team_shared(team_shared),
        m_league_rank(league_rank),
        m_league_size(league_size) {
    KOKKOS_ASSERT(team_rank == m_team_rank);
    KOKKOS_ASSERT(team_size == m_team_size);
  }
};

template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::NextSilicon, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  //----------------------------------------

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelForTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelReduceTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType, class ReducerType>
  static int team_size_max(const FunctorType&, const ReducerType&,
                           const ParallelReduceTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&, const ParallelForTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&,
                                   const ParallelReduceTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType, class ReducerType>
  static int team_size_recommended(const FunctorType&, const ReducerType&,
                                   const ParallelReduceTag&) {
    return s_team_size_max;
  }

  // FIXME_NEXTSILICON: L0 limit is arbitrary (both levels currently live in
  // the global league scratch buffer). L1 matches the Serial and OpenMP
  // backends.
  static int scratch_size_max(int level) {
    return (level == 0 ? 64 * 1024 : 80 * 1024 * 1024);
  }

  //----------------------------------------

 private:
  // FIXME_NEXTSILICON: team size is always 1, so this is also the recommended
  // and AUTO team size
  static constexpr int s_team_size_max = 1;
  // Use arbitrary large number, is meant as a vectorizable length (as Serial)
  static constexpr int s_vector_length_max  = 1024;
  static constexpr int s_auto_vector_length = 32;

  int m_league_size;
  // FIXME_NEXTSILICON: team size is always 1
  static constexpr int m_team_size = s_team_size_max;
  int m_team_alloc                 = 0;
  int m_team_iter                  = 0;
  std::array<size_t, 2> m_team_scratch_size;
  std::array<size_t, 2> m_thread_scratch_size;
  int m_chunk_size;

  void init(const int league_size_request, const int team_size_request) {
    m_league_size = league_size_request;
    set_team_size(team_size_request);
    set_auto_chunk_size();
  }

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 public:
  bool impl_auto_team_size() const { return false; }
  bool impl_auto_vector_length() const { return false; }
  constexpr int impl_vector_length() const { return 1; }
  void impl_set_team_size(const int size) {
    // Part of tuner interface
    set_team_size(size);
  }
  void impl_set_vector_length(const int) {
    // Part of tuner interface
  }
  // FIXME_NEXTSILICON: team size is always 1
  constexpr int team_size() const { return m_team_size; }
  constexpr int league_size() const { return m_league_size; }

  size_t scratch_size(const int& level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = team_size();
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  size_t team_scratch_size(int level) const {
    return m_team_scratch_size[level];
  }

  size_t thread_scratch_size(int level) const {
    return m_thread_scratch_size[level];
  }

  Kokkos::Experimental::NextSilicon space() const {
    return Kokkos::Experimental::NextSilicon();
  }

  template <class... OtherProperties>
  TeamPolicyInternal(const TeamPolicyInternal<OtherProperties...>& p)
      : m_league_size(p.m_league_size),
        // FIXME_NEXTSILICON: team size is always 1
        // m_team_size(p.m_team_size),
        m_team_alloc(p.m_team_alloc),
        m_team_scratch_size(p.m_team_scratch_size),
        m_thread_scratch_size(p.m_thread_scratch_size),
        m_chunk_size(p.m_chunk_size) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, s_team_size_max);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, s_team_size_max);
  }
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, s_team_size_max);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /*team_size_request*/,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, s_team_size_max);
  }

  TeamPolicyInternal(const PolicyUpdate, const TeamPolicyInternal& other,
                     typename traits::execution_space)
      : TeamPolicyInternal(other) {
    // FIXME_NEXTSILICON: implement PolicyUpdate for NextSilicon
  }
  static int vector_length_max() { return s_vector_length_max; }
  int team_alloc() const { return m_team_alloc; }
  // FIXME_NEXTSILICON: unused?
  // int team_iter() const { return m_team_iter; }
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
  void set_team_size(const int size) {
    // Same message format as Serial; TestTeamBasic's exceed_max_team_size
    // matches on it. Sizes < 1 are rejected by TeamPolicy before we get here.
    if (size > s_team_size_max) {
      std::stringstream ss;
      ss << "Kokkos::TeamPolicy<NextSilicon>: Requested too large team size. "
            "Requested: "
         << size << ", Maximum: " << s_team_size_max;
      Kokkos::Impl::throw_runtime_exception(ss.str());
    }
    // FIXME_NEXTSILICON: team size is always 1
    // m_team_size = size;
  }

  /** \brief finalize chunk_size if it was set to AUTO*/
  void set_auto_chunk_size() {
    // FIXME_NEXTSILICON: set_auto_chunk_size logic copied from OpenACC, needs
    // to be redone. The team drivers do not pass chunk_size() to their inner
    // RangePolicy yet.

    int concurrency = 2048 * m_team_size;

    int new_chunk_size = 1;
    while (new_chunk_size * 100 * concurrency < m_league_size)
      new_chunk_size *= 2;
    // FIXME_NEXTSILICON: team size is always 1
    // if (new_chunk_size < m_team_size) {
    //   new_chunk_size = 1;
    //   while ((new_chunk_size * 40 * concurrency < m_league_size) &&
    //          (new_chunk_size < m_team_size))
    //     new_chunk_size *= 2;
    // }
    m_chunk_size = new_chunk_size;
  }

 public:
  using member_type = Impl::NextSiliconTeamMember;
};

// Throw if the per-team scratch request on `level` (for level 0 including the
// functor's team shmem) exceeds scratch_size_max(level).
inline void nextsilicon_check_team_scratch_size(const char* construct,
                                                const int level,
                                                const size_t requested) {
  using Policy       = Kokkos::TeamPolicy<Kokkos::Experimental::NextSilicon>;
  const auto maximum = static_cast<size_t>(Policy::scratch_size_max(level));
  if (requested > maximum) {
    std::stringstream error;
    error << construct << ": Requested too much scratch memory on level "
          << level << ". Requested: " << requested << ", Maximum: " << maximum;
    Kokkos::Impl::throw_runtime_exception(error.str());
  }
}
}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <typename iType>
struct TeamThreadRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const iType start;
  const iType end;
  const NextSiliconTeamMember& team;

  TeamThreadRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  iType count)
      : start(0), end(count), team(thread_) {}
  TeamThreadRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  iType begin_, iType end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const NextSiliconTeamMember& team;

  ThreadVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                    index_type count)
      : start(0), end(count), team(thread_) {}
  ThreadVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                    index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct TeamVectorRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const NextSiliconTeamMember& team;

  TeamVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  index_type count)
      : start(0), end(count), team(thread_) {}
  TeamVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    TeamThreadRange(const Impl::NextSiliconTeamMember& thread,
                    const iType& count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
TeamThreadRange(const Impl::NextSiliconTeamMember& thread, const iType1& begin,
                const iType2& end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    ThreadVectorRange(const Impl::NextSiliconTeamMember& thread,
                      const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::ThreadVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
ThreadVectorRange(const Impl::NextSiliconTeamMember& thread,
                  const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::NextSiliconTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamVectorRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    TeamVectorRange(const Impl::NextSiliconTeamMember& thread,
                    const iType& count) {
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
TeamVectorRange(const Impl::NextSiliconTeamMember& thread,
                const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember> PerTeam(
    const Impl::NextSiliconTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::NextSiliconTeamMember> PerThread(
    const Impl::NextSiliconTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>(thread);
}
}  // namespace Kokkos

namespace Kokkos {

// One lane does the lambda, then broadcasts val
template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda) {
  // vector length is always 1
  lambda();
}

// One team member does the lambda, then broadcasts val
template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<
        Impl::NextSiliconTeamMember>& /*single_struct*/,
    const FunctorType& lambda) {
  // team size is always 1
  lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda, ValueType& val) {
  // vector length is always 1
  lambda(val);
}

// One team member does the lambda, then broadcasts val
template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<
        Impl::NextSiliconTeamMember>& /*single_struct*/,
    const FunctorType& lambda, ValueType& val) {
  // team size is always 1
  lambda(val);
}
}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_TEAM_HPP */
