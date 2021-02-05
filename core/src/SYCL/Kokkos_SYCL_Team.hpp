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

#ifndef KOKKOS_SYCL_TEAM_HPP
#define KOKKOS_SYCL_TEAM_HPP

#include <Kokkos_Macros.hpp>

#ifdef KOKKOS_ENABLE_SYCL

#include <utility>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/**\brief  Team member_type passed to TeamPolicy or TeamTask closures.
 */
class SYCLTeamMember {
 public:
  using execution_space      = Kokkos::Experimental::SYCL;
  using scratch_memory_space = execution_space::scratch_memory_space;

 private:
  mutable void* m_team_reduce;
  scratch_memory_space m_team_shared;
  int m_team_reduce_size;
  sycl::nd_item<1> m_item;

 public:
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& team_scratch(
      const int& level) const {
    return m_team_shared.set_team_thread_mode(level, 1, 0);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space& thread_scratch(
      const int& level) const {
    return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_item.get_group(0); }
  KOKKOS_INLINE_FUNCTION int league_size() const {
    return m_item.get_group_range(0);
  }
  KOKKOS_INLINE_FUNCTION int team_rank() const {
    return m_item.get_local_id(0);
  }
  KOKKOS_INLINE_FUNCTION int team_size() const {
    return m_item.get_local_range(0);
  }
  KOKKOS_INLINE_FUNCTION void team_barrier() const { m_item.barrier(); }

  //--------------------------------------------------------------------------

  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType& /*val*/,
                                             const int& /*thread_id*/) const {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(Closure const& f, ValueType& val,
                                             const int& thread_id) const {
    f(val);
    team_broadcast(val, thread_id);
  }

  //--------------------------------------------------------------------------
  /**\brief  Reduction across a team
   */
  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      team_reduce(ReducerType const& reducer) const noexcept {
    team_reduce(reducer, reducer.reference());
  }

  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      team_reduce(ReducerType const& /*reducer*/,
                  typename ReducerType::value_type& /*value*/) const noexcept {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }

  //--------------------------------------------------------------------------
  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering
   *          with intra-team non-deterministic ordering accumulation.
   *
   *  The global inter-team accumulation value will, at the end of the
   *  league's parallel execution, be the scan's total.
   *  Parallel execution ordering of the league's teams is non-deterministic.
   *  As such the base value for each team's scan operation is similarly
   *  non-deterministic.
   */
  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type team_scan(const Type& value,
                                        Type* const /*global_accum*/) const {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
    return value;
  }

  /** \brief  Intra-team exclusive prefix sum with team_rank() ordering.
   *
   *  The highest rank thread can compute the reduction total as
   *    reduction_total = dev.team_scan( value ) + value ;
   */
  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type team_scan(const Type& value) const {
    return this->template team_scan<Type>(value, nullptr);
  }

  //----------------------------------------

  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION static
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      vector_reduce(ReducerType const& reducer) {
    vector_reduce(reducer, reducer.reference());
  }

  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION static
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      vector_reduce(ReducerType const& /*reducer*/,
                    typename ReducerType::value_type& /*value*/) {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }

  //--------------------------------------------------------------------------
  /**\brief  Global reduction across all blocks
   *
   *  Return !0 if reducer contains the final value
   */
  template <typename ReducerType>
  KOKKOS_INLINE_FUNCTION static
      typename std::enable_if<is_reducer<ReducerType>::value, int>::type
      global_reduce(ReducerType const& /*reducer*/,
                    int* const /*global_scratch_flags*/,
                    void* const /*global_scratch_space*/, void* const /*shmem*/,
                    int const /*shmem_size*/) {
    // FIXME_SYCL
    Kokkos::abort("Not implemented!");
  }

  //----------------------------------------
  // Private for the driver

  KOKKOS_INLINE_FUNCTION
  SYCLTeamMember(void* shared, const int shared_begin, const int shared_size,
                 void* scratch_level_1_ptr, const int scratch_level_1_size,
                 const sycl::nd_item<1> item)
      : m_team_reduce(shared),
        m_team_shared(static_cast<char*>(shared) + shared_begin, shared_size,
                      scratch_level_1_ptr, scratch_level_1_size),
        m_team_reduce_size(shared_begin),
        m_item(item) {}

 public:
  // Declare to avoid unused private member warnings which are trigger
  // when SFINAE excludes the member function which uses these variables
  // Making another class a friend also surpresses these warnings
  bool impl_avoid_sfinae_warning() const noexcept {
    return m_team_reduce_size > 0 && m_team_reduce != nullptr;
  }
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <typename iType>
struct TeamThreadRangeBoundariesStruct<iType, SYCLTeamMember> {
  using index_type = iType;
  const SYCLTeamMember& member;
  const iType start;
  const iType end;

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const SYCLTeamMember& thread_, iType count)
      : member(thread_), start(0), end(count) {}

  KOKKOS_INLINE_FUNCTION
  TeamThreadRangeBoundariesStruct(const SYCLTeamMember& thread_, iType begin_,
                                  iType end_)
      : member(thread_), start(begin_), end(end_) {}
};

template <typename iType>
struct TeamVectorRangeBoundariesStruct<iType, SYCLTeamMember> {
  using index_type = iType;
  const SYCLTeamMember& member;
  const iType start;
  const iType end;

  KOKKOS_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const SYCLTeamMember& thread_,
                                  const iType& count)
      : member(thread_), start(0), end(count) {}

  KOKKOS_INLINE_FUNCTION
  TeamVectorRangeBoundariesStruct(const SYCLTeamMember& thread_,
                                  const iType& begin_, const iType& end_)
      : member(thread_), start(begin_), end(end_) {}
};

template <typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, SYCLTeamMember> {
  using index_type = iType;
  const SYCLTeamMember& member;
  const index_type start;
  const index_type end;

  KOKKOS_INLINE_FUNCTION
  ThreadVectorRangeBoundariesStruct(const SYCLTeamMember& thread,
                                    index_type count)
      : member(thread), start(static_cast<index_type>(0)), end(count) {}

  KOKKOS_INLINE_FUNCTION
  ThreadVectorRangeBoundariesStruct(const SYCLTeamMember& thread,
                                    index_type arg_begin, index_type arg_end)
      : member(thread), start(arg_begin), end(arg_end) {}
};

}  // namespace Impl

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    TeamThreadRange(const Impl::SYCLTeamMember& thread, iType count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type, Impl::SYCLTeamMember>
TeamThreadRange(const Impl::SYCLTeamMember& thread, iType1 begin, iType2 end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    TeamVectorRange(const Impl::SYCLTeamMember& thread, const iType& count) {
  return Impl::TeamVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type, Impl::SYCLTeamMember>
TeamVectorRange(const Impl::SYCLTeamMember& thread, const iType1& begin,
                const iType2& end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    ThreadVectorRange(const Impl::SYCLTeamMember& thread, iType count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, count);
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>
    ThreadVectorRange(const Impl::SYCLTeamMember& thread, iType arg_begin,
                      iType arg_end) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>(
      thread, arg_begin, arg_end);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::SYCLTeamMember> PerTeam(
    const Impl::SYCLTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::SYCLTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::SYCLTeamMember> PerThread(
    const Impl::SYCLTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::SYCLTeamMember>(thread);
}

//----------------------------------------------------------------------------

/** \brief  Inter-thread parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N).
 *
 * The range [0..N) is mapped to all threads of the calling thread team.
 */
template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
        loop_boundaries,
    const Closure& closure) {
  for (iType i = loop_boundaries.start + loop_boundaries.member.team_rank();
       i < loop_boundaries.end; i += loop_boundaries.member.team_size())
    closure(i);
}

//----------------------------------------------------------------------------

/** \brief  Inter-thread parallel_reduce with a reducer.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& /*loop_boundaries*/,
                    const Closure& /*closure*/,
                    const ReducerType& /*reducer*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

/** \brief  Inter-thread parallel_reduce assuming summation.
 *
 *  Executes closure(iType i, ValueType & val) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all threads of the
 *  calling thread team and a summation of val is
 *  performed and put into result.
 */
template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& /*loop_boundaries*/,
                    const Closure& /*closure*/, ValueType& /*result*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

/** \brief  Inter-thread parallel exclusive prefix sum.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
 *
 *  The range [0..N) is mapped to each rank in the team (whose global rank is
 *  less than N) and a scan operation is performed. The last call to closure has
 *  final == true.
 */
// This is the same code as in CUDA and largely the same as in OpenMPTarget
template <typename iType, typename FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
        loop_bounds,
    const FunctorType& lambda) {
  // Extract value_type from lambda
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void,
      FunctorType>::value_type;

  const auto start     = loop_bounds.start;
  const auto end       = loop_bounds.end;
  auto& member         = loop_bounds.member;
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
}

template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
        loop_boundaries,
    const Closure& closure) {
  for (auto i = loop_boundaries.start; i != loop_boundaries.end; ++i)
    closure(i);
}

template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<Kokkos::is_reducer<ReducerType>::value>::type
    parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& /*loop_boundaries*/,
                    const Closure& /*closure*/,
                    const ReducerType& /*reducer*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
    parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember>& /*loop_boundaries*/,
                    const Closure& /*closure*/, ValueType& /*result*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel_for.
 *
 *  Executes closure(iType i) for each i=[0..N)
 *
 * The range [0..N) is mapped to all vector lanes of the calling thread.
 */
template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
        loop_boundaries,
    const Closure& closure) {
  for (auto i = loop_boundaries.start; i != loop_boundaries.end; ++i)
    closure(i);
}

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls closure(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed using +=
 *  and output into result.
 *
 *  The identity value for the += operator is assumed to be the default
 *  constructed value.
 */
template <typename iType, class Closure, class ReducerType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<is_reducer<ReducerType>::value>::type
    parallel_reduce(Impl::ThreadVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember> const& /*loop_boundaries*/,
                    Closure const& /*closure*/,
                    ReducerType const& /*reducer*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

/** \brief  Intra-thread vector parallel_reduce.
 *
 *  Calls closure(iType i, ValueType & val) for each i=[0..N).
 *
 *  The range [0..N) is mapped to all vector lanes of
 *  the calling thread and a reduction of val is performed using +=
 *  and output into result.
 *
 *  The identity value for the += operator is assumed to be the default
 *  constructed value.
 */
template <typename iType, class Closure, typename ValueType>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<!is_reducer<ValueType>::value>::type
    parallel_reduce(Impl::ThreadVectorRangeBoundariesStruct<
                        iType, Impl::SYCLTeamMember> const& /*loop_boundaries*/,
                    Closure const& /*closure*/, ValueType& /*result*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

//----------------------------------------------------------------------------

/** \brief  Intra-thread vector parallel exclusive prefix sum.
 *
 *  Executes closure(iType i, ValueType & val, bool final) for each i=[0..N)
 *
 *  The range [0..N) is mapped to all vector lanes in the
 *  thread and a scan operation is performed.
 *  The last call to closure has final == true.
 */
template <typename iType, class Closure>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::SYCLTeamMember>&
    /*loop_boundaries*/,
    const Closure& /*closure*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

}  // namespace Kokkos

namespace Kokkos {

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::SYCLTeamMember>&,
    const FunctorType& /*lambda*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::SYCLTeamMember>&,
    const FunctorType& /*lambda*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::SYCLTeamMember>&,
    const FunctorType& /*lambda*/, ValueType& /*val*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::SYCLTeamMember>& /*single_struct*/,
    const FunctorType& /*lambda*/, ValueType& /*val*/) {
  // FIXME_SYCL
  Kokkos::abort("Not implemented!");
}

}  // namespace Kokkos

#endif

#endif /* #ifndef KOKKOS_SYCL_TEAM_HPP */
