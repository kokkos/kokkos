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

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP

#include <OpenACC/Kokkos_OpenACC_Team.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>

#ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class ReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenACC,
                                                  Properties...>;


  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using value_type   = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    const FunctorType a_functor(m_functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    static constexpr int UseReducer = is_reducer<ReducerType>::value;

    if constexpr (!UseReducer) {
#pragma acc parallel loop gang vector reduction(+ : tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, 1,
                                          vector_length);
        a_functor(team, tmp);
      }
      m_result_ptr[0] = tmp;
    } else {
      OpenACCReducerWrapperTeams<ReducerType, FunctorType, Policy,
                                 TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result_view,
      std::enable_if_t<Kokkos::is_view<ViewType>::value,
                       void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp = ValueType();
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

//FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
      static_assert(Kokkos::Impl::always_false<Lambda>::value, "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Thread vector level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType const& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType vector_reduce;
  ReducerType::init(vector_reduce);

#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, vector_reduce);
  }
  result.reference() = vector_reduce;
}

//FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
      static_assert(Kokkos::Impl::always_false<Lambda>::value, "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

//Team-vector-level reduction is not supported in the OpenACC backend.
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos

#else

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class ReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy = Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenACC,
                                                  Properties...>;


  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using value_type   = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

 public:
  inline void execute() const {
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    const FunctorType a_functor(m_functor);
    value_type tmp;
    ValueInit::init(a_functor, &tmp);
    static constexpr int UseReducer = is_reducer<ReducerType>::value;

    if constexpr (!UseReducer) {
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size)  vector_length(vector_length) reduction(+:tmp) copyin(a_functor)
      for (int i = 0; i < league_size; i++) {
        int league_id = i;
        typename Policy::member_type team(league_id, league_size, team_size,
                                          vector_length);
        a_functor(team, tmp);
      }
      m_result_ptr[0] = tmp;
    } else {
      OpenACCReducerWrapperTeams<ReducerType, FunctorType, Policy,
                                 TagType>::reduce(tmp, m_policy, a_functor);
      m_result_ptr[0] = tmp;
    }
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result_view,
      std::enable_if_t<Kokkos::is_view<ViewType>::value,
                       void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  inline ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine worker
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#pragma acc loop worker reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

#pragma acc routine worker
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType tmp = ValueType();
#pragma acc loop worker reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  result = tmp;
}

//FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
      static_assert(Kokkos::Impl::always_false<Lambda>::value, "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Thread vector level implementation
#pragma acc routine vector
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}

#pragma acc routine vector
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ReducerType const& result) {
  using ValueType = typename ReducerType::value_type;

  ValueType vector_reduce;
  ReducerType::init(vector_reduce);

#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, vector_reduce);
  }
  result.reference() = vector_reduce;
}

//FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
      static_assert(Kokkos::Impl::always_false<Lambda>::value, "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();

//Team-vector-level reduction is not supported in the OpenACC backend.
#pragma acc loop seq
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos

#endif /* #ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS */

#endif /* #ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP */
