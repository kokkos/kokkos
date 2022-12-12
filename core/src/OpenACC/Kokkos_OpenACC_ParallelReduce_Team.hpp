//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP

#include <OpenACC/Kokkos_OpenACC_Team.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct OpenACCParallelReduceTeamHelper {
  OpenACCParallelReduceTeamHelper(Functor const&, Reducer const&,
                                  Policy const&) {
    static_assert(!Kokkos::Impl::always_true<Functor>::value,
                  "not implemented");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class FunctorType, class ReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<FunctorType,
                                   Kokkos::TeamPolicy<Properties...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
 private:
  using Policy =
      TeamPolicyInternal<Kokkos::Experimental::OpenACC, Properties...>;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, ReducerTypeFwd>;
  using value_type   = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;

  FunctorType m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  pointer_type m_result_ptr;

 public:
  void execute() const {
    auto league_size   = m_policy.league_size();
    auto team_size     = m_policy.team_size();
    auto vector_length = m_policy.impl_vector_length();

    value_type tmp;
    typename Analysis::Reducer final_reducer(
        &ReducerConditional::select(m_functor, m_reducer));
    final_reducer.init(&tmp);

    Kokkos::Experimental::Impl::OpenACCParallelReduceTeamHelper(
        Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy>(
            m_functor),
        std::conditional_t<is_reducer_v<ReducerType>, ReducerType,
                           Sum<value_type>>(tmp),
        m_policy);

    m_result_ptr[0] = tmp;
  }

  template <class ViewType>
  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ViewType& arg_result_view,
                 std::enable_if_t<Kokkos::is_view_v<ViewType>>* = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {}

  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

#ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS

namespace Kokkos::Experimental::Impl {

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamSum(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(+ : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamProd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(* : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamMin(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(min : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamMax(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(max : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamLAnd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(&& : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamLOr(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(|| : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamBAnd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(& : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamBOr(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) vector_length(team_size*vector_length) reduction(| : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size * team_size * vector_length; i++) {
    int league_id = i / (team_size * vector_length);
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

}  // namespace Kokkos::Experimental::Impl

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer_v<ValueType>>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() / loop_boundaries.team.vector_length();
  if (j_start == 0) {
#pragma acc loop seq
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, tmp);
    result = tmp;
  }
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp;
  reducer.init(tmp);
  iType j_start =
      loop_boundaries.team.team_rank() / loop_boundaries.team.vector_length();
  if (j_start == 0) {
#pragma acc loop seq
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, tmp);
    reducer.reference() = tmp;
  }
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Thread vector level implementation
#pragma acc routine seq
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer_v<ValueType>>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
#pragma acc loop seq
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    result = tmp;
  }
}

#pragma acc routine seq
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp;
  reducer.init(tmp);
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
#pragma acc loop seq
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    reducer.reference() = tmp;
  }
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
#pragma acc loop seq
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    result = tmp;
  }
}

}  // namespace Kokkos

#else /* #ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS */

// FIXME_OPENACC: below implementation conforms to the OpenACC standard, but
// the NVHPC compiler (V22.11) fails due to the lack of support for lambda
// expressions containing parallel loops.
// Disabled for the time being.
#if 0

namespace Kokkos::Experimental::Impl {

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamSum(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(+ : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamProd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(* : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamMin(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(min : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamMax(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(max : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamLAnd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(&& : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamLOr(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(|| : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamBAnd(Policy const policy, ValueType& aval,
                                   Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(& : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

template <class Policy, class ValueType, class Functor>
void OpenACCParallelReduceTeamBOr(Policy const policy, ValueType& aval,
                                  Functor const& afunctor, int async_arg) {
  auto const functor       = afunctor;
  auto val                 = aval;
  auto const league_size   = policy.league_size();
  auto const team_size     = policy.team_size();
  auto const vector_length = policy.impl_vector_length();
// clang-format off
#pragma acc parallel loop gang vector num_gangs(league_size) num_workers(team_size) vector_length(vector_length) reduction(| : val) copyin(functor) async(async_arg)
  // clang-format on
  for (int i = 0; i < league_size; i++) {
    int league_id = i;
    typename Policy::member_type team(league_id, league_size, team_size,
                                      vector_length);
    functor(team, val);
  }
  aval = val;
}

}  // namespace Kokkos::Experimental::Impl

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
#pragma acc routine worker
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer_v<ValueType>>
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
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp = ValueType();
  reducer.init(tmp);
#pragma acc loop worker reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, tmp);
  reducer.reference() = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
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
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::OpenACCTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType & reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp;
  reducer.init(tmp);
#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  reducer.reference() = tmp;
}

// FIXME_OPENACC: custom reduction is not implemented.
template <typename iType, class Lambda, typename ValueType, class JoinType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenACCTeamMember>& loop_boundaries,
    const Lambda& lambda, const JoinType& join, ValueType& init_result) {
  static_assert(!Kokkos::Impl::always_true<Lambda>::value,
                "custom reduction is not implemented");
}

// Hierarchical Parallelism -> Team vector level implementation
#pragma acc routine vector
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<iType, Impl::OpenACCTeamMember>&
        loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
#pragma acc loop vector reduction(+ : tmp)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
    lambda(i, tmp);
  }
  result = tmp;
}
}  // namespace Kokkos
#endif /* #if 0 */

#endif /* #ifdef KOKKOS_ENABLE_OPENACC_COLLAPSE_HIERARCHICAL_CONSTRUCTS */

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(REDUCER)           \
  template <class Functor, class Scalar, class Space, class... Traits>     \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceTeamHelper<      \
      Functor, Kokkos::REDUCER<Scalar, Space>,                             \
      Kokkos::Impl::TeamPolicyInternal<Traits...>, true> {                 \
    using Policy    = Kokkos::Impl::TeamPolicyInternal<Traits...>;         \
    using Reducer   = REDUCER<Scalar, Space>;                              \
    using ValueType = typename Reducer::value_type;                        \
                                                                           \
    OpenACCParallelReduceTeamHelper(Functor const& functor,                \
                                    Reducer const& reducer,                \
                                    Policy const& policy) {                \
      auto league_size   = policy.league_size();                           \
      auto team_size     = policy.team_size();                             \
      auto vector_length = policy.impl_vector_length();                    \
                                                                           \
      if (league_size <= 0) {                                              \
        return;                                                            \
      }                                                                    \
                                                                           \
      ValueType val;                                                       \
      reducer.init(val);                                                   \
                                                                           \
      int const async_arg = policy.space().acc_async_queue();              \
                                                                           \
      OpenACCParallelReduceTeam##REDUCER(policy, val, functor, async_arg); \
                                                                           \
      reducer.reference() = val;                                           \
    }                                                                      \
  }

KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(Sum);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(Prod);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(Min);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(Max);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(LAnd);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(LOr);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(BAnd);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER(BOr);

#undef KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_TEAM_HELPER

#endif /* #ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_TEAM_HPP */
