// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce.hpp>
#include <mutex>

namespace Kokkos::Experimental::Impl {
template <typename Member, typename Functor>
class NextSiliconParallelReduceTeamPolicyFunctor {
 public:
  NextSiliconParallelReduceTeamPolicyFunctor(Functor const& functor,
                                             const int league_size,
                                             const int vector_length,
                                             std::byte* league_scratch_buffer,
                                             const size_t L0_size,
                                             const size_t L1_size)
      : m_functor(functor),
        m_league_size(league_size),
        m_vector_length(vector_length),
        m_league_scratch_buffer(league_scratch_buffer),
        m_L0_size(L0_size),
        m_L1_size(L1_size) {}

  template <typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(const int league_rank,
                                         ReducerValueType& update) const {
    m_functor(league_rank_to_member(league_rank), update);
  }

  template <typename Tag, typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(Tag, const int league_rank,
                                         ReducerValueType& update) const {
    m_functor(Tag{}, league_rank_to_member(league_rank), update);
  }

  template <typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(const int league_rank,
                                         ReducerValueType* update_ptr) const {
    m_functor(league_rank_to_member(league_rank), update_ptr);
  }

  template <typename Tag, typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(Tag, const int league_rank,
                                         ReducerValueType* update_ptr) const {
    m_functor(Tag{}, league_rank_to_member(league_rank), update_ptr);
  }

 private:
  KOKKOS_INLINE_FUNCTION Member
  league_rank_to_member(const int league_rank) const {
    std::byte* team_scratch_buffer =
        m_league_scratch_buffer + league_rank * (m_L0_size + m_L1_size);
    using scratch_memory_space = typename Member::scratch_memory_space;
    return Member(
        league_rank, m_league_size, m_vector_length,
        scratch_memory_space(team_scratch_buffer, m_L0_size,
                             team_scratch_buffer + m_L0_size, m_L1_size));
  }

  const Functor m_functor;
  int m_league_size;
  int m_vector_length;
  std::byte* m_league_scratch_buffer;
  size_t m_L0_size;
  size_t m_L1_size;
};
}  // namespace Kokkos::Experimental::Impl

template <class CombinedFunctorReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<CombinedFunctorReducerType,
                                   Kokkos::TeamPolicy<Properties...>,
                                   Kokkos::Experimental::NextSilicon> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::NextSilicon,
                                       Properties...>;
  using Member      = typename Policy::member_type;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using value_type   = typename ReducerType::value_type;
  using pointer_type = typename ReducerType::pointer_type;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  pointer_type m_result_ptr;

 public:
  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 const Policy& arg_policy, const ViewType& arg_result_view)
      : m_functor_reducer(arg_functor_reducer),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()) {}

  void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    const int league_size = m_policy.league_size();
    const int team_size   = m_policy.team_size();

    auto const& functor = m_functor_reducer.get_functor();
    auto const& reducer = m_functor_reducer.get_reducer();

    const auto L0_size =
        m_policy.scratch_size(0, team_size) +
        FunctorTeamShmemSize<FunctorType>::value(functor, team_size);

    const auto L1_size = m_policy.scratch_size(1, team_size);

    nextsilicon_check_team_scratch_size("Kokkos::parallel_reduce<NextSilicon>",
                                        0, L0_size);
    nextsilicon_check_team_scratch_size("Kokkos::parallel_reduce<NextSilicon>",
                                        1, L1_size);

    // Make sure there's a scratch allocation big enough for all our teams
    // TODO: support alignment of scratch memory?
    auto internal_instance = m_policy.space().impl_internal_space_instance();
    std::byte* league_scratch_buffer =
        internal_instance->resize_league_scratch_buffer(league_size *
                                                        (L0_size + L1_size));

    Experimental::Impl::NextSiliconParallelReduceTeamPolicyFunctor<Member,
                                                                   FunctorType>
        wrapped_functor(functor, league_size, m_policy.vector_length(),
                        league_scratch_buffer, L0_size, L1_size);

    CombinedFunctorReducer combinedWrappedFunctorReducer(wrapped_functor,
                                                         reducer);

    auto policy =
        RangePolicy<Properties...>(m_policy.space(), 0,
                                   league_size);  // team size always 1

    NextSiliconParallelReduceImpl<decltype(combinedWrappedFunctorReducer),
                                  Properties...>{combinedWrappedFunctorReducer,
                                                 policy, m_result_ptr}
        .execute();
  }
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
// FIXME_NEXTSILICON: single-thread implementation
template <typename iType, class Lambda, typename ReducerType>
  requires(Kokkos::is_reducer<ReducerType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>,
      ReducerType, value_type>::Reducer;

  // team size is 1
  WrappedReducer wrappedReducer(reducer);
  value_type val;
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  reducer.reference() = val;
}
template <typename iType, class Lambda, typename ValueType>
  requires(!Kokkos::is_reducer<ValueType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  // team size is 1
  ValueType val;
  WrappedReducer wrappedReducer(lambda);
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  result = val;
}

// Hierarchical Parallelism -> Thread vector level implementation
// FIXME_NEXTSILICON: single-vector implementation
template <typename iType, class Lambda, typename ReducerType>
  requires(Kokkos::is_reducer<ReducerType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>,
      ReducerType, value_type>::Reducer;

  // team size is 1
  WrappedReducer wrappedReducer(reducer);
  value_type val;
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  reducer.reference() = val;
}
template <typename iType, class Lambda, typename ValueType>
  requires(!Kokkos::is_reducer<ValueType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  // team size is 1
  ValueType val;
  WrappedReducer wrappedReducer(lambda);
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  result = val;
}

// Hierarchical Parallelism -> Team vector level implementation
// FIXME_NEXTSILICON: single-vector implementation
template <typename iType, class Lambda, typename ReducerType>
  requires(Kokkos::is_reducer<ReducerType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>,
      ReducerType, value_type>::Reducer;

  // team size is 1
  WrappedReducer wrappedReducer(reducer);
  value_type val;
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  reducer.reference() = val;
}
template <typename iType, class Lambda, typename ValueType>
  requires(!Kokkos::is_reducer<ValueType>::value)
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  // team size is 1
  ValueType val;
  WrappedReducer wrappedReducer(lambda);
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val);
  wrappedReducer.final(&val);
  result = val;
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP */
