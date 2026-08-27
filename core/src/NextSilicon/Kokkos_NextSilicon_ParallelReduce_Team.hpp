// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce.hpp>
#include <mutex>

template <typename Functor>
class NextSiliconParallelReduceTeamPolicyFunctor {
  const Functor functor_;
  int leagueSize_;
  void* globalScratchBuffer_;
  size_t L0size_;
  size_t L1size_;

  Kokkos::Impl::NextSiliconTeamMember worker_id_to_member(
      int workerId) const noexcept {
    int league_rank = workerId;  // team size is 1

    char* teamScratchBuffer = static_cast<char*>(globalScratchBuffer_) +
                              league_rank * (L0size_ + L1size_);

    return Kokkos::Impl::NextSiliconTeamMember(
        league_rank, leagueSize_,
        Kokkos::Impl::NextSiliconTeamMember::scratch_memory_space(
            teamScratchBuffer, L0size_, teamScratchBuffer + L0size_, L1size_));
  }

 public:
  NextSiliconParallelReduceTeamPolicyFunctor(Functor const& functor,
                                             int leagueSize,
                                             void* globalScratchBuffer,
                                             const size_t L0size,
                                             const size_t L1size)
      : functor_(functor),
        leagueSize_(leagueSize),
        globalScratchBuffer_(globalScratchBuffer),
        L0size_(L0size),
        L1size_(L1size) {}

  template <typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(int workerId,
                                         ReducerValueType& update) const {
    functor_(worker_id_to_member(workerId), update);
  }

  template <typename Tag, typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(Tag, int workerId,
                                         ReducerValueType& update) const {
    functor_(Tag{}, worker_id_to_member(workerId), update);
  }

  template <typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(int workerId,
                                         ReducerValueType* update_ptr) const {
    functor_(worker_id_to_member(workerId), update_ptr);
  }

  template <typename Tag, typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(Tag, int workerId,
                                         ReducerValueType* update_ptr) const {
    functor_(Tag{}, worker_id_to_member(workerId), update_ptr);
  }
};

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

    int leagueSize = m_policy.league_size();

    auto const& functor = m_functor_reducer.get_functor();
    auto const& reducer = m_functor_reducer.get_reducer();

    const auto L0size =
        m_policy.scratch_size(0, 1 /* team size */) +
        FunctorTeamShmemSize<FunctorType>::value(functor, 1 /* team size */);

    const auto L1size = m_policy.scratch_size(1, 1 /* team size */);

    // Make sure there's a scratch allocation big enough for all our teams
    // TODO: support alignment of scratch memory?
    auto internal_instance = m_policy.space().impl_internal_space_instance();
    void* scratchData      = internal_instance->resize_league_scratch_buffer(
        leagueSize * (L0size + L1size));

    auto wrapped_functor = NextSiliconParallelReduceTeamPolicyFunctor(
        functor, leagueSize, scratchData, L0size, L1size);

    CombinedFunctorReducer /*<decltype(wrapped_functor), ReducerType>*/
        combinedWrappedFunctorReducer(wrapped_functor, reducer);

    auto policy = RangePolicy<Properties...>(m_policy.space(), 0,
                                             leagueSize);  // team size always 1

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

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);
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

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);

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

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);

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
