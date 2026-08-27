// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>

namespace Kokkos {

// FIXME_NEXTSILICON: single-thread implementation. implement multiple threads.
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_scan(const Impl::ThreadVectorRangeBoundariesStruct<
                  iType, Impl::NextSiliconTeamMember>& loop_boundaries,
              const Lambda& lambda, const ReducerType& reducer) {
  using value_type = typename ReducerType::value_type;
  using functor_analysis_type =
      typename Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN, void,
                                     ReducerType, value_type>;
  using WrappedReducer = typename functor_analysis_type::Reducer;
  // vector size always 1
  WrappedReducer wrappedReducer(reducer);
  value_type val;
  wrappedReducer.init(&val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
    lambda(i, val, /*final*/ true);

  wrappedReducer.final(&val);
  reducer.reference() = val;
}

template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, Lambda,
      void>::value_type;
  value_type dummy;
  parallel_scan(loop_boundaries, lambda, Kokkos::Sum<value_type>(dummy));
}

template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& return_val) {
  using lambda_value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, Lambda,
      ValueType>::value_type;
  static_assert(std::is_same<lambda_value_type, ValueType>::value,
                "Non-matching value types of lambda and return type");

  ValueType accum;
  parallel_scan(loop_boundaries, lambda, Kokkos::Sum<ValueType>(accum));

  return_val = accum;
}

/** \brief  Inter-thread parallel exclusive prefix sum. Executes
 * lambda(iType i, ValueType & val, bool final) for each i=0..N-1.
 * FIXME_NEXTSILICON: Implement multiple threads.
 */
template <typename iType, class FunctorType, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_bounds,
    const FunctorType& lambda, ValueType& return_val) {
  // Extract ValueType from the Closure
  using closure_value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, FunctorType,
      void>::value_type;
  static_assert(std::is_same_v<closure_value_type, ValueType>,
                "Non-matching value types of closure and return type");

  // team size always 1
  auto scan_val = ValueType{};

  for (iType i = loop_bounds.start; i < loop_bounds.end; i++)
    lambda(i, scan_val, false);

  // 'scan_val' output is the exclusive prefix sum
  scan_val = loop_bounds.team.team_scan(scan_val);

  for (iType i = loop_bounds.start; i < loop_bounds.end; i++)
    lambda(i, scan_val, true);

  loop_bounds.team.team_broadcast(scan_val, loop_bounds.team.team_size() - 1);
  return_val = scan_val;
}

template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_bounds,
    const FunctorType& lambda) {
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, FunctorType,
      void>::value_type;

  value_type scan_val;
  parallel_scan(loop_bounds, lambda, scan_val);
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP */
