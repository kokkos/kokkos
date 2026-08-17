// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_MDRANGE_HPP
#define KOKKOS_NEXTSILICON_MDRANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_MDRangePolicy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ParallelReduce.hpp>
#include <Kokkos_Parallel_Reduce.hpp>
#include <Kokkos_Parallel.hpp>
#include <impl/Kokkos_CheckedIntegerOps.hpp>
#include <mutex>
#include <type_traits>
#include <utility>

namespace Kokkos::Experimental::Impl {

template <int N>
using NextSiliconMDRangeBegin =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_lower);
template <int N>
using NextSiliconMDRangeEnd =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_upper);
template <int N>
using NextSiliconMDRangeTile =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_tile);

template <typename WorkTag, typename Direction, typename Functor, int Dim>
class NextSiliconParallelMDRangePolicyFunctor {
 public:
  using IndexType = typename NextSiliconMDRangeEnd<Dim>::value_type;

  NextSiliconParallelMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<Dim> const& begin,
      NextSiliconMDRangeEnd<Dim> const& end)
      : functor_(functor), begin_(begin) {
    for (int i = 0; i < Dim; ++i) {
      ext_[i] = end[i] - begin[i];
    }
  }

  template <typename FlatIndexType>
  void operator()(const FlatIndexType i) const {
    IndexType x[Dim];
    map(i, x);
    call_functor(std::make_index_sequence<Dim>() /* 0, 1, ..., Dim-1 */, x);
  }

  template <typename FlatIndexType, typename ReducerValueType>
  KOKKOS_INLINE_FUNCTION void operator()(const FlatIndexType i,
                                         ReducerValueType& update) const {
    IndexType x[Dim];
    map(i, x);
    call_functor_reduce(std::make_index_sequence<Dim>() /* 0, 1, ..., Dim-1 */,
                        x, update);
  }

 private:
  // call the functor on a Dim-dimensional index
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION void call_functor(std::index_sequence<I...>,
                                           const IndexType (&x)[Dim]) const {
    if constexpr (std::is_void_v<WorkTag>) {
      functor_(x[I]...);
    } else {
      functor_(WorkTag(), x[I]...);
    }
  }

  template <typename ReducerValueType, size_t... I>
  KOKKOS_INLINE_FUNCTION void call_functor_reduce(
      std::index_sequence<I...>, const IndexType (&x)[Dim],
      ReducerValueType& update) const {
    if constexpr (std::is_void_v<WorkTag>) {
      functor_(x[I]..., update);
    } else {
      functor_(WorkTag(), x[I]..., update);
    }
  }

  template <typename FlatIndexType>
  KOKKOS_INLINE_FUNCTION void map(FlatIndexType i, IndexType (&x)[Dim]) const {
    if constexpr (std::is_same_v<Direction, std::integral_constant<
                                                Iterate, Iterate::Left>>) {
      // like
      // for (auto i2 = begin2; i2 < end2; ++i2) {
      //   for (auto i1 = begin1; i1 < end1; ++i1) {
      //     for (auto i0 = begin0; i0 < end0; ++i0) {
      //       ...
      //     }
      //   }
      // }

#pragma unroll
      for (int j = 0; j < Dim; ++j) {
        std::lldiv_t r;
        r.quot = i / ext_[j];
        r.rem  = i % ext_[j];
        x[j]   = begin_[j] + r.rem;
        i      = r.quot;
      }
    } else if constexpr (std::is_same_v<
                             Direction,
                             std::integral_constant<Iterate, Iterate::Right>>) {
      // like
      // for (auto i0 = begin0; i0 < end0; ++i0) {
      //   for (auto i1 = begin1; i1 < end1; ++i1) {
      //     for (auto i2 = begin2; i2 < end2; ++i2) {
      //       ...
      //     }
      //   }
      // }

#pragma unroll
      for (int j = Dim - 1; j >= 0; --j) {
        std::lldiv_t r;
        r.quot = i / ext_[j];
        r.rem  = i % ext_[j];
        x[j]   = begin_[j] + r.rem;
        i      = r.quot;
      }
    } else {
      static_assert(std::is_void_v<Functor>,
                    "Expected left or right MDRange iteration");
    }
  }

  Functor functor_;
  NextSiliconMDRangeBegin<Dim> begin_;
  NextSiliconMDRangeEnd<Dim> ext_;  // end - begin
};

using FlatIndexType = uint64_t;

template <int Dim>
FlatIndexType getFlatRange(NextSiliconMDRangeBegin<Dim> const& begin,
                           NextSiliconMDRangeEnd<Dim> const& end) {
  FlatIndexType flat = end[0] - begin[0];
  for (int i = 1; i < Dim; ++i) {
    const FlatIndexType factor = end[i] - begin[i];
    flat = Kokkos::Impl::multiply_overflow_abort(flat, factor);
  }

  return flat;
}
}  // namespace Kokkos::Experimental::Impl

namespace Kokkos::Impl {

template <class Functor, class... Traits>
class ParallelFor<Functor, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::NextSilicon> {
  using Policy  = MDRangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  Functor m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {
    static_assert(0 < Policy::rank && Policy::rank < 7);
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);
  }

  void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    constexpr int rank = Policy::rank;
    for (int i = 0; i < rank; ++i) {
      if (m_policy.m_lower[i] >= m_policy.m_upper[i]) {
        return;
      }
    }

    // FIXME_NEXTSILICON: throw away requested tiling
    using Direction = std::integral_constant<Iterate, Policy::inner_direction>;

    auto total_range = Kokkos::Experimental::Impl::getFlatRange<rank>(
        m_policy.m_lower, m_policy.m_upper);
    auto wrapped_functor =
        Kokkos::Experimental::Impl::NextSiliconParallelMDRangePolicyFunctor<
            WorkTag, Direction, Functor, rank>(m_functor, m_policy.m_lower,
                                               m_policy.m_upper);

    Kokkos::parallel_for(total_range, wrapped_functor);
  }
};

template <class CombinedFunctorReducerType, class... Traits>
class ParallelReduce<CombinedFunctorReducerType, MDRangePolicy<Traits...>,
                     Experimental::NextSilicon> {
  using MDRangePolicy = MDRangePolicy<Traits...>;
  using ReducerType   = typename CombinedFunctorReducerType::reducer_type;
  using FunctorType   = typename CombinedFunctorReducerType::functor_type;
  using pointer_type  = typename ReducerType::pointer_type;
  using ValueType     = typename ReducerType::value_type;
  using WorkTag       = typename MDRangePolicy::work_tag;

  CombinedFunctorReducerType m_functor_reducer;
  MDRangePolicy m_policy;
  pointer_type m_result_ptr;

 public:
  template <class ViewType>
  ParallelReduce(CombinedFunctorReducerType const& arg_functor_reducer,
                 MDRangePolicy const& arg_policy,
                 ViewType const& arg_result_view)
      : m_functor_reducer(arg_functor_reducer),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()) {
    static_assert(0 < MDRangePolicy::rank && MDRangePolicy::rank < 7);
    static_assert(MDRangePolicy::inner_direction == Iterate::Left ||
                  MDRangePolicy::inner_direction == Iterate::Right);
  }

  void execute() const {
    constexpr int rank  = MDRangePolicy::rank;
    auto const& functor = m_functor_reducer.get_functor();
    auto const& reducer = m_functor_reducer.get_reducer();

    using Direction =
        std::integral_constant<Iterate, MDRangePolicy::inner_direction>;
    using WrappedFunctorType =
        Experimental::Impl::NextSiliconParallelMDRangePolicyFunctor<
            WorkTag, Direction, FunctorType, rank>;
    using CombinedWrappedFunctorReducerType =
        CombinedFunctorReducer<WrappedFunctorType, ReducerType>;
    using RangePolicyType = RangePolicy<Experimental::NextSilicon>;

    // FIXME_NEXTSILICON: throw away requested tiling
    auto total_range = Experimental::Impl::getFlatRange<rank>(m_policy.m_lower,
                                                              m_policy.m_upper);
    auto wrapped_functor =
        WrappedFunctorType(functor, m_policy.m_lower, m_policy.m_upper);

    auto policy = RangePolicyType(0, total_range);

    NextSiliconParallelReduceImpl<CombinedWrappedFunctorReducerType> impl(
        CombinedWrappedFunctorReducerType(wrapped_functor, reducer), policy,
        m_result_ptr);
    impl.execute();
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_NEXTSILICON_MDRANGE_HPP
