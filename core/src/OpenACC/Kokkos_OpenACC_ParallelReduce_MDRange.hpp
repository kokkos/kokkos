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

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_MDRANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_MDRANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_Macros.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <OpenACC/Kokkos_OpenACC_MDRangePolicy.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct OpenACCParallelReduceMDRangeHelper {
  OpenACCParallelReduceMDRangeHelper(Functor const&, Reducer const&,
                                     Policy const&) {
    static_assert(!Kokkos::Impl::always_true<Functor>::value,
                  "not implemented");
  }
};
}  // namespace Kokkos::Experimental::Impl

template <class Functor, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<Functor, Kokkos::MDRangePolicy<Traits...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
  using Policy = MDRangePolicy<Traits...>;

  using ReducerConditional =
      if_c<std::is_same_v<InvalidType, ReducerType>, Functor, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, ReducerTypeFwd>;

  using Pointer   = typename Analysis::pointer_type;
  using ValueType = typename Analysis::value_type;

  Functor m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  Pointer m_result_ptr;

 public:
  ParallelReduce(Functor const& functor, Policy const& policy,
                 ReducerType const& reducer)
      : m_functor(functor),
        m_policy(policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}

  template <class ViewType>
  ParallelReduce(
      const Functor& functor, const Policy& policy, const ViewType& result,
      std::enable_if_t<Kokkos::is_view<ViewType>::value, void*> = nullptr)
      : m_functor(functor),
        m_policy(policy),
        m_reducer(InvalidType()),
        m_result_ptr(result.data()) {}

  void execute() const {
    static_assert(1 < Policy::rank && Policy::rank < 7);
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);
    constexpr int rank = Policy::rank;
    for (int i = 0; i < rank; ++i) {
      if (m_policy.m_lower[i] >= m_policy.m_upper[i]) {
        return;
      }
    }

    ValueType val;
    typename Analysis::Reducer final_reducer(
        &ReducerConditional::select(m_functor, m_reducer));
    final_reducer.init(&val);

    Kokkos::Experimental::Impl::OpenACCParallelReduceMDRangeHelper(
        Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>(m_functor),
        std::conditional_t<is_reducer_v<ReducerType>, ReducerType,
                           Sum<ValueType>>(val),
        m_policy);

    *m_result_ptr = val;
  }
};

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_DISPATCH_ITERATE(REDUCER,         \
                                                             OPERATOR)        \
  namespace Kokkos::Experimental::Impl {                                      \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateLeft, ValueType& aval,    \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<2> const& begin,    \
                                      OpenACCMDRangeEnd<2> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(2) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i1 = begin1; i1 < end1; ++i1) {                                 \
      for (auto i0 = begin0; i0 < end0; ++i0) {                               \
        functor(i0, i1, val);                                                 \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateRight, ValueType& aval,   \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<2> const& begin,    \
                                      OpenACCMDRangeEnd<2> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(2) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i0 = begin0; i0 < end0; ++i0) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        functor(i0, i1, val);                                                 \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateLeft, ValueType& aval,    \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<3> const& begin,    \
                                      OpenACCMDRangeEnd<3> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    /* clang-format off */                                                  \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(3) reduction( \
        OPERATOR                                                            \
        : val) copyin(functor) async(async_arg)) \
    /* clang-format on */                                                     \
    for (auto i2 = begin2; i2 < end2; ++i2) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        for (auto i0 = begin0; i0 < end0; ++i0) {                             \
          functor(i0, i1, i2, val);                                           \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateRight, ValueType& aval,   \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<3> const& begin,    \
                                      OpenACCMDRangeEnd<3> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    /* clang-format off */                                                  \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(3) reduction( \
        OPERATOR                                                            \
        : val) copyin(functor) async(async_arg)) \
    /* clang-format on */                                                     \
    for (auto i0 = begin0; i0 < end0; ++i0) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        for (auto i2 = begin2; i2 < end2; ++i2) {                             \
          functor(i0, i1, i2, val);                                           \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateLeft, ValueType& aval,    \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<4> const& begin,    \
                                      OpenACCMDRangeEnd<4> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(4) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i3 = begin3; i3 < end3; ++i3) {                                 \
      for (auto i2 = begin2; i2 < end2; ++i2) {                               \
        for (auto i1 = begin1; i1 < end1; ++i1) {                             \
          for (auto i0 = begin0; i0 < end0; ++i0) {                           \
            functor(i0, i1, i2, i3, val);                                     \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateRight, ValueType& aval,   \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<4> const& begin,    \
                                      OpenACCMDRangeEnd<4> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(4) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i0 = begin0; i0 < end0; ++i0) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        for (auto i2 = begin2; i2 < end2; ++i2) {                             \
          for (auto i3 = begin3; i3 < end3; ++i3) {                           \
            functor(i0, i1, i2, i3, val);                                     \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateLeft, ValueType& aval,    \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<5> const& begin,    \
                                      OpenACCMDRangeEnd<5> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin4 = begin[4];                                                    \
    int end4   = end[4];                                                      \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(5) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i4 = begin4; i4 < end4; ++i4) {                                 \
      for (auto i3 = begin3; i3 < end3; ++i3) {                               \
        for (auto i2 = begin2; i2 < end2; ++i2) {                             \
          for (auto i1 = begin1; i1 < end1; ++i1) {                           \
            for (auto i0 = begin0; i0 < end0; ++i0) {                         \
              functor(i0, i1, i2, i3, i4, val);                               \
            }                                                                 \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateRight, ValueType& aval,   \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<5> const& begin,    \
                                      OpenACCMDRangeEnd<5> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    int begin4 = begin[4];                                                    \
    int end4   = end[4];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(5) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i0 = begin0; i0 < end0; ++i0) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        for (auto i2 = begin2; i2 < end2; ++i2) {                             \
          for (auto i3 = begin3; i3 < end3; ++i3) {                           \
            for (auto i4 = begin4; i4 < end4; ++i4) {                         \
              functor(i0, i1, i2, i3, i4, val);                               \
            }                                                                 \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateLeft, ValueType& aval,    \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<6> const& begin,    \
                                      OpenACCMDRangeEnd<6> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin5 = begin[5];                                                    \
    int end5   = end[5];                                                      \
    int begin4 = begin[4];                                                    \
    int end4   = end[4];                                                      \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(6) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i5 = begin5; i5 < end5; ++i5) {                                 \
      for (auto i4 = begin4; i4 < end4; ++i4) {                               \
        for (auto i3 = begin3; i3 < end3; ++i3) {                             \
          for (auto i2 = begin2; i2 < end2; ++i2) {                           \
            for (auto i1 = begin1; i1 < end1; ++i1) {                         \
              for (auto i0 = begin0; i0 < end0; ++i0) {                       \
                functor(i0, i1, i2, i3, i4, i5, val);                         \
              }                                                               \
            }                                                                 \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
                                                                              \
  template <class ValueType, class Functor>                                   \
  void OpenACCParallelReduce##REDUCER(OpenACCIterateRight, ValueType& aval,   \
                                      Functor const& afunctor,                \
                                      OpenACCMDRangeBegin<6> const& begin,    \
                                      OpenACCMDRangeEnd<6> const& end,        \
                                      int async_arg) {                        \
    auto val = aval;                                                          \
    auto const functor(afunctor);                                             \
    int begin0 = begin[0];                                                    \
    int end0   = end[0];                                                      \
    int begin1 = begin[1];                                                    \
    int end1   = end[1];                                                      \
    int begin2 = begin[2];                                                    \
    int end2   = end[2];                                                      \
    int begin3 = begin[3];                                                    \
    int end3   = end[3];                                                      \
    int begin4 = begin[4];                                                    \
    int end4   = end[4];                                                      \
    int begin5 = begin[5];                                                    \
    int end5   = end[5];                                                      \
    /* clang-format off */ \
    KOKKOS_IMPL_ACC_PRAGMA(parallel loop gang vector collapse(6) reduction(OPERATOR:val) copyin(functor) async(async_arg))                                                  \
    /* clang-format on */                                                     \
    for (auto i0 = begin0; i0 < end0; ++i0) {                                 \
      for (auto i1 = begin1; i1 < end1; ++i1) {                               \
        for (auto i2 = begin2; i2 < end2; ++i2) {                             \
          for (auto i3 = begin3; i3 < end3; ++i3) {                           \
            for (auto i4 = begin4; i4 < end4; ++i4) {                         \
              for (auto i5 = begin5; i5 < end5; ++i5) {                       \
                functor(i0, i1, i2, i3, i4, i5, val);                         \
              }                                                               \
            }                                                                 \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    aval = val;                                                               \
  }                                                                           \
  }  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(REDUCER, OPERATOR) \
  KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_DISPATCH_ITERATE(REDUCER, OPERATOR)     \
  template <class Functor, class Scalar, class Space, class... Traits>        \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceMDRangeHelper<      \
      Functor, Kokkos::REDUCER<Scalar, Space>,                                \
      Kokkos::MDRangePolicy<Traits...>, true> {                               \
    using Policy    = MDRangePolicy<Traits...>;                               \
    using Reducer   = REDUCER<Scalar, Space>;                                 \
    using ValueType = typename Reducer::value_type;                           \
                                                                              \
    OpenACCParallelReduceMDRangeHelper(Functor const& functor,                \
                                       Reducer const& reducer,                \
                                       Policy const& policy) {                \
      ValueType val;                                                          \
      reducer.init(val);                                                      \
                                                                              \
      int const async_arg = policy.space().acc_async_queue();                 \
                                                                              \
      OpenACCParallelReduce##REDUCER(                                         \
          std::integral_constant<Iterate, Policy::inner_direction>(), val,    \
          functor, policy.m_lower, policy.m_upper, async_arg);                \
                                                                              \
      reducer.reference() = val;                                              \
    }                                                                         \
  }

KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(Sum, +);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(Prod, *);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(Min, min);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(Max, max);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(LAnd, &&);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(LOr, ||);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(BAnd, &);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER(BOr, |);

#undef KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_MDRANGE_HELPER
#undef KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_DISPATCH_ITERATE

#endif
