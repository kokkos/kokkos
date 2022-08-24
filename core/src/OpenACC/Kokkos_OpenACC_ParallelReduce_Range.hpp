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

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_RANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_RANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <OpenACC/Kokkos_OpenACC_ScheduleType.hpp>
#include <Kokkos_Parallel.hpp>
#include <type_traits>

namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct OpenACCParallelReduceHelper {
  OpenACCParallelReduceHelper(Functor const&, Reducer const&, Policy const&) {
    static_assert(std::is_void_v<Functor>, "not implemented");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class Functor, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<Functor, Kokkos::RangePolicy<Traits...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
  using Policy = RangePolicy<Traits...>;

  using Analysis = FunctorAnalysis<
      FunctorPatternInterface::REDUCE, Policy,
      std::conditional_t<std::is_same_v<InvalidType, ReducerType>, Functor,
                         ReducerType>>;

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

  void execute() {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    if (end <= begin) {
      return;
    }

    ValueType val;
    typename Analysis::Reducer final_reducer(&m_functor);
    final_reducer.init(&val);

    Kokkos::Experimental::Impl::OpenACCParallelReduceHelper(
        Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>(m_functor),
        std::conditional_t<is_reducer_v<ReducerType>, ReducerType,
                           Sum<ValueType>>(val),
        m_policy);

    *m_result_ptr = val;
  }
};

namespace Kokkos::Experimental::Impl {

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceSum(Schedule<Static>, int chunk_size, IndexType begin,
                              IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(+:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(+:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceSum(Schedule<Dynamic>, int chunk_size,
                              IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(+:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(+:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceProd(Schedule<Static>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(*:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(*:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceProd(Schedule<Dynamic>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(*:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(*:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMin(Schedule<Static>, int chunk_size, IndexType begin,
                              IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(min:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(min:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMin(Schedule<Dynamic>, int chunk_size,
                              IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(min:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(min:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMax(Schedule<Static>, int chunk_size, IndexType begin,
                              IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(max:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(max:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMax(Schedule<Dynamic>, int chunk_size,
                              IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(max:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(max:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLAnd(Schedule<Static>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLAnd(Schedule<Dynamic>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLOr(Schedule<Static>, int chunk_size, IndexType begin,
                              IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(||:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(||:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLOr(Schedule<Dynamic>, int chunk_size,
                              IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(||:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(||:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBAnd(Schedule<Static>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBAnd(Schedule<Dynamic>, int chunk_size,
                               IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(&&:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBOr(Schedule<Static>, int chunk_size, IndexType begin,
                              IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(|:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang(static:*) vector reduction(|:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBOr(Schedule<Dynamic>, int chunk_size,
                              IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
  if (chunk_size >= 1) {
// clang-format off
#pragma acc parallel loop gang(static:chunk_size) vector reduction(|:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  } else {
// clang-format off
#pragma acc parallel loop gang vector reduction(|:val) copyin(functor) async(async_arg)
    // clang-format on
    for (auto i = begin; i < end; i++) {
      functor(i, val);
    }
  }
}

}  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(REDUCER)                    \
  template <class Functor, class Scalar, class Space, class... Traits>         \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceHelper<              \
      Functor, Kokkos::REDUCER<Scalar, Space>, Kokkos::RangePolicy<Traits...>, \
      true> {                                                                  \
    using Policy = RangePolicy<Traits...>;                                     \
    using ScheduleType =                                                       \
        Kokkos::Experimental::Impl::OpenACCScheduleType<Policy>;               \
    using Reducer   = REDUCER<Scalar, Space>;                                  \
    using ValueType = typename Reducer::value_type;                            \
                                                                               \
    OpenACCParallelReduceHelper(Functor const& functor,                        \
                                Reducer const& reducer,                        \
                                Policy const& policy) {                        \
      auto const begin = policy.begin();                                       \
      auto const end   = policy.end();                                         \
                                                                               \
      if (end <= begin) {                                                      \
        return;                                                                \
      }                                                                        \
                                                                               \
      ValueType val;                                                           \
      reducer.init(val);                                                       \
                                                                               \
      int const async_arg  = policy.space().acc_async_queue();                 \
      int const chunk_size = policy.chunk_size();                              \
                                                                               \
      OpenACCParallelReduce##REDUCER(ScheduleType(), chunk_size, begin, end,   \
                                     val, functor, async_arg);                 \
                                                                               \
      reducer.reference() = val;                                               \
    }                                                                          \
  }

KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(Sum);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(Prod);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(Min);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(Max);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(LAnd);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(LOr);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(BAnd);
KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(BOr);

#undef KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER

#endif
