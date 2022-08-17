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

  using Pointer   = typename ReducerType::pointer_type;
  using ValueType = typename ReducerType::value_type;

  Functor m_functor;
  Policy m_policy;
  ReducerType m_reducer;
  Pointer m_result_ptr;

 public:
  template <class ViewType>
  ParallelReduce(const Functor& functor, const Policy& policy,
                 const ReducerType& reducer, const ViewType& result)
      : m_functor(functor),
        m_policy(policy),
        m_reducer(reducer),
        m_result_ptr(result.data()) {}

  void execute() {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    if (end <= begin) {
      return;
    }

    ValueType val;
    m_reducer.init(&val);

    Kokkos::Experimental::Impl::OpenACCParallelReduceHelper(
        Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>(m_functor),
        std::conditional_t<
            std::is_same_v<typename ReducerType::functor_type, Functor>,
            Sum<ValueType>, typename ReducerType::functor_type>(val),
        m_policy);

    m_reducer.final(&val);
    *m_result_ptr = val;
  }
};

namespace Kokkos::Experimental::Impl {

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceSum(IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(+ : val) copyin(functor) async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceProd(IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(* : val) copyin(functor) async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMin(IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(min                    \
                                                : val) copyin(functor) \
    async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceMax(IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_argc) {
#pragma acc parallel loop gang vector reduction(max                    \
                                                : val) copyin(functor) \
    async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLAnd(IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(&& : val) copyin(functor) async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceLOr(IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(||                     \
                                                : val) copyin(functor) \
    async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBAnd(IndexType begin, IndexType end, ValueType& val,
                               Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(&                     \
                                                : val) copyin(functor) \
    async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

template <class IndexType, class ValueType, class Functor>
void OpenACCParallelReduceBOr(IndexType begin, IndexType end, ValueType& val,
                              Functor const& functor, int async_arg) {
#pragma acc parallel loop gang vector reduction(|                      \
                                                : val) copyin(functor) \
    async(async_arg)
  for (auto i = begin; i < end; i++) {
    functor(i, val);
  }
}

}  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(REDUCER)                    \
  template <class Functor, class Scalar, class Space, class... Traits>         \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceHelper<              \
      Functor, Kokkos::REDUCER<Scalar, Space>, Kokkos::RangePolicy<Traits...>, \
      true> {                                                                  \
    using Policy    = RangePolicy<Traits...>;                                  \
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
      int const async_arg = policy.space().acc_async_queue();                  \
                                                                               \
      OpenACCParallelReduce##REDUCER(begin, end, val, functor, async_arg);     \
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
