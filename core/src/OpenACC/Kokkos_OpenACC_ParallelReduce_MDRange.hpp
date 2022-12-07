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

#ifndef KOKKOS_OPENACC_PARALLEL_REDUCE_MDRANGE_HPP
#define KOKKOS_OPENACC_PARALLEL_REDUCE_MDRANGE_HPP

#include <OpenACC/Kokkos_OpenACC.hpp>
#include <OpenACC/Kokkos_OpenACC_FunctorAdapter.hpp>
#include <OpenACC/Kokkos_OpenACC_MDRangePolicy.hpp>
#include <Kokkos_Parallel.hpp>
#include <type_traits>

namespace Kokkos::Experimental::Impl {

template <int N>
using OpenACCMDRangeBegin = decltype(MDRangePolicy<OpenACC, Rank<N>>::m_lower);
template <int N>
using OpenACCMDRangeEnd = decltype(MDRangePolicy<OpenACC, Rank<N>>::m_upper);

// primary template: catch-all unimplemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct OpenACCParallelReduceMDRangeHelper {
  OpenACCParallelReduceMDRangeHelper(Functor const&, Reducer const&,
                                     Policy const&) {
    static_assert(
        !Kokkos::Impl::always_true<Functor>::value,
        "Kokkos Error: unimplemented reducer type for the OpenACC backend");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class Functor, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<Functor, Kokkos::MDRangePolicy<Traits...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
  using Policy = MDRangePolicy<Traits...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, Functor>;

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
  ParallelReduce(const Functor& functor, const Policy& policy,
                 const ViewType& result,
                 std::enable_if_t<Kokkos::is_view_v<ViewType>>* = nullptr)
      : m_functor(functor),
        m_policy(policy),
        m_reducer(InvalidType()),
        m_result_ptr(result.data()) {}

  void execute() {
    static_assert(Policy::rank < 7 && Policy::rank > 1,
                  "OpenACC Backend MDRangePolicy Error: Unsupported rank...");
    ValueType val;
    typename Analysis::Reducer final_reducer(&m_functor);
    final_reducer.init(&val);

    Kokkos::Experimental::Impl::OpenACCParallelReduceMDRangeHelper(
        Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>(m_functor),
        std::conditional_t<is_reducer_v<ReducerType>, ReducerType,
                           Sum<ValueType>>(val),
        m_policy);

    *m_result_ptr = val;
  }
};

namespace Kokkos::Experimental::Impl {

template <class ValueType, class Functor>
void OpenACCParallelReduceSum(OpenACCMDRangeBegin<2> const& begin,
                              OpenACCMDRangeEnd<2> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(+ : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceSum(OpenACCMDRangeBegin<3> const& begin,
                              OpenACCMDRangeEnd<3> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(+ : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceSum(OpenACCMDRangeBegin<4> const& begin,
                              OpenACCMDRangeEnd<4> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(+ : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceSum(OpenACCMDRangeBegin<5> const& begin,
                              OpenACCMDRangeEnd<5> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(+ : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceSum(OpenACCMDRangeBegin<6> const& begin,
                              OpenACCMDRangeEnd<6> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(+ : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceProd(OpenACCMDRangeBegin<2> const& begin,
                               OpenACCMDRangeEnd<2> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(* : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceProd(OpenACCMDRangeBegin<3> const& begin,
                               OpenACCMDRangeEnd<3> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(* : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceProd(OpenACCMDRangeBegin<4> const& begin,
                               OpenACCMDRangeEnd<4> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(* : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceProd(OpenACCMDRangeBegin<5> const& begin,
                               OpenACCMDRangeEnd<5> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(* : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceProd(OpenACCMDRangeBegin<6> const& begin,
                               OpenACCMDRangeEnd<6> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(* : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMin(OpenACCMDRangeBegin<2> const& begin,
                              OpenACCMDRangeEnd<2> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(min : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMin(OpenACCMDRangeBegin<3> const& begin,
                              OpenACCMDRangeEnd<3> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(min : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMin(OpenACCMDRangeBegin<4> const& begin,
                              OpenACCMDRangeEnd<4> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(min : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMin(OpenACCMDRangeBegin<5> const& begin,
                              OpenACCMDRangeEnd<5> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(min : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMin(OpenACCMDRangeBegin<6> const& begin,
                              OpenACCMDRangeEnd<6> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(min : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMax(OpenACCMDRangeBegin<2> const& begin,
                              OpenACCMDRangeEnd<2> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(max : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMax(OpenACCMDRangeBegin<3> const& begin,
                              OpenACCMDRangeEnd<3> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(max : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMax(OpenACCMDRangeBegin<4> const& begin,
                              OpenACCMDRangeEnd<4> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(max : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMax(OpenACCMDRangeBegin<5> const& begin,
                              OpenACCMDRangeEnd<5> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(max : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceMax(OpenACCMDRangeBegin<6> const& begin,
                              OpenACCMDRangeEnd<6> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(max : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLAnd(OpenACCMDRangeBegin<2> const& begin,
                               OpenACCMDRangeEnd<2> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(&& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLAnd(OpenACCMDRangeBegin<3> const& begin,
                               OpenACCMDRangeEnd<3> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(&& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLAnd(OpenACCMDRangeBegin<4> const& begin,
                               OpenACCMDRangeEnd<4> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(&& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLAnd(OpenACCMDRangeBegin<5> const& begin,
                               OpenACCMDRangeEnd<5> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(&& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLAnd(OpenACCMDRangeBegin<6> const& begin,
                               OpenACCMDRangeEnd<6> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(&& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLOr(OpenACCMDRangeBegin<2> const& begin,
                              OpenACCMDRangeEnd<2> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(|| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLOr(OpenACCMDRangeBegin<3> const& begin,
                              OpenACCMDRangeEnd<3> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(|| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLOr(OpenACCMDRangeBegin<4> const& begin,
                              OpenACCMDRangeEnd<4> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(|| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLOr(OpenACCMDRangeBegin<5> const& begin,
                              OpenACCMDRangeEnd<5> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(|| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceLOr(OpenACCMDRangeBegin<6> const& begin,
                              OpenACCMDRangeEnd<6> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(|| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBAnd(OpenACCMDRangeBegin<2> const& begin,
                               OpenACCMDRangeEnd<2> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBAnd(OpenACCMDRangeBegin<3> const& begin,
                               OpenACCMDRangeEnd<3> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBAnd(OpenACCMDRangeBegin<4> const& begin,
                               OpenACCMDRangeEnd<4> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBAnd(OpenACCMDRangeBegin<5> const& begin,
                               OpenACCMDRangeEnd<5> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBAnd(OpenACCMDRangeBegin<6> const& begin,
                               OpenACCMDRangeEnd<6> const& end, ValueType& val,
                               Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(& : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBOr(OpenACCMDRangeBegin<2> const& begin,
                              OpenACCMDRangeEnd<2> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(2) reduction(| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, l_val);
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBOr(OpenACCMDRangeBegin<3> const& begin,
                              OpenACCMDRangeEnd<3> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(3) reduction(| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, l_val);
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBOr(OpenACCMDRangeBegin<4> const& begin,
                              OpenACCMDRangeEnd<4> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(4) reduction(| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, l_val);
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBOr(OpenACCMDRangeBegin<5> const& begin,
                              OpenACCMDRangeEnd<5> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(5) reduction(| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, l_val);
          }
        }
      }
    }
  }
  val = l_val;
}

template <class ValueType, class Functor>
void OpenACCParallelReduceBOr(OpenACCMDRangeBegin<6> const& begin,
                              OpenACCMDRangeEnd<6> const& end, ValueType& val,
                              Functor const& functor, int const async_arg) {
  auto const begin5 = begin[5];
  auto const end5   = end[5];
  auto const begin4 = begin[4];
  auto const end4   = end[4];
  auto const begin3 = begin[3];
  auto const end3   = end[3];
  auto const begin2 = begin[2];
  auto const end2   = end[2];
  auto const begin1 = begin[1];
  auto const end1   = end[1];
  auto const begin0 = begin[0];
  auto const end0   = end[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const a_functor(functor);
  auto l_val = val;

// clang-format off
#pragma acc parallel loop gang vector collapse(6) reduction(| : l_val) copyin(a_functor) async(async_arg)
  // clang-format on
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, l_val);
            }
          }
        }
      }
    }
  }
  val = l_val;
}

}  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(REDUCER)               \
  template <class Functor, class Scalar, class Space, class... Traits>    \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceMDRangeHelper<  \
      Functor, Kokkos::REDUCER<Scalar, Space>,                            \
      Kokkos::MDRangePolicy<Traits...>, true> {                           \
    using Policy    = MDRangePolicy<Traits...>;                           \
    using Reducer   = REDUCER<Scalar, Space>;                             \
    using ValueType = typename Reducer::value_type;                       \
                                                                          \
    OpenACCParallelReduceMDRangeHelper(Functor const& functor,            \
                                       Reducer const& reducer,            \
                                       Policy const& policy) {            \
      ValueType val;                                                      \
      reducer.init(val);                                                  \
                                                                          \
      int const async_arg = policy.space().acc_async_queue();             \
                                                                          \
      OpenACCParallelReduce##REDUCER(policy.m_lower, policy.m_upper, val, \
                                     functor, async_arg);                 \
                                                                          \
      reducer.reference() = val;                                          \
    }                                                                     \
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
