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
#include <Kokkos_Parallel.hpp>
#include <type_traits>

namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct OpenACCParallelReduceMDRangeHelper {
  OpenACCParallelReduceMDRangeHelper(Functor const&, Reducer const&,
                                     Policy const&) {
    static_assert(std::is_void_v<Functor>, "not implemented");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class Functor, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<Functor, Kokkos::MDRangePolicy<Traits...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
  using Policy = MDRangePolicy<Traits...>;

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

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceSum(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(+ : val) copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceSum(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(+ : val) copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceSum(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(+ : val) copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceSum(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(+ : val) copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceSum(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(+ : val) copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceProd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(* : val) copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceProd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(* : val) copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceProd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(* : val) copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceProd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(* : val) copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceProd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(* : val) copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceMin(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(min    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceMin(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(min    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceMin(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(min    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceMin(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(min    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceMin(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(min    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceMax(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_argc) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(max    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      a_functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceMax(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_argc) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(max    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        a_functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceMax(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_argc) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(max    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          a_functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceMax(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_argc) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(max    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            a_functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceMax(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_argc) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(max    \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              a_functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceLAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(&& : val) copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceLAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(&& : val) copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceLAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(&& : val) copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceLAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(&& : val) copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceLAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(&& : val) copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceLOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(||     \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceLOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(||     \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceLOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(||     \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceLOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(||     \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceLOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(||     \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceBAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(&                     \
                                                : val) copyin(a_functor) \
    async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceBAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(&                     \
                                                : val) copyin(a_functor) \
    async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceBAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(&                     \
                                                : val) copyin(a_functor) \
    async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceBAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(&                     \
                                                : val) copyin(a_functor) \
    async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceBAnd(Policy const& policy,
                                                      ValueType& val,
                                                      Functor const& functor,
                                                      int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(&                     \
                                                : val) copyin(a_functor) \
    async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 2> OpenACCParallelReduceBOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(2) reduction(|      \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i1 = begin1; i1 < end1; i1++) {
    for (auto i0 = begin0; i0 < end0; i0++) {
      functor(i0, i1, val);
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 3> OpenACCParallelReduceBOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(3) reduction(|      \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i2 = begin2; i2 < end2; i2++) {
    for (auto i1 = begin1; i1 < end1; i1++) {
      for (auto i0 = begin0; i0 < end0; i0++) {
        functor(i0, i1, i2, val);
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 4> OpenACCParallelReduceBOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(4) reduction(|      \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i3 = begin3; i3 < end3; i3++) {
    for (auto i2 = begin2; i2 < end2; i2++) {
      for (auto i1 = begin1; i1 < end1; i1++) {
        for (auto i0 = begin0; i0 < end0; i0++) {
          functor(i0, i1, i2, i3, val);
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 5> OpenACCParallelReduceBOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(5) reduction(|      \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i4 = begin4; i4 < end4; i4++) {
    for (auto i3 = begin3; i3 < end3; i3++) {
      for (auto i2 = begin2; i2 < end2; i2++) {
        for (auto i1 = begin1; i1 < end1; i1++) {
          for (auto i0 = begin0; i0 < end0; i0++) {
            functor(i0, i1, i2, i3, i4, val);
          }
        }
      }
    }
  }
}

template <class Policy, class ValueType, class Functor, int Rank>
std::enable_if_t<Rank == 6> OpenACCParallelReduceBOr(Policy const& policy,
                                                     ValueType& val,
                                                     Functor const& functor,
                                                     int async_arg) {
  auto const begin5 = policy.m_lower[5];
  auto const end5   = policy.m_upper[5];
  auto const begin4 = policy.m_lower[4];
  auto const end4   = policy.m_upper[4];
  auto const begin3 = policy.m_lower[3];
  auto const end3   = policy.m_upper[3];
  auto const begin2 = policy.m_lower[2];
  auto const end2   = policy.m_upper[2];
  auto const begin1 = policy.m_lower[1];
  auto const end1   = policy.m_upper[1];
  auto const begin0 = policy.m_lower[0];
  auto const end0   = policy.m_upper[0];

  if ((end0 <= begin0) || (end1 <= begin1) || (end2 <= begin2) ||
      (end3 <= begin3) || (end4 <= begin4) || (end5 <= begin5)) {
    return;
  }

  auto const& a_functor(functor);

#pragma acc parallel loop gang vector collapse(6) reduction(|      \
                                                            : val) \
    copyin(a_functor) async(async_arg)
  for (auto i5 = begin5; i5 < end5; i5++) {
    for (auto i4 = begin4; i4 < end4; i4++) {
      for (auto i3 = begin3; i3 < end3; i3++) {
        for (auto i2 = begin2; i2 < end2; i2++) {
          for (auto i1 = begin1; i1 < end1; i1++) {
            for (auto i0 = begin0; i0 < end0; i0++) {
              functor(i0, i1, i2, i3, i4, i5, val);
            }
          }
        }
      }
    }
  }
}

}  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_OPENACC_PARALLEL_REDUCE_HELPER(REDUCER)              \
  template <class Functor, class Scalar, class Space, class... Traits>   \
  struct Kokkos::Experimental::Impl::OpenACCParallelReduceMDRangeHelper< \
      Functor, Kokkos::REDUCER<Scalar, Space>,                           \
      Kokkos::MDRangePolicy<Traits...>, true> {                          \
    using Policy    = MDRangePolicy<Traits...>;                          \
    using Reducer   = REDUCER<Scalar, Space>;                            \
    using ValueType = typename Reducer::value_type;                      \
                                                                         \
    OpenACCParallelReduceMDRangeHelper(Functor const& functor,           \
                                       Reducer const& reducer,           \
                                       Policy const& policy) {           \
      ValueType val;                                                     \
      reducer.init(val);                                                 \
                                                                         \
      int const async_arg = policy.space().acc_async_queue();            \
                                                                         \
      OpenACCParallelReduce##REDUCER<Policy::rank>(policy, val, functor, \
                                                   async_arg);           \
                                                                         \
      reducer.reference() = val;                                         \
    }                                                                    \
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
