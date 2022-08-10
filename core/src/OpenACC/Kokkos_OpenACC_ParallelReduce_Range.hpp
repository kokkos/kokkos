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

namespace Kokkos {
namespace Impl {

// Default to catch all non implemented Reducers
template <class Reducer, class FunctorType, class ExePolicy>
struct OpenACCReductionWrapper {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in reduce()] The given Reducer is not implemented in the "
        "OpenACC backend.\n");
  }

  KOKKOS_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] The given Reducer is not implemented in the "
        "OpenACC backend.\n");
  }
};

// Specializations with implemented Reducers

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<Sum<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    value_type ltmp;
    init(ltmp);

#pragma acc parallel loop gang vector reduction(+ : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<Prod<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(* : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<Min<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(min : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<Max<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(max : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<LAnd<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(&& : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<LOr<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

  using result_view_type = Kokkos::View<value_type, Space>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(|| : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<BAnd<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(& : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) {
      a_functor(i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class... Traits>
struct OpenACCReductionWrapper<BOr<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();
    Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> a_functor =
        m_functor;
    value_type ltmp;
    init(ltmp);
#pragma acc parallel loop gang vector reduction(| : ltmp) copyin(a_functor)
    for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    tmp = ltmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

template <class FunctorType, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>,
                                   ReducerType, Kokkos::Experimental::OpenACC> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;

  using Analysis =
      Kokkos::Impl::FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy,
                                    ReducerTypeFwd>;

 public:
  using pointer_type = typename Analysis::pointer_type;
  using value_type   = typename Analysis::value_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  void execute() {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    if (end <= begin) {
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::ParallelFor< OpenACC > can not be executed with "
          "a range <= 0."));
    }

    auto const& a_functor = m_functor;
    value_type tmp;
    typename Analysis::Reducer final_reducer(&a_functor);
    final_reducer.init(&tmp);
    OpenACCReductionWrapper<std::conditional_t<is_reducer_v<ReducerType>,
                                               ReducerType, Sum<value_type>>,
                            FunctorType, Policy>::reduce(tmp, m_policy,
                                                         a_functor);
    m_result_ptr[0] = tmp;
  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result,
      std::enable_if_t<Kokkos::is_view<ViewType>::value, void*> = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()) {}

  ParallelReduce(const FunctorType& arg_functor, const Policy& arg_policy,
                 const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

#endif
