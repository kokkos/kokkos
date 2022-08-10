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

namespace Kokkos {
namespace Impl {

// Default to catch all non implemented Reducers
template <class Reducer, class FunctorType, class ExePolicy, class TagType>
struct OpenACCReductionWrapper {
  using value_type = typename Reducer::value_type;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type&) {
    Kokkos::abort(
        "[ERROR in reduce()] The given Reducer is not implemented in the "
        "OpenACC "
        "backend.\n");
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type&, const ExePolicy&, const FunctorType&) {
    Kokkos::abort(
        "[ERROR in reduce()] The given Reducer is not implemented in the "
        "OpenACC "
        "backend.\n");
  }
};

// Specializations with implemented Reducers

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<Sum<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::sum();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(+ : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(+ : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<Prod<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::prod();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(* : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(* : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<Min<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::min();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(min : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(min : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<Max<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::max();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(max : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(max : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<LAnd<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::land();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(&& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(&& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<LOr<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

  using result_view_type = Kokkos::View<value_type, Space>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::lor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(|| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(|| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<BAnd<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::band();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(& : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

template <class Scalar, class Space, class FunctorType, class TagType,
          class... Traits>
struct OpenACCReductionWrapper<BOr<Scalar, Space>, FunctorType,
                               Kokkos::RangePolicy<Traits...>, TagType> {
 public:
  using value_type = typename std::remove_cv<Scalar>::type;
  using Policy     = Kokkos::RangePolicy<Traits...>;

#pragma acc routine seq
  KOKKOS_INLINE_FUNCTION
  static void init(value_type& val) {
    val = reduction_identity<value_type>::bor();
  }

  KOKKOS_INLINE_FUNCTION
  static void reduce(value_type& tmp, const Policy& m_policy,
                     const FunctorType& m_functor) {
    const FunctorType a_functor(m_functor);
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();
    value_type ltmp;
    init(ltmp);
    if constexpr (std::is_same<TagType, void>::value) {
#pragma acc parallel loop gang vector reduction(| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(i, ltmp);
    } else {
#pragma acc parallel loop gang vector reduction(| : ltmp) copyin(a_functor)
      for (auto i = begin; i < end; i++) a_functor(TagType(), i, ltmp);
    }
    tmp = ltmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

template <class FunctorType, class ReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
    Kokkos::Experimental::OpenACC> {
 public:
  using Policy = Kokkos::RangePolicy<Traits...>;

 private:
  using WorkRange    = typename Policy::WorkRange;
  using WorkTag      = typename Policy::work_tag;
  using Member       = typename Policy::member_type;
  using LaunchBounds = typename Policy::launch_bounds;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
          FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
          WorkTag, void>::type;

  using Analysis =
      Kokkos::Impl::FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy,
          ReducerTypeFwd>;

 public:
  using pointer_type   = typename Analysis::pointer_type;
  using value_type     = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;
  using functor_type   = FunctorType;
  using size_type      = Kokkos::Experimental::OpenACC::size_type;
  using index_type     = typename Policy::index_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  inline void execute() {
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();

    if (end <= begin) {
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::ParallelFor< OpenACC > can not be executed with "
          "a range <= 0."));
    }

    const FunctorType a_functor(m_functor);
    value_type tmp;
    typename Analysis::Reducer final_reducer(&a_functor);
    final_reducer.init(&tmp);
    static constexpr int UseReducer = is_reducer<ReducerType>::value;

    if constexpr (!UseReducer) {
      OpenACCReductionWrapper<Sum<value_type>, FunctorType, Policy,
          WorkTag>::reduce(tmp, m_policy, a_functor);
    } else {
      OpenACCReductionWrapper<ReducerType, FunctorType, Policy,
          WorkTag>::reduce(tmp, m_policy, a_functor);
    }
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
