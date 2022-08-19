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

#ifndef KOKKOS_FUNCTORANALYSIS_HPP
#define KOKKOS_FUNCTORANALYSIS_HPP

#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <impl/Kokkos_Traits.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

struct FunctorPatternInterface {
  struct FOR {};
  struct REDUCE {};
  struct SCAN {};
};

template <typename T>
struct DeduceFunctorPatternInterface;

template <class FunctorType, class ExecPolicy, class ExecutionSpace>
struct DeduceFunctorPatternInterface<
    ParallelFor<FunctorType, ExecPolicy, ExecutionSpace>> {
  using type = FunctorPatternInterface::FOR;
};

template <class FunctorType, class ExecPolicy, class ReducerType,
          class ExecutionSpace>
struct DeduceFunctorPatternInterface<
    ParallelReduce<FunctorType, ExecPolicy, ReducerType, ExecutionSpace>> {
  using type = FunctorPatternInterface::REDUCE;
};

template <class FunctorType, class ExecPolicy, class ExecutionSpace>
struct DeduceFunctorPatternInterface<
    ParallelScan<FunctorType, ExecPolicy, ExecutionSpace>> {
  using type = FunctorPatternInterface::SCAN;
};

template <class FunctorType, class ExecPolicy, class ReturnType,
          class ExecutionSpace>
struct DeduceFunctorPatternInterface<ParallelScanWithTotal<
    FunctorType, ExecPolicy, ReturnType, ExecutionSpace>> {
  using type = FunctorPatternInterface::SCAN;
};

/** \brief  Query Functor and execution policy argument tag for value type.
 *
 *  If 'value_type' is not explicitly declared in the functor
 *  then attempt to deduce the type from FunctorType::operator()
 *  interface used by the pattern and policy.
 *
 *  For the REDUCE pattern generate a Reducer and finalization function
 *  derived from what is available within the functor.
 */
template <typename PatternInterface, class Policy, class Functor,
          typename OverrideValueType = void>
struct FunctorAnalysis {
 private:
  using FOR    = FunctorPatternInterface::FOR;
  using REDUCE = FunctorPatternInterface::REDUCE;
  using SCAN   = FunctorPatternInterface::SCAN;

  //----------------------------------------

  struct void_tag {};

  template <typename P = Policy, typename = std::false_type>
  struct has_work_tag {
    using type = void;
    using wtag = void_tag;
  };

  template <typename P>
  struct has_work_tag<P, typename std::is_void<typename P::work_tag>::type> {
    using type = typename P::work_tag;
    using wtag = typename P::work_tag;
  };

  using Tag  = typename has_work_tag<>::type;
  using WTag = typename has_work_tag<>::wtag;

  //----------------------------------------
  // Check for Functor::value_type, which is either a simple type T or T[]

  template <typename F, typename = std::false_type>
  struct has_value_type {
    using type = OverrideValueType;
  };

  template <typename F>
  struct has_value_type<F,
                        typename std::is_void<typename F::value_type>::type> {
    using type = typename F::value_type;

    static_assert(!std::is_reference<type>::value &&
                      std::rank<type>::value <= 1 &&
                      std::extent<type>::value == 0,
                  "Kokkos Functor::value_type is T or T[]");
  };

  //----------------------------------------
  // If Functor::value_type does not exist then evaluate operator(),
  // depending upon the pattern and whether the policy has a work tag,
  // to determine the reduction or scan value_type.

  template <typename F, typename P = PatternInterface,
            typename V = typename has_value_type<F>::type,
            bool T     = std::is_void<Tag>::value>
  struct deduce_value_type {
    using type = V;
  };

  template <typename F>
  struct deduce_value_type<F, REDUCE, void, true> {
    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, M, M, M,
                                                             A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, M, M, M,
                                                             M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, M, M, M, M, M,
                                                             M, M, A&) const);

    using type = decltype(deduce(&F::operator()));
  };

  template <typename F>
  struct deduce_value_type<F, REDUCE, void, false> {
    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, M,
                                                             A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, M,
                                                             M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, M,
                                                             M, M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, M,
                                                             M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, M, M, M,
                                                             M, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, M, A&) const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, M, M, M, M, A&)
                                               const);

    template <typename M, typename A>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, M,
                                                             M, M, M, M, M, M,
                                                             A&) const);

    using type = decltype(deduce(&F::operator()));
  };

  template <typename F>
  struct deduce_value_type<F, SCAN, void, true> {
    template <typename M, typename A, typename I>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(M, A&, I) const);

    using type = decltype(deduce(&F::operator()));
  };

  template <typename F>
  struct deduce_value_type<F, SCAN, void, false> {
    template <typename M, typename A, typename I>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag, M, A&, I)
                                               const);

    template <typename M, typename A, typename I>
    KOKKOS_INLINE_FUNCTION static A deduce(void (Functor::*)(WTag const&, M, A&,
                                                             I) const);

    using type = decltype(deduce(&F::operator()));
  };

  //----------------------------------------

  using candidate_type = typename deduce_value_type<Functor>::type;

  static constexpr bool candidate_is_void = std::is_void<candidate_type>::value;
  static constexpr bool candidate_is_array =
      std::rank<candidate_type>::value == 1;

  //----------------------------------------

 public:
  using value_type = std::remove_extent_t<candidate_type>;

  static_assert(!std::is_const<value_type>::value,
                "Kokkos functor operator reduce argument cannot be const");

 private:
  // Stub to avoid defining a type 'void &'
  using ValueType = std::conditional_t<candidate_is_void, void_tag, value_type>;

 public:
  using pointer_type = std::conditional_t<candidate_is_void, void, ValueType*>;

  using reference_type = std::conditional_t<
      candidate_is_array, ValueType*,
      std::conditional_t<!candidate_is_void, ValueType&, void>>;

  // FIXME still needed for parallel_scan here instead of in Reducer
  static constexpr unsigned StaticValueSize =
      !candidate_is_void && !candidate_is_array ? sizeof(ValueType) : 0;

  // FIXME still needed for parallel_scan here instead of in Reducer
  KOKKOS_FORCEINLINE_FUNCTION static constexpr unsigned value_count(
      const Functor& f) {
    if constexpr (candidate_is_array)
      return f.value_count;
    else
      return candidate_is_void ? 0 : 1;
  }

  // FIXME still needed for parallel_scan here instead of in Reducer
  KOKKOS_FORCEINLINE_FUNCTION static constexpr unsigned value_size(
      const Functor& f) {
    return value_count(f) * sizeof(ValueType);
  }

 private:
  //----------------------------------------
  // parallel_reduce join operator

  template <class F, bool is_array = candidate_is_array>
  struct has_join_no_tag_function;

  template <class F>
  struct has_join_no_tag_function<F, /*is_array*/ false> {
    using ref_type  = ValueType&;
    using cref_type = const ValueType&;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ref_type, cref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(*dst, *src);
    }
  };

  template <class F>
  struct has_join_no_tag_function<F, /*is_array*/ true> {
    using ref_type  = ValueType*;
    using cref_type = const ValueType*;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ref_type, cref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(dst, src);
    }
  };

  template <class F, bool is_array = candidate_is_array>
  struct has_volatile_join_no_tag_function;

  template <class F>
  struct KOKKOS_DEPRECATED_WITH_COMMENT(
      "Reduce/scan join() taking `volatile`-qualified parameters is "
      "deprecated. Remove the `volatile` qualifier.")
      has_volatile_join_no_tag_function<F, /*is_array*/ false> {
    using vref_type  = volatile ValueType&;
    using cvref_type = const volatile ValueType&;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(*dst, *src);
    }
  };

  template <class F>
  struct KOKKOS_DEPRECATED_WITH_COMMENT(
      "Reduce/scan join() taking `volatile`-qualified parameters is "
      "deprecated. Remove the `volatile` qualifier.")
      has_volatile_join_no_tag_function<F, /*is_array*/ true> {
    using vref_type  = volatile ValueType*;
    using cvref_type = const volatile ValueType*;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(dst, src);
    }
  };

  template <class F, bool is_array = candidate_is_array>
  struct has_join_tag_function;

  template <class F>
  struct has_join_tag_function<F, /*is_array*/ false> {
    using ref_type  = ValueType&;
    using cref_type = const ValueType&;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ref_type,
                                                          cref_type));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&, ref_type,
                                                          cref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(WTag(), *dst, *src);
    }
  };

  template <class F>
  struct has_join_tag_function<F, /*is_array*/ true> {
    using ref_type  = ValueType*;
    using cref_type = const ValueType*;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ref_type,
                                                          cref_type));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ref_type,
                                                             cref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&, ref_type,
                                                          cref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(WTag(), dst, src);
    }
  };

  template <class F, bool is_array = candidate_is_array>
  struct has_volatile_join_tag_function;

  template <class F>
  struct KOKKOS_DEPRECATED_WITH_COMMENT(
      "Reduce/scan join() taking `volatile`-qualified parameters is "
      "deprecated. Remove the `volatile` qualifier.")
      has_volatile_join_tag_function<F, /*is_array*/ false> {
    using vref_type  = volatile ValueType&;
    using cvref_type = const volatile ValueType&;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(WTag(), *dst, *src);
    }
  };

  template <class F>
  struct KOKKOS_DEPRECATED_WITH_COMMENT(
      "Reduce/scan join() taking `volatile`-qualified parameters is "
      "deprecated. Remove the `volatile` qualifier.")
      has_volatile_join_tag_function<F, /*is_array*/ true> {
    using vref_type  = volatile ValueType*;
    using cvref_type = const volatile ValueType*;

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             vref_type,
                                                             cvref_type) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          vref_type,
                                                          cvref_type));

    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      f->join(WTag(), dst, src);
    }
  };

  template <class F, class = void>
  struct detected_join_no_tag : std::false_type {};

  template <class F>
  struct detected_join_no_tag<
      F, decltype(has_join_no_tag_function<F>::enable_if(&F::join))>
      : std::true_type {};

  template <class F, class = void>
  struct detected_volatile_join_no_tag : std::false_type {};

  template <class F>
  struct detected_volatile_join_no_tag<
      F, decltype(has_volatile_join_no_tag_function<F>::enable_if(&F::join))>
      : std::true_type {};

  template <class F, class = void>
  struct detected_join_tag : std::false_type {};

  template <class F>
  struct detected_join_tag<F, decltype(has_join_tag_function<F>::enable_if(
                                  &F::join))> : std::true_type {};

  template <class F, class = void>
  struct detected_volatile_join_tag : std::false_type {};

  template <class F>
  struct detected_volatile_join_tag<
      F, decltype(has_volatile_join_tag_function<F>::enable_if(&F::join))>
      : std::true_type {};

  template <class F = Functor, typename = void>
  struct DeduceJoinNoTag : std::false_type {
    KOKKOS_INLINE_FUNCTION static void join(F const* const f, ValueType* dst,
                                            ValueType const* src) {
      const int n = FunctorAnalysis::value_count(*f);
      for (int i = 0; i < n; ++i) dst[i] += src[i];
    }
  };

  template <class F>
  struct DeduceJoinNoTag<F, std::enable_if_t<(is_reducer<F>::value ||
                                              (!is_reducer<F>::value &&
                                               std::is_void<Tag>::value)) &&
                                             detected_join_no_tag<F>::value>>
      : public has_join_no_tag_function<F>, std::true_type {};

  template <class F>
  struct DeduceJoinNoTag<
      F,
      std::enable_if_t<(is_reducer<F>::value ||
                        (!is_reducer<F>::value && std::is_void<Tag>::value)) &&
                       (!detected_join_no_tag<F>::value &&
                        detected_volatile_join_no_tag<F>::value)>>
      : public has_volatile_join_no_tag_function<F>, std::true_type {};

  template <class F = Functor, typename = void>
  struct DeduceJoin : public DeduceJoinNoTag<F> {};

  template <class F>
  struct DeduceJoin<
      F, std::enable_if_t<!is_reducer<F>::value && detected_join_tag<F>::value>>
      : public has_join_tag_function<F>, std::true_type {};

  template <class F>
  struct DeduceJoin<F, std::enable_if_t<!is_reducer<F>::value &&
                                        (!detected_join_tag<F>::value &&
                                         detected_volatile_join_tag<F>::value)>>
      : public has_volatile_join_tag_function<F>, std::true_type {};

  //----------------------------------------

  template <class, bool is_array = candidate_is_array>
  struct has_init_no_tag_function;

  template <class F>
  struct has_init_no_tag_function<F, /*is_array*/ false> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ValueType&) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ValueType&));

    KOKKOS_INLINE_FUNCTION static void init(F const* const f, ValueType* dst) {
      f->init(*dst);
    }
  };

  template <class F>
  struct has_init_no_tag_function<F, /*is_array*/ true> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ValueType*) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ValueType*));

    KOKKOS_INLINE_FUNCTION static void init(F const* const f, ValueType* dst) {
      f->init(dst);
    }
  };

  template <class, bool is_array = candidate_is_array>
  struct has_init_tag_function;

  template <class F>
  struct has_init_tag_function<F, /*is_array*/ false> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ValueType&)
                                                     const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ValueType&) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ValueType&));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          ValueType&));

    KOKKOS_INLINE_FUNCTION static void init(F const* const f, ValueType* dst) {
      f->init(WTag(), *dst);
    }
  };

  template <class F>
  struct has_init_tag_function<F, /*is_array*/ true> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ValueType*)
                                                     const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ValueType*) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ValueType*));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          ValueType*));

    KOKKOS_INLINE_FUNCTION static void init(F const* const f, ValueType* dst) {
      f->init(WTag(), dst);
    }
  };

  template <class F = Functor, typename = void>
  struct DeduceInitNoTag : std::false_type {
    KOKKOS_INLINE_FUNCTION static void init(F const* const, ValueType* dst) {
      new (dst) ValueType();
    }
  };

  template <class F>
  struct DeduceInitNoTag<
      F, std::enable_if_t<is_reducer<F>::value || (!is_reducer<F>::value &&
                                                   std::is_void<Tag>::value),
                          decltype(has_init_no_tag_function<F>::enable_if(
                              &F::init))>> : public has_init_no_tag_function<F>,
                                             std::true_type {};

  template <class F = Functor, typename = void>
  struct DeduceInit : public DeduceInitNoTag<F> {};

  template <class F>
  struct DeduceInit<
      F,
      std::enable_if_t<!is_reducer<F>::value,
                       decltype(has_init_tag_function<F>::enable_if(&F::init))>>
      : public has_init_tag_function<F>, std::true_type {};

  //----------------------------------------

  template <class, bool is_array = candidate_is_array>
  struct has_final_no_tag_function;

  // No tag, not array
  template <class F>
  struct has_final_no_tag_function<F, /*is_array*/ false> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ValueType&) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ValueType&));

    KOKKOS_INLINE_FUNCTION static void final(F const* const f, ValueType* dst) {
      f->final(*dst);
    }
  };

  // No tag, is array
  template <class F>
  struct has_final_no_tag_function<F, /*is_array*/ true> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(ValueType*) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(ValueType*));

    KOKKOS_INLINE_FUNCTION static void final(F const* const f, ValueType* dst) {
      f->final(dst);
    }
  };

  template <class, bool is_array = candidate_is_array>
  struct has_final_tag_function;

  // Has tag, not array
  template <class F>
  struct has_final_tag_function<F, /*is_array*/ false> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ValueType&)
                                                     const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ValueType&) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ValueType&));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          ValueType&));

    KOKKOS_INLINE_FUNCTION static void final(F const* const f, ValueType* dst) {
      f->final(WTag(), *dst);
    }
  };

  // Has tag, is array
  template <class F>
  struct has_final_tag_function<F, /*is_array*/ true> {
    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag, ValueType*)
                                                     const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (F::*)(WTag const&,
                                                             ValueType*) const);

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag, ValueType*));

    KOKKOS_INLINE_FUNCTION static void enable_if(void (*)(WTag const&,
                                                          ValueType*));

    KOKKOS_INLINE_FUNCTION static void final(F const* const f, ValueType* dst) {
      f->final(WTag(), dst);
    }
  };

  template <class F = Functor, typename = void>
  struct DeduceFinalNoTag : std::false_type {
    KOKKOS_INLINE_FUNCTION
    static void final(F const* const, ValueType*) {}
  };

  template <class F>
  struct DeduceFinalNoTag<
      F, std::enable_if_t<is_reducer<F>::value || (!is_reducer<F>::value &&
                                                   std::is_void<Tag>::value),
                          decltype(has_final_no_tag_function<F>::enable_if(
                              &F::final))>>
      : public has_final_no_tag_function<F>, std::true_type {};

  template <class F = Functor, typename = void>
  struct DeduceFinal : public DeduceFinalNoTag<F> {};

  template <class F>
  struct DeduceFinal<F, std::enable_if_t<!is_reducer<F>::value,
                                         decltype(has_final_tag_function<
                                                  F>::enable_if(&F::final))>>
      : public has_final_tag_function<F>, std::true_type {};

  //----------------------------------------

 public:
  // FIXME_OPENMPTARGET Only OpenMPTarget is really using these three variables,
  // instead of the corresponding member functions below and it shouldn't really
  // need it.
  static constexpr bool has_join_member_function  = DeduceJoin<>::value;
  static constexpr bool has_init_member_function  = DeduceInit<>::value;
  static constexpr bool has_final_member_function = DeduceFinal<>::value;

  static_assert((Kokkos::is_reducer<Functor>::value &&
                 has_join_member_function) ||
                    !Kokkos::is_reducer<Functor>::value,
                "Reducer must have a join member function!");

  struct Reducer {
   private:
    Functor const m_functor;

   public:
    using reducer        = Reducer;
    using value_type     = std::remove_const_t<FunctorAnalysis::value_type>;
    using pointer_type   = value_type*;
    using reference_type = FunctorAnalysis::reference_type;
    using functor_type   = Functor;  // Adapts a functor

    static constexpr bool has_join_member_function() {
      return DeduceJoin<>::value;
    }
    static constexpr bool has_init_member_function() {
      return DeduceInit<>::value;
    }
    static constexpr bool has_final_member_function() {
      return DeduceFinal<>::value;
    }

    KOKKOS_FUNCTION unsigned value_size() const {
      return FunctorAnalysis::value_size(m_functor);
    }

    KOKKOS_FUNCTION unsigned value_count() const {
      return FunctorAnalysis::value_count(m_functor);
    }

    KOKKOS_FUNCTION static constexpr unsigned int static_value_size() {
      return StaticValueSize;
    }

    template <bool is_array = candidate_is_array>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<is_array, reference_type>
    reference(ValueType* dst) noexcept {
      return dst;
    }

    template <bool is_array = candidate_is_array>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<!is_array, reference_type>
    reference(ValueType* dst) noexcept {
      return *dst;
    }

    KOKKOS_INLINE_FUNCTION
    void copy(ValueType* const dst, ValueType const* const src) const noexcept {
      for (unsigned int i = 0; i < value_count(); ++i) dst[i] = src[i];
    }

    KOKKOS_INLINE_FUNCTION
    void join(ValueType* dst, ValueType const* src) const noexcept {
      DeduceJoin<>::join(&m_functor, dst, src);
    }

    KOKKOS_INLINE_FUNCTION reference_type init(ValueType* const dst) const
        noexcept {
      DeduceInit<>::init(&m_functor, dst);
      return reference(dst);
    }

    KOKKOS_INLINE_FUNCTION
    void final(ValueType* dst) const noexcept {
      DeduceFinal<>::final(&m_functor, dst);
    }

    Reducer(Reducer const&) = default;
    Reducer(Reducer&&)      = default;
    Reducer& operator=(Reducer const&) = delete;
    Reducer& operator=(Reducer&&) = delete;
    ~Reducer()                    = default;

    KOKKOS_INLINE_FUNCTION explicit constexpr Reducer(
        Functor const& arg_functor) noexcept
        : m_functor(arg_functor) {}
  };
};

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* KOKKOS_FUNCTORANALYSIS_HPP */
