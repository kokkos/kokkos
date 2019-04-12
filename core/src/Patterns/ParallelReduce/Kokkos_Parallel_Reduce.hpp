/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#ifndef KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_PARALLEL_REDUCE_HPP
#define KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_PARALLEL_REDUCE_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Patterns/ParallelReduce/impl/Kokkos_DefaultReducers.hpp>

#include <Concepts/Functor/Kokkos_Functor_Concept.hpp>
#include <Concepts/Functor/Kokkos_ReductionFunctor_Concept.hpp>

#include <impl/Kokkos_Utilities.hpp> // remove_cvref_t, for one

#include <Kokkos_Concepts.hpp> // TODO more granular inclusion of concepts
#include <Kokkos_View.hpp> // is_view

#include <string> // std::string, for labels


namespace Kokkos {

namespace Impl {

// TODO better code reuse here
template <
  class PolicyType,
  class FunctorType,
  class ViewType
>
inline void
parallel_reduce_with_view_impl(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ViewType&& result_view
) // TODO noexcept specification
{
  using reducer_type =
    Impl::ReducerFromFunctorAndReturnValue<
      remove_cvref_t<FunctorType>,
      remove_cvref_t<PolicyType>,
      remove_cvref_t<ViewType>
    >;

  using closure_impl_type =
    Impl::ParallelReduce<
      remove_cvref_t<FunctorType>,
      remove_cvref_t<PolicyType>,
      reducer_type
    >;

  #if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    auto name =
      Kokkos::Impl::ParallelConstructName<
        remove_cvref_t<FunctorType>,
        Concepts::execution_policy_work_tag_t<remove_cvref_t<PolicyType>>
      >(label);
    Kokkos::Profiling::beginParallelReduce(name.get(), 0, &kpID);
  }
  #endif

  Kokkos::Impl::shared_allocation_tracking_disable();
  // *Intentionally* don't do perfect forwarding of the functor and result view
  // here to trigger the copy constructors of the functor's data members.
  auto closure =
    closure_impl_type(functor, Impl::forward<PolicyType>(policy), result_view);
  Kokkos::Impl::shared_allocation_tracking_enable();

  closure.execute();

  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endParallelReduce(kpID);
  }
  #endif
}

template <
  class PolicyType,
  class FunctorType,
  class ReducerType
>
inline void
parallel_reduce_with_reducer_impl(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ReducerType&& reducer
) // TODO noexcept specification
{
  using closure_impl_type =
    Impl::ParallelReduce<
      remove_cvref_t<FunctorType>,
      remove_cvref_t<PolicyType>,
      remove_cvref_t<ReducerType>
    >;

  #if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    auto name =
      Kokkos::Impl::ParallelConstructName<
        remove_cvref_t<FunctorType>,
        Concepts::execution_policy_work_tag_t<remove_cvref_t<PolicyType>>
      >(label);
    Kokkos::Profiling::beginParallelReduce(name.get(), 0, &kpID);
  }
  #endif

  Kokkos::Impl::shared_allocation_tracking_disable();

  // *Intentionally* don't do perfect forwarding of the functor
  // here to trigger the copy constructors of the functor's data members.
  auto closure = closure_impl_type(
      functor,
      Impl::forward<PolicyType>(policy),
      Impl::forward<ReducerType>(reducer)
    );
  Kokkos::Impl::shared_allocation_tracking_enable();

  closure.execute();

  #if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endParallelReduce(kpID);
  }
  #endif
}

} // end namespace Impl

/**
 *  Implementer's note: A lot of these overloads could be written as fewer
 *  fewer candidates, which may improve compilation times; however, if we did
 *  that, the user would see an error internal to Kokkos when an incorrect
 *  function call is made, making it harder to understand compiler output.
 *  For a function as prominant as `parallel_reduce`, we've decided this cost
 *  is worth it (for now, at least).
 */

/*! \fn void parallel_reduce(label,policy,functor,return_argument)
    \brief Perform a parallel reduction.

    Given:

      - A functor or lambda of type `f`
      - ...

    Allowed overloads:

      parallel_reduce(label, policy, f, x)
      @TODO finish this
*/

//==============================================================================

/**
 *  @todo document this overload
 *
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ReturnType
 * @param label
 * @param policy
 * @param functor
 * @param return_value
 */
template <
  class PolicyType,
  class FunctorType,
  class ReturnType
>
inline
typename std::enable_if<
  // Needed for ambiguity with size_t overload
  is_execution_policy<Impl::remove_cvref_t<PolicyType>>::value
  // needed to resolve ambiguity with Kokkos::View and Reducer overloads
  && !is_reducer<ReturnType>::value
  && !is_view<ReturnType>::value
  && !std::is_pointer<ReturnType>::value
  && !std::is_array<ReturnType>::value
>::type
parallel_reduce(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ReturnType& return_value
)
{
  static_assert(
    !std::is_const<ReturnType>::value,
    "Reduction result can't be a const reference!"
  );

  using view_type =
    Kokkos::View<
      ReturnType,
      Kokkos::AnonymousSpace,
      Kokkos::MemoryTraits<
        Kokkos::MemoryTraitsFlags::Unmanaged | Kokkos::MemoryTraitsFlags::Restrict
      >
    >;

  // Synthesize a reducer and forward to the overload that takes a reducer
  Impl::parallel_reduce_with_view_impl(
    label,
    Impl::forward<PolicyType>(policy),
    Impl::forward<FunctorType>(functor),
    view_type(
      Kokkos::view_wrap(&return_value)
    )
  );
}

/**
 *  Array return type overload.
 *
 *  `Kokkos::Impl::Concepts::array_reduction_functor_has_value_count<FunctorType>`
 *  must be `true`.
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ReturnType
 * @param label
 * @param policy
 * @param functor
 * @param return_value
 * @return
 */
template <
  class PolicyType,
  class FunctorType,
  class ReturnType
>
inline
typename std::enable_if<
  // Needed for ambiguity with size_t overload
  is_execution_policy<Impl::remove_cvref_t<PolicyType>>::value
>::type
parallel_reduce(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ReturnType* return_value
)
{
  static_assert(
    !std::is_const<ReturnType>::value,
    "Reduction result can't be a pointer to const!"
  );


  // TODO static assertions about FunctorType being an array functor

  using view_type =
    Kokkos::View<
      ReturnType*,
      Kokkos::AnonymousSpace,
      Kokkos::MemoryTraits<
        Kokkos::MemoryTraitsFlags::Unmanaged | Kokkos::MemoryTraitsFlags::Restrict
      >
    >;

  static_assert(
    Impl::Concepts::array_reduction_functor_has_value_count<FunctorType, ReturnType*>::value,
    "Array Reduction Functor must have value_count data member"
  );

  // Synthesize a reducer and forward to the overload that takes a reducer
  Impl::parallel_reduce_with_view_impl(
    label,
    Impl::forward<PolicyType>(policy),
    Impl::forward<FunctorType>(functor),
    view_type(
      Kokkos::view_wrap(return_value),
      Impl::Concepts::array_reduction_functor_value_count(functor, return_value)
    )
  );
}

/**
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ReducerType
 * @param label
 * @param policy
 * @param functor
 * @param reducer
 * @return
 */
template <
  class PolicyType,
  class FunctorType,
  class ReducerType
>
inline
typename std::enable_if<
  // Needed for ambiguity with size_t overload
  is_execution_policy<Impl::remove_cvref_t<PolicyType>>::value
    // needed to resolve ambiguity with Kokkos::View and Reducer overloads
    && is_reducer<Impl::remove_cvref_t<ReducerType>>::value
>::type
parallel_reduce(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ReducerType&& reducer
)
{
  Impl::parallel_reduce_with_reducer_impl(
    label,
    Impl::forward<PolicyType>(policy),
    Impl::forward<FunctorType>(functor),
    reducer
  );
}

/**
 *  @todo document this overload
 *
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ResultViewType
 * @param label
 * @param policy
 * @param functor
 * @param return_value
 */
template <
  class PolicyType,
  class FunctorType,
  class ResultViewType
>
inline
typename std::enable_if<
  // Needed for ambiguity with size_t overload
  is_execution_policy<Impl::remove_cvref_t<PolicyType>>::value
    // needed to resolve ambiguity with Kokkos::View and Reducer overloads
    && is_view<Impl::remove_cvref_t<ResultViewType>>::value
>::type
parallel_reduce(
  std::string const& label,
  PolicyType&& policy,
  FunctorType&& functor,
  ResultViewType&& return_value
)
{
  static_assert(
    Impl::remove_cvref_t<ResultViewType>::Rank == 0,
    "Only rank 0 view return types are currently supported."
  );

  // Synthesize a reducer and forward to the overload that takes a reducer
  Impl::parallel_reduce_with_view_impl(
    label,
    Impl::forward<PolicyType>(policy),
    Impl::forward<FunctorType>(functor),
    Impl::forward<ResultViewType>(return_value)
  );
}

/**
 *   @todo document this
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ResultTypeOrReducer
 * @param label
 * @param end
 * @param functor
 * @param out
 */
template <
  class FunctorType,
  class ResultTypeOrReducer
>
inline void
parallel_reduce(
  std::string const& label,
  size_t end,
  FunctorType&& functor,
  ResultTypeOrReducer&& out
)
{
  static_assert(
    is_reducer<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || is_view<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || std::is_pointer<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || std::is_lvalue_reference<ResultTypeOrReducer>::value,
    "Last argument to parallel_reduce must be a Reducer, a View, a pointer,"
    " or a non-const lvalue reference"
  );

  using policy_type =
    Kokkos::RangePolicy<
      Impl::Concepts::functor_execution_space_t<Impl::remove_cvref_t<FunctorType>>
    >;

  Kokkos::parallel_reduce(
    label,
    policy_type{0, end},
    Impl::forward<FunctorType>(functor),
    Impl::forward<ResultTypeOrReducer>(out)
  );
}

/**
 *
 *   @todo document this
 *
 * @tparam PolicyType
 * @tparam FunctorType
 * @tparam ResultTypeOrReducer
 * @param policy
 * @param functor
 * @param out
 */
template <
  class PolicyType,
  class FunctorType,
  class ResultTypeOrReducer
>
inline
typename std::enable_if<
  // Needed for ambiguity with, e.g., TeamThreadBoundaryStruct overloads
  is_execution_policy<Impl::remove_cvref_t<PolicyType>>::value
    || std::is_convertible<PolicyType, size_t>::value
>::type
parallel_reduce(
  PolicyType&& policy,
  FunctorType&& functor,
  ResultTypeOrReducer&& out
)
{
  static_assert(
    is_reducer<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || is_view<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || std::is_pointer<Impl::remove_cvref_t<ResultTypeOrReducer>>::value
      || std::is_lvalue_reference<ResultTypeOrReducer>::value,
    "Last argument to parallel_reduce must be a Reducer, a View, a pointer,"
    " or a non-const lvalue reference"
  );

  Kokkos::parallel_reduce(
    "",
    Impl::forward<PolicyType>(policy),
    Impl::forward<FunctorType>(functor),
    Impl::forward<ResultTypeOrReducer>(out)
  );
}

} // end namespace Kokkos

#endif //KOKKOS_PATTERNS_PARALLEL_REDUCE_KOKKOS_PARALLEL_REDUCE_HPP
