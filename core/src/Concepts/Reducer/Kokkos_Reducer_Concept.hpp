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

#ifndef KOKKOS_CONCEPTS_REDUCER_KOKKOS_REDUCER_CONCEPT_HPP
#define KOKKOS_CONCEPTS_REDUCER_KOKKOS_REDUCER_CONCEPT_HPP

#include <Concepts/Kokkos_Concepts_Common.hpp>
#include <Concepts/Kokkos_Concepts_Macros.hpp>

#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {
namespace Impl {
namespace Concepts {

/**
 *  `reducer_value_type<R>::type` is `R::value_type` by default.  (It can be
 *  specialized).  This is a requirement of the `Reducer` concept.
 *
 *  @tparam R A type that meets the requirements of the `Reducer` concept.
 */
template <class R>
struct reducer_value_type :
  is_detected<_intrusive_value_type, R> { };

template <class R>
using reducer_value_type_t = typename reducer_value_type<R>::type;


//==============================================================================
// <editor-fold desc="Reducer pointer_type"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_pointer_type_archetype, R,
  typename R::pointer_type
);

/**
 *  @todo document this
 */
template <class R>
struct reducer_pointer_type :
  detected_or<
    reducer_value_type_t<R>,
    _intrusive_pointer_type_archetype, R
  > { };

template <class R>
using reducer_pointer_type_t = typename reducer_pointer_type<R>::type;

// </editor-fold> end Reducer pointer_type }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer reference_type"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_reference_type_archetype, R,
  typename R::reference_type
);

/**
 *  @todo document this
 */
template <class R>
struct reducer_reference_type :
  detected_or<
    typename std::add_lvalue_reference<reducer_value_type_t<R>>::type,
    _intrusive_reference_type_archetype, R
  > { };

template <class R>
using reducer_reference_type_t = typename reducer_reference_type<R>::type;

// </editor-fold> end Reducer pointer_type }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer const_reference_type"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_const_reference_type_archetype, R,
  typename R::const_reference_type
);

/**
 *  @todo document this
 */
template <class R>
struct reducer_const_reference_type :
  detected_or<
    reducer_value_type_t<R> const&,
    _intrusive_const_reference_type_archetype, R
  > { };

template <class R>
using reducer_const_reference_type_t = typename reducer_const_reference_type<R>::type;

// </editor-fold> end Reducer pointer_type }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer init()"> {{{1

template <class R>
struct reducer_has_intrusive_init
  : is_detected_t<_intrusive_init_archetype, R, reducer_reference_type_t<remove_cvref_t<R>>>
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_init<R const&>::value
>::type
reducer_init(R const& r, reducer_reference_type_t<R> ref)
  noexcept(noexcept(r.init(ref)))
{
  r.init(ref);
}

// </editor-fold> end Reducer init() }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer join()"> {{{1

template <class R>
struct reducer_has_intrusive_join
  : is_detected_t<
      _intrusive_join_archetype,
      R, reducer_reference_type_t<remove_cvref_t<R>>, reducer_const_reference_type_t<remove_cvref_t<R>>
    >
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_join<R const&>::value
>::type
reducer_join(R const& r, reducer_reference_type_t<R> dst, reducer_const_reference_type_t<R> src)
  noexcept(noexcept(r.join(dst, src)))
{
  r.join(dst, src);
}

// </editor-fold> end Reducer join() }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer final()"> {{{1

template <class R>
struct reducer_has_intrusive_final
  : is_detected_t<
    _intrusive_final_archetype, R, reducer_reference_type_t<remove_cvref_t<R>>
  >
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_final<R const&>::value
>::type
reducer_final(R const& r, reducer_reference_type_t<R> ref)
  noexcept(noexcept(r.final(ref)))
{
  r.final(ref);
}

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_final<R const&>::value
>::type
reducer_final(R const& r, reducer_reference_type_t<R> ref) noexcept
{ /* intentionally empty */ }


// </editor-fold> end Reducer final() }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer view()"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_result_view_type_archetype, R,
  typename R::result_view_type
);


template <class R>
struct reducer_result_view_type
  : is_detected<_intrusive_result_view_type_archetype, R>
{ };

template <class R>
using reducer_result_view_type_t = typename reducer_result_view_type<R>::type;


KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_view_function_archetype, R,
  decltype(
    Impl::declval<R>().view()
  )
);

template <class R>
struct reducer_has_intrusive_view
  : is_detected_t<_intrusive_view_function_archetype, R>
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_view<R&>::value,
  reducer_result_view_type_t<R>
>::type
reducer_result_view(R& r)
  noexcept(noexcept(r.view()))
{
  return r.view();
}

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _reducer_result_view_customization_point_archetype, R,
  decltype(
    Concepts::reducer_result_view(Impl::declval<R>())
  )
);

template <class R>
struct reducer_has_result_view
  : is_detected_t<_reducer_result_view_customization_point_archetype, R>
{ };

// </editor-fold> end Reducer view() }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Reducer value_count()"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_value_count_function_archetype, R,
  decltype(
    Impl::declval<R>().value_count()
  )
);

template <class R>
struct reducer_has_intrusive_value_count
  : is_detected_convertible_t<unsigned, _intrusive_value_count_function_archetype, R>
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_value_count<R const&>::value,
  unsigned
>::type
reducer_value_count(R const& r)
  noexcept(noexcept(r.value_count()))
{
  return r.value_count();
}

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION constexpr
typename std::enable_if<
  !reducer_has_intrusive_value_count<R const&>::value,
  unsigned
>::type
reducer_value_count(R const& r) noexcept
{
  return 1;
}


KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _reducer_value_count_customization_point_archetype, R,
  decltype(
    Concepts::reducer_value_count(Impl::declval<R>())
  )
);

template <class R>
struct reducer_has_value_count
  : is_detected_t<_reducer_value_count_customization_point_archetype, R>
{ };

// </editor-fold> end Reducer value_count() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Reducer value_size()"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_value_size_function_archetype, R,
  decltype(
    Impl::declval<R>().value_size()
  )
);

template <class R>
struct reducer_has_intrusive_value_size
  : is_detected_convertible_t<size_t, _intrusive_value_size_function_archetype, R>
{ };

/**
 *  @todo document this
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_value_size<R const&>::value,
  size_t
>::type
reducer_value_size(R const& r)
  noexcept(noexcept(r.value_size()))
{
  return r.value_size();
}

template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_value_size<R const&>::value
  && reducer_has_value_count<R const&>::value,
  size_t
>::type
reducer_value_size(R const& r) noexcept
{
  return sizeof(reducer_value_type_t<R>) * Concepts::reducer_value_count(r);
}

// </editor-fold> end Reducer value_size() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Reducer bind_reference()"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_bind_reference_archetype, R,
  decltype(
    Impl::declval<R>().bind_reference(Impl::declval<void*>())
  )
);

template <class R>
struct reducer_has_intrusive_bind_reference
  : is_detected_exact_t<reducer_reference_type_t<remove_cvref_t<R>>, _intrusive_bind_reference_archetype, R>
{ };

template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_bind_reference<R const&>::value,
  reducer_reference_type_t<R>
>::type
reducer_bind_reference(R const& r, void* ptr) noexcept
{
  r.bind_reference(ptr);
}

template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_bind_reference<R const&>::value
    && std::is_same<reducer_reference_type_t<R>, reducer_value_type_t<R>&>::value,
  reducer_reference_type_t<R>
>::type
reducer_bind_reference(R const& r, void* ptr) noexcept
{
  return *static_cast<reducer_value_type_t<R>*>(ptr);
}

template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_bind_reference<R const&>::value
    && std::is_same<reducer_reference_type_t<R>, reducer_value_type_t<R>*>::value,
  reducer_reference_type_t<R>
>::type
reducer_bind_reference(R const& r, void* ptr) noexcept
{
  return static_cast<reducer_value_type_t<R>*>(ptr);
}


// </editor-fold> end Reducer bind_reference() }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="Reducer assign_result()"> {{{1

KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _intrusive_assign_result_archetype, R,
  decltype(
    Impl::declval<R>().assign_result(Impl::declval<reducer_reference_type_t<R>>())
  )
);


template <class R>
struct reducer_has_intrusive_assign_result
  : is_detected_t<_intrusive_assign_result_archetype, R>
{ };

/**
 *  @todo document this
 *
 *  @tparam R
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  reducer_has_intrusive_assign_result<R&>::value
>::type
reducer_assign_result(R& r, reducer_reference_type_t<R> result_to_assign_from)
  noexcept(noexcept(r.assign_result(result_to_assign_from)))
{
  r.assign_result(result_to_assign_from);
}

/**
 *  @todo document this
 *
 *  @tparam R
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_assign_result<R&>::value
  && reducer_result_view_type_t<R>::Rank == 0
>::type
reducer_assign_result(R& r, reducer_reference_type_t<R> result_to_assign_from)
  noexcept(noexcept(Concepts::reducer_result_view(r)() = result_to_assign_from))
{
  Concepts::reducer_result_view(r)() = result_to_assign_from;
}


KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _square_bracket_op_archetype, T, I,
  decltype(
    Impl::declval<T>()[Impl::declval<I>()]
  )
);

template <class T, class Index=unsigned>
struct has_square_bracket_op
  : is_detected_t<_square_bracket_op_archetype, T, Index>
{ };

/**
 *  @todo document this
 *
 *  @tparam R
 */
template <class R>
KOKKOS_CUSTOMIZATION_POINT_FUNCTION
typename std::enable_if<
  !reducer_has_intrusive_assign_result<R&>::value
    && (reducer_result_view_type_t<R>::Rank == 1)
    && reducer_has_value_count<R&>::value
    && has_square_bracket_op<reducer_reference_type_t<R>>::value
>::type
reducer_assign_result(R& r, reducer_reference_type_t<R> result_to_assign_from)
  noexcept(noexcept(Concepts::reducer_result_view(r)[int()] = result_to_assign_from[int()]))
{
  auto value_count = static_cast<int>(Concepts::reducer_value_count(r));
  for(int i = 0; i < value_count; ++i) {
    Concepts::reducer_result_view(r)[i] = result_to_assign_from[i];
  }
}

// </editor-fold> end Reducer assign_result() }}}1
//==============================================================================

} // end namespace Concepts
} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_CONCEPTS_REDUCER_KOKKOS_REDUCER_CONCEPT_HPP
