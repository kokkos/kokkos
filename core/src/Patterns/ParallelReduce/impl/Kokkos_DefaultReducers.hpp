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

#ifndef KOKKOS_PATTERNS_PARALLELREDUCE_IMPL_KOKKOS_DEFAULTREDUCERS_HPP
#define KOKKOS_PATTERNS_PARALLELREDUCE_IMPL_KOKKOS_DEFAULTREDUCERS_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Properties/Kokkos_Detection.hpp>

#include <Concepts/Functor/Kokkos_ReductionFunctor_Concept.hpp>

#include <Kokkos_PointerOwnership.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="ViewConvertibleToReductionReference"> {{{1

// TODO use this?!?

template <class ViewType>
class ViewConvertibleToReductionReferenceConstructorsImpl
  : public ViewType
{
private:
  using base_t = ViewType;
public:

  using base_t::base_t;

  KOKKOS_INLINE_FUNCTION
  ViewConvertibleToReductionReferenceConstructorsImpl(
    ViewType const& arg_view
  ) : base_t(arg_view)
  { }

};

template <class ViewType, class Enable=void>
class ViewConvertibleToReductionReference;

template <class ViewType>
class ViewConvertibleToReductionReference<
  ViewType,
  typename std::enable_if<ViewType::Rank == 0>
> : public ViewConvertibleToReductionReferenceConstructorsImpl<ViewType>
{
private:
  using base_t = ViewConvertibleToReductionReferenceConstructorsImpl<ViewType>;
public:
  using value_type = typename ViewType::value_type;
  using reference_type = typename ViewType::reference_type;

  KOKKOS_FORCEINLINE_FUNCTION
  operator reference_type() noexcept { return this->ViewType::operator()(); }

  KOKKOS_INLINE_FUNCTION_DELETED operator value_type() = delete;
};

template <class ViewType>
class ViewConvertibleToReductionReference<
  ViewType,
  typename std::enable_if<ViewType::Rank == 1>
> : public ViewType
{
private:
  using base_t = ViewConvertibleToReductionReferenceConstructorsImpl<ViewType>;
public:
  using value_type = typename ViewType::value_type;

  KOKKOS_FORCEINLINE_FUNCTION
  operator value_type*() noexcept { return this->ViewType::data(); }

  KOKKOS_INLINE_FUNCTION_DELETED operator value_type() = delete;

  using base_t::base_t;
};

// </editor-fold> end ViewConvertibleToReductionReference }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="ReducerFromFunctorArrayAspectsImpl"> {{{1

// Scalar version
template <class Functor, class ViewType, class Enable=void>
struct ReducerFromFunctorArrayAspectsImpl
{
protected:
  using value_type = typename ViewType::value_type;
  using pointer_type = typename ViewType::pointer_type;
  using reference_type = value_type&;
  using const_reference_type = value_type const&;  // TODO get the const_reference from the View also?

  KOKKOS_INLINE_FUNCTION
  unsigned get_value_count(Functor const&, ViewType const&) const noexcept { return 1; }

  KOKKOS_INLINE_FUNCTION
  reference_type get_reference(ViewType const& v) const noexcept
  {
    return v();
  }
};


template <class Functor, class ViewType>
struct ReducerFromFunctorArrayAspectsImpl<
  Functor, ViewType,
  typename std::enable_if<
    Concepts::array_reduction_functor_has_value_count<Functor, ViewType>::value
  >::type
>
{
protected:

  using value_type = typename ViewType::value_type;
  using pointer_type = typename ViewType::pointer_type;

  // reference_type is a bit of a misnomer here; it's really "the type that
  // gets passed to the user's functor in the return value position"
  using reference_type = value_type*;
  using const_reference_type = value_type const*;


  KOKKOS_INLINE_FUNCTION
  unsigned get_value_count(Functor const& f, ViewType const& v) const noexcept
  {
    return Concepts::array_reduction_functor_value_count(f, v);
  }

  KOKKOS_INLINE_FUNCTION
  reference_type get_reference(ViewType const& v) const noexcept
  {
    return v.data();
  }

public:

  static
  KOKKOS_INLINE_FUNCTION
  reference_type
  bind_reference(void* ptr) noexcept
  {
    return static_cast<value_type*>(ptr);
  }
};


// </editor-fold> end ReducerFromFunctorArrayAspectsImpl }}}1
//==============================================================================


template <
  class Functor,
  class Policy,
  class ViewType
>
class ReducerFromFunctorAndReturnValue
  : ReducerFromFunctorArrayAspectsImpl<Functor, ViewType>
{
private:

  ObservingRawPtr<Policy const> m_policy;
  ObservingRawPtr<Functor const> m_functor;
  ViewType m_result;

  using base_t = ReducerFromFunctorArrayAspectsImpl<Functor, ViewType>;

public:

  using value_type = typename base_t::value_type;
  using pointer_type = typename base_t::pointer_type;
  using reference_type = typename base_t::reference_type;
  using const_reference_type = typename base_t::const_reference_type;
  using result_view_type = ViewType;

  using reducer = ReducerFromFunctorAndReturnValue;

  KOKKOS_INLINE_FUNCTION
  void init(reference_type uninitialized) const noexcept
  {
    Concepts::reduction_functor_init(
      *m_policy, *m_functor, uninitialized
    );
  }

  KOKKOS_INLINE_FUNCTION
  void join(reference_type dst, const_reference_type src) const noexcept
    // TODO noexcept specification
  {
    Concepts::reduction_functor_join(
      *m_policy, *m_functor, dst, src
    );
  }

  KOKKOS_INLINE_FUNCTION
  void final(reference_type val) const noexcept
  {
    Concepts::reduction_functor_final(
      *m_policy, *m_functor, val
    );
  }

  KOKKOS_INLINE_FUNCTION
  reference_type reference() const noexcept
  {
    return this->base_t::get_reference(m_result);
  }

  KOKKOS_INLINE_FUNCTION
  ViewType view() const noexcept
  {
    return m_result;
  }

  KOKKOS_INLINE_FUNCTION
  unsigned value_count() const noexcept
  {
    return this->base_t::get_value_count(*m_functor, m_result);
  }

  KOKKOS_INLINE_FUNCTION constexpr
  ReducerFromFunctorAndReturnValue(
    Functor const& arg_functor,
    Policy const& arg_policy,
    ViewType arg_result
  ) noexcept
    : m_policy(&arg_policy),
      m_functor(&arg_functor),
      m_result(std::move(arg_result))
  { }

};

} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_PATTERNS_PARALLELREDUCE_IMPL_KOKKOS_DEFAULTREDUCERS_HPP
