/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
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

#ifndef KOKKOS_PROPERTIES_QUERYPROPERTY_HPP
#define KOKKOS_PROPERTIES_QUERYPROPERTY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {
namespace Impl {

namespace QueryPropertyFnImpl {


struct QueryPropertyFn
{

private:

  template <class T, class Prop>
  using _has_query_property_method_archetype =
    decltype(declval<T>().query_property(declval<Prop>()));
  template <class T, class Prop>
  using has_query_property_method =
    is_detected<_has_query_property_method_archetype, T, Prop>;
  template <class T, class Prop>
  using query_property_method_result =
    detected_t<_has_query_property_method_archetype, T, Prop>;

  template <class T, class Prop>
  using _has_adl_query_property_archetype =
    decltype(query_property(declval<T>(), declval<Prop>()));
  template <class T, class Prop>
  using has_adl_query_property =
    is_detected<_has_adl_query_property_archetype, T, Prop>;
  template <class T, class Prop>
  using adl_query_property_result =
    detected_t<_has_adl_query_property_archetype, T, Prop>;

public:

  // TODO @properties DSH propagate noexcept
  // TODO @properties DSH think about how reference qualified methods affect this

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr
  typename std::enable_if<
    has_query_property_method<T, Prop>::value,
    query_property_method_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return std::forward<T>(obj).query_property(std::forward<Prop>(prop));
  }

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr
  typename std::enable_if<
    has_adl_query_property<T, Prop>::value
      && not has_query_property_method<T, Prop>::value,
    adl_query_property_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return query_property(std::forward<T>(obj), std::forward<Prop>(prop));
  }

};

// For C++11 compatibility, we to do this since we can't use variable templates
// Use a template here to avoid ODR violations caused by putting this in a header file
template <class _always_void=void>
struct QueryFnCustomizationContainer
{
  static constexpr const QueryPropertyFn customization_point_object = { };
};

} // end namespace QueryPropertyFnImpl

} // end namespace Impl

namespace Experimental {

namespace {

/**
 *  @todo document this
 */
constexpr const auto query_property =
  Kokkos::Impl::QueryPropertyFnImpl::QueryFnCustomizationContainer<>::customization_point_object;

} // end anonymous namespace

} // end namespace Experimental
} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_QUERYPROPERTY_HPP
