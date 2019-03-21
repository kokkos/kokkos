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

#ifndef KOKKOS_PROPERTIES_PREFERPROPERTY_HPP
#define KOKKOS_PROPERTIES_PREFERPROPERTY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {
namespace Impl {

namespace PreferPropertyFnImpl {


struct PreferPropertyFn
{

private:

  struct _not_preferable {
    static constexpr auto is_preferable = false;
  };
  template <class Prop>
  using _is_preferable_archetype = decltype(Prop::is_preferable);
  template <class Prop>
  using is_preferable_t = detected_or_t<
    _not_preferable, _is_preferable_archetype, Prop
  >;

  //template <class T, class Prop>
  //using _has_prefer_property_method_archetype =
  //  decltype(declval<T>().prefer_property(declval<Prop>()));
  KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
    _has_prefer_property_method_archetype, T, Prop,
    decltype(declval<T>().prefer_property(declval<Prop>()))
  );
  template <class T, class Prop>
  using has_prefer_property_method =
    is_detected<_has_prefer_property_method_archetype, T, Prop>;
  template <class T, class Prop>
  using prefer_property_method_result =
    detected_t<_has_prefer_property_method_archetype, T, Prop>;

  //template <class T, class Prop>
  //using _has_adl_prefer_property_archetype =
  //  decltype(prefer_property(declval<T>(), declval<Prop>()));
  KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
    _has_adl_prefer_property_archetype, T, Prop,
    decltype(prefer_property(declval<T>(), declval<Prop>()))
  );
  template <class T, class Prop>
  using has_adl_prefer_property =
    is_detected<_has_adl_prefer_property_archetype, T, Prop>;
  template <class T, class Prop>
  using adl_prefer_property_result =
    detected_t<_has_adl_prefer_property_archetype, T, Prop>;

public:

  // TODO @properties DSH propagate noexcept
  // TODO @properties DSH think about how reference qualified methods affect this
  // TODO @properties DSH avoid causing instantiation of property method/function if is_preferable is false
  // TODO @properties DSH is_applicable_property guard

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<
    is_preferable_t<Prop>::value
      && has_prefer_property_method<T, Prop>::value,
    prefer_property_method_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return std::forward<T>(obj).prefer_property(std::forward<Prop>(prop));
  }

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  typename std::enable_if<
    is_preferable_t<Prop>::value
      && has_adl_prefer_property<T, Prop>::value
      && not has_prefer_property_method<T, Prop>::value,
    adl_prefer_property_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return prefer_property(std::forward<T>(obj), std::forward<Prop>(prop));
  }

};

// For C++11 compatibility, we to do this since we can't use variable templates
// Use a template here to avoid ODR violations caused by putting this in a header file
template <class _always_void=void>
struct PreferFnCustomizationContainer
{
  static constexpr const PreferPropertyFn customization_point_object = { };
};

} // end namespace PreferPropertyFnImpl

} // end namespace Impl

namespace Experimental {

namespace {

/**
 *  @todo document this
 */
constexpr const auto prefer_property =
  Kokkos::Impl::PreferPropertyFnImpl::PreferFnCustomizationContainer<>::customization_point_object;

} // end anonymous namespace

} // end namespace Experimental
} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_PREFERPROPERTY_HPP
