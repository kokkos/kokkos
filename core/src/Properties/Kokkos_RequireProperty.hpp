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

#ifndef KOKKOS_PROPERTIES_REQUIREPROPERTY_HPP
#define KOKKOS_PROPERTIES_REQUIREPROPERTY_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Properties/Kokkos_Detection.hpp>

namespace Kokkos {

namespace Experimental {

template <class T, class... Properties>
struct can_require;

} // end namespace Experimental

namespace Impl {

namespace RequirePropertyFnImpl {


struct RequirePropertyFn
{

private:

  struct _not_requirable {
    static constexpr auto is_requirable = false;
  };
  //template <class Prop>
  //using _is_requirable_archetype = decltype(Prop::is_requirable);
  KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
    _is_requirable_archetype, Prop,
    decltype(Prop::is_requirable)
  );
  template <class Prop>
  using is_requirable_t = detected_or_t<
    _not_requirable, _is_requirable_archetype, Prop
  >;

  //template <class T, class Prop>
  //using _has_require_property_method_archetype =
  //  decltype(declval<T>().require_property(declval<Prop>()));
  KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
    _has_require_property_method_archetype, T, Prop,
    decltype(declval<T>().require_property(declval<Prop>()))
  );
  template <class T, class Prop>
  using has_require_property_method =
    is_detected<_has_require_property_method_archetype, T, Prop>;
  template <class T, class Prop>
  using require_property_method_result =
    detected_t<_has_require_property_method_archetype, T, Prop>;

  //template <class T, class Prop>
  //using _has_adl_require_property_archetype =
  //  decltype(require_property(declval<T>(), declval<Prop>()));
  KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
    _has_adl_require_property_archetype, T, Prop,
    decltype(require_property(declval<T>(), declval<Prop>()))
  );
  template <class T, class Prop>
  using has_adl_require_property =
    is_detected<_has_adl_require_property_archetype, T, Prop>;
  template <class T, class Prop>
  using adl_require_property_result =
    detected_t<_has_adl_require_property_archetype, T, Prop>;

public:

  // TODO @properties DSH propagate noexcept
  // TODO @properties DSH think about how reference qualified methods affect this
  // TODO @properties DSH avoid causing instantiation of property method/function if is_requirable is false
  // TODO @properties DSH is_applicable_property guard

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr
  typename std::enable_if<
    is_requirable_t<Prop>::value
      && has_require_property_method<T, Prop>::value,
    require_property_method_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return std::forward<T>(obj).require_property(std::forward<Prop>(prop));
  }

  template <class T, class Prop>
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr
  typename std::enable_if<
    is_requirable_t<Prop>::value
      && has_adl_require_property<T, Prop>::value
      && not has_require_property_method<T, Prop>::value,
    adl_require_property_result<T, Prop>
  >::type
  operator()(T&& obj, Prop&& prop) const {
    return require_property(std::forward<T>(obj), std::forward<Prop>(prop));
  }

  template <class T, class Prop1, class Prop2, class... Props>
  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto
  operator()(T&& obj, Prop1&& p1, Prop2&& p2, Props&&... props) const
    // This is SFINAE that basically says that if a left fold of the single
    // argument calls is valid, then this overload should be valid
    noexcept(noexcept(
      declval<RequirePropertyFn>()(
        declval<RequirePropertyFn>()(std::forward<T>(obj), std::forward<Prop1>(p1)),
        std::forward<Prop2>(p2),
        std::forward<Props>(props)...
      )
    ))
    // Yes, the trailing return type is the same thing as the noexcept clause
    -> decltype(
      declval<RequirePropertyFn>()(
        declval<RequirePropertyFn>()(std::forward<T>(obj), std::forward<Prop1>(p1)),
        std::forward<Prop2>(p2),
        std::forward<Props>(props)...
      )
    )
  {
    return
      (*this)(
        (*this)(std::forward<T>(obj), std::forward<Prop1>(p1)),
        std::forward<Prop2>(p2),
        std::forward<Props>(props)
      );
  }

};

// For C++11 compatibility, we to do this since we can't use variable templates
// Use a template here to avoid ODR violations caused by putting this in a header file
template <class _always_void=void>
struct RequireFnCustomizationContainer
{
  static constexpr const RequirePropertyFn customization_point_object = { };
};

} // end namespace RequirePropertyFnImpl

} // end namespace Impl

namespace Experimental {

/**
 *  @todo document this
 */
constexpr const auto require_property =
  Kokkos::Impl::RequirePropertyFnImpl::RequireFnCustomizationContainer<>::customization_point_object;

} // end namespace Experimental
} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_REQUIREPROPERTY_HPP
