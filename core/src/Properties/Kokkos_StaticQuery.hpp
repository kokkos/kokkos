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

#ifndef KOKKOS_PROPERTIES_PROPERTYBASE_HPP
#define KOKKOS_PROPERTIES_PROPERTYBASE_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Properties/Kokkos_Detection.hpp>
#include <Kokkos_Concepts.hpp>
#include <Properties/Kokkos_QueryProperty.hpp>

#include <type_traits>

namespace Kokkos {

namespace Experimental {

template <class T, class Property, class=void>
struct static_query_property_enabled_if { };

template <class T, class Property>
struct static_query_property
  : static_query_property_enabled_if<T, Property>
{
  // By default, no value is available at compile time unless one is provided
};

} // end namespace Experimental


namespace Impl {

namespace StaticQueryImpl {

// TODO explain this with some comments

//template <class T, class Property>
//using _nested_static_query_property_archetype =
//  decltype(Property::template static_query_property<T>::value);
KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _nested_static_query_property_archetype, T, Property,
  decltype(Property::template static_query_property<T>::value)
);
template <class T, class Property>
using has_nested_static_query_property = is_detected<
  _nested_static_query_property_archetype, T, Property
>;
template <class T, class Property>
using nested_static_query_property_type = detected_t<
  _nested_static_query_property_archetype, T, Property
>;


//template <class T, class Property>
//using _static_query_property_archetype = decltype(T::query(declval<Property>()));
KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _static_query_property_archetype, T, Property,
  decltype(T::query(declval<Property>()))
);
template <class T, class Property>
using _has_static_query_property = is_detected<_static_query_property_archetype, T, Property>;
template <class T, class Property>
using _static_query_property_type = detected_t<_static_query_property_archetype, T, Property>;

template <class T, class Property>
KOKKOS_INLINE_FUNCTION
constexpr auto
_do_static_query_property_if_valid(
  Property prop
) -> typename std::enable_if<
       _has_static_query_property<T, Property>::value,
       _static_query_property_type<T, Property>
     >
{
  return T::query_property(prop);
}

template <class T, class Property>
KOKKOS_INLINE_FUNCTION
constexpr auto
_do_static_query_property_if_valid_constexpr(
  Property prop
) -> typename std::enable_if<
       (_do_static_query_property_if_valid(prop), true),
       _static_query_property_type<T, Property>
     >::type
{
  return _do_static_if_valid(prop);
}

//template <class T, class Property>
//using _constexpr_static_query_property_archetype =
//  decltype(_do_static_query_property_if_valid_constexpr(declval<Property>()));
KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _constexpr_static_query_property_archetype, T, Property,
  decltype(_do_static_query_property_if_valid_constexpr(declval<Property>()))
);
template <class T, class Property>
using has_constexpr_static_query_property = is_detected<
  _constexpr_static_query_property_archetype, T, Property
>;

//template <class T, class Property>
//using _static_query_property_archetype =
//  decltype(Kokkos::Experimental::static_query_property<T, Property>::value);
KOKKOS_DECLARE_DETECTION_ARCHETYPE_2PARAMS(
  _static_query_property_archetype, T, Property,
  decltype(Kokkos::Experimental::static_query_property<T, Property>::value)
);

} // end namespace StaticQueryImpl

} // end namespace Impl

namespace Experimental {


template <class T, class Property>
struct static_query_property_enabled_if<
  T, Property,
  typename std::enable_if<
    Kokkos::Impl::StaticQueryImpl::has_constexpr_static_query_property<T, Property>::value
  >::type
>
{
  static constexpr auto value = T::query(Property{});
};

template <class T, class Property>
struct static_query_property_enabled_if<
  T, Property,
  typename std::enable_if<
    // Prefer the static member function of the type, if valid as a constant expression
    not Kokkos::Impl::StaticQueryImpl::has_constexpr_static_query_property<T, Property>::value
    && Kokkos::Impl::StaticQueryImpl::has_nested_static_query_property<T, Property>::value
  >::type
>
{
  static constexpr auto value = Property::template static_query_property<T>::value;
};


template <class T, class Property>
struct can_static_query_property :
  std::integral_constant<bool,
    Kokkos::Impl::is_detected<
      Kokkos::Impl::StaticQueryImpl::_static_query_property_archetype, T, Property
    >::value
  >
{ };

} // end namespace Experimental

} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_PROPERTYBASE_HPP
