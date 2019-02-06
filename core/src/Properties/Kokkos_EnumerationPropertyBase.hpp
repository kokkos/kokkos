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

#ifndef KOKKOS_PROPERTIES_ENUMERATIONPROPERTYBASE_HPP
#define KOKKOS_PROPERTIES_ENUMERATIONPROPERTYBASE_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Concepts.hpp>
#include <Properties/Kokkos_PropertyBase.hpp>


namespace Kokkos {
namespace Impl {

template <
  class Derived,
  bool EnumeratorsRequirable,
  bool EnumeratorsPreferable,
  class ValueRepresentation = unsigned
>
struct MultiFlagPropertyBase
  : PropertyBase<
      Derived, false, false
    >
{
protected:

  using value_representation_t = ValueRepresentation;

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr value_representation_t
  value_representation() const noexcept
  {
    return m_value;
  }

  template <class DerivedEnumerator>
  struct default_enumerator
    : PropertyBase<DerivedEnumerator, EnumeratorsRequirable, EnumeratorsPreferable>
  {
    KOKKOS_INLINE_FUNCTION
    static constexpr Derived value() noexcept { return Derived{0}; }

    KOKKOS_INLINE_FUNCTION
    constexpr bool operator==(DerivedEnumerator) const noexcept { return true; }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    friend
    constexpr bool
    query_property(T&& obj, DerivedEnumerator) noexcept {
      return (
        static_cast<MultiFlagPropertyBase&&>(
          Kokkos::Experimental::query_property(obj, Derived{})
        ).value_representation()
      ) == 0;
    }

  };


  template <
    class DerivedEnumerator,
    ValueRepresentation EnumeratorValue
  >
  struct enumerator
    : PropertyBase<DerivedEnumerator, EnumeratorsRequirable, EnumeratorsPreferable>
  {
    KOKKOS_INLINE_FUNCTION
    static constexpr Derived value() noexcept { return Derived{EnumeratorValue}; }

    KOKKOS_INLINE_FUNCTION
    constexpr bool operator==(DerivedEnumerator) const noexcept { return true; }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    friend
    constexpr bool
    query_property(T&& obj, DerivedEnumerator) noexcept {
      return (
        static_cast<MultiFlagPropertyBase&&>(
          Kokkos::Experimental::query_property(obj, Derived{})
        ).value_representation() & EnumeratorValue
      ) != 0;
    }

  };

private:

  value_representation_t m_value = value_representation_t();

public:

  KOKKOS_INLINE_FUNCTION
  constexpr
  MultiFlagPropertyBase() noexcept = default;

  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  MultiFlagPropertyBase(value_representation_t arg_value) noexcept
    : m_value(arg_value)
  { }

  KOKKOS_INLINE_FUNCTION
  friend
  constexpr bool operator==(Derived const& a, Derived const& b) noexcept
  {
    return
      static_cast<MultiFlagPropertyBase const&>(a).value_representation()
        == static_cast<MultiFlagPropertyBase const&>(b).value_representation();
  }

};

} // end namespace Impl

} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_ENUMERATIONPROPERTYBASE_HPP
