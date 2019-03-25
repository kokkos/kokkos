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

#ifndef KOKKOS_PROPERTIES_VIEWMEMORYTRAITS_HPP
#define KOKKOS_PROPERTIES_VIEWMEMORYTRAITS_HPP

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_Concepts.hpp>
#include <Properties/Kokkos_IsApplicableProperty.hpp>
#include <Properties/Kokkos_EnumerationPropertyBase.hpp>

#include <Kokkos_MemoryTraits.hpp>

// TODO these should include forward declarations, not implementations
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_View.hpp>

namespace Kokkos {
namespace Experimental {
namespace ViewProperties {

struct memory_traits_t
  : Kokkos::Impl::MultiFlagPropertyBase<
      memory_traits_t,
      true, // enumerators are requireable
      false, // enumerators are not preferable
      // Value representation should be compatible with Kokkos::MemoryTraitsFlags
      typename std::underlying_type<Kokkos::MemoryTraitsFlags>::type
    >
{
public:

  using integral_representation =
    typename std::underlying_type<Kokkos::MemoryTraitsFlags>::type;

private:

  using base_t =
    Kokkos::Impl::MultiFlagPropertyBase<
      memory_traits_t,
      true, // enumerators are requireable
      false, // enumerators are not preferable
      // Value representation should be compatible with Kokkos::MemoryTraitsFlags
      typename std::underlying_type<Kokkos::MemoryTraitsFlags>::type
    >;

public:

  //----------------------------------------------------------------------------

  using polymorphic_query_result_type = memory_traits_t;

  //----------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION
  constexpr
  memory_traits_t() noexcept = default;

  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  memory_traits_t(integral_representation arg_value) noexcept
    : base_t(arg_value)
  { }

  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  memory_traits_t(Kokkos::MemoryTraitsFlags arg_value) noexcept
    : base_t(arg_value)
  { }

  template <unsigned Flags>
  KOKKOS_INLINE_FUNCTION
  constexpr explicit
  memory_traits_t(Kokkos::MemoryTraits<Flags>) noexcept
    : base_t(Flags)
  { }

  //----------------------------------------------------------------------------

  struct unmanaged_t
    : base_t::template enumerator<unmanaged_t, MemoryTraitsFlags::Unmanaged>
  {
    static constexpr auto flag = MemoryTraitsFlags::Unmanaged;
  };

  static constexpr auto unmanaged = unmanaged_t { };

  struct random_access_t
    : base_t::template enumerator<random_access_t, MemoryTraitsFlags::RandomAccess>
  {
    static constexpr auto flag = MemoryTraitsFlags::RandomAccess;
  };

  static constexpr auto random_access = random_access_t { };

  struct atomic_t
    : base_t::template enumerator<atomic_t, MemoryTraitsFlags::Atomic>
  {
    static constexpr auto flag = MemoryTraitsFlags::Atomic;
  };

  static constexpr auto atomic = atomic_t { };

  struct restrict_t
    : base_t::template enumerator<restrict_t, MemoryTraitsFlags::Restrict>
  {
    static constexpr auto flag = MemoryTraitsFlags::Restrict;
  };

  static constexpr auto restrict = restrict_t { };

  struct aligned_t
    : base_t::template enumerator<aligned_t, MemoryTraitsFlags::Aligned>
  {
    static constexpr auto flag = MemoryTraitsFlags::Aligned;
  };

  static constexpr auto aligned = aligned_t { };
};

namespace {

static constexpr auto memory_traits = memory_traits_t{ };

} // end anonymous namespace

} // end namespace ViewProperties
} // end namespace Experimental

//==============================================================================

namespace Impl {

//template <class Trait>
//using _is_memory_traits_prop_archetype = decltype(Trait::flag);
KOKKOS_DECLARE_DETECTION_ARCHETYPE_1PARAM(
  _is_memory_traits_prop_archetype, Trait,
  decltype(Trait::flag)
);

template <class Trait>
using is_memory_traits_property = is_detected_convertible<
  Kokkos::MemoryTraitsFlags, _is_memory_traits_prop_archetype, Trait
>;

template <class ViewType, class MemTraitsProp, class=void>
struct MakeAnalogousViewWithTraits;

template <class, Kokkos::MemoryTraitsFlags Flag>
struct AddMemoryTrait;

template <unsigned Flags, Kokkos::MemoryTraitsFlags NewFlag>
struct AddMemoryTrait<MemoryTraits<Flags>, NewFlag>
{
  using type = MemoryTraits<Flags | NewFlag>;
};

template <class MemTraitsProp, class DataType, class... ViewProperties>
struct MakeAnalogousViewWithTraits<
  Kokkos::View<DataType, ViewProperties...>, MemTraitsProp,
  typename std::enable_if<is_memory_traits_property<MemTraitsProp>::value>::type
>
{
  using view_type = Kokkos::View<DataType, ViewProperties...>;
  using type = Kokkos::View<
    DataType,
    typename view_type::array_layout,
    typename view_type::memory_space,
    typename Impl::AddMemoryTrait<
      typename view_type::memory_traits, MemTraitsProp::flag
    >::type
  >;
};

} // end namespace Impl

//==============================================================================

template <class MemoryTraitsProperty, class... ViewProperties>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Impl::is_memory_traits_property<MemoryTraitsProperty>::value,
  // To avoid instantiating Kokkos::View with potentially-invalid template parameters,
  // we get the ::type member of MakeAnalogousViewWithTraits outside of the enable_if
  Impl::MakeAnalogousViewWithTraits<Kokkos::View<ViewProperties...>, MemoryTraitsProperty>
>::type::type
require_property(Kokkos::View<ViewProperties...> const& view, MemoryTraitsProperty)
{
  return { view };
}

template <class MemoryTraitsProperty, class... ViewProperties>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
  Impl::is_memory_traits_property<MemoryTraitsProperty>::value,
  // To avoid instantiating Kokkos::View with potentially-invalid template parameters,
  // we get the ::type member of MakeAnalogousViewWithTraits outside of the enable_if
  Impl::MakeAnalogousViewWithTraits<Kokkos::View<ViewProperties...>, MemoryTraitsProperty>
>::type::type
require_property(Kokkos::View<ViewProperties...>&& view, MemoryTraitsProperty)
{
  return { std::move(view) };
}


} // end namespace Kokkos

#endif //KOKKOS_PROPERTIES_VIEWMEMORYTRAITS_HPP
