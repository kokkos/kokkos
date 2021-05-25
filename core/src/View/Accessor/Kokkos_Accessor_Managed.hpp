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

#ifndef KOKKOS_KOKKOS_ACCESSOR_MANAGED_HPP
#define KOKKOS_KOKKOS_ACCESSOR_MANAGED_HPP

#include <Kokkos_Macros.hpp>

#include <View/Accessor/Kokkos_Accessor_fwd.hpp>

#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="use_count is an optional accessor customization"> {{{1

template <class Accessor, class SFINAESafeDetectionSlot = void>
struct InvokeUseCountCustomization {
  KOKKOS_INLINE_FUNCTION
  static constexpr auto use_count(Accessor const& acc) { return 0; }
};

template <class Accessor>
struct InvokeUseCountCustomization<
    Accessor, void_t<decltype(std::declval<Accessor const&>().use_count())>> {
  KOKKOS_INLINE_FUNCTION
  static constexpr auto use_count(Accessor const& acc) { return acc.use_count(); }
};

template <class Accessor>
KOKKOS_INLINE_FUNCTION auto use_count_for_accessor(
    Accessor const& acc) {
  return InvokeUseCountCustomization<Accessor>::use_count(acc);
}

// </editor-fold> end use_count is an optional accessor customization }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="get_label is an optional accessor customization"> {{{1

template <class Accessor, class SFINAESafeDetectionSlot = void>
struct InvokeGetLabelCustomization {
  static const std::string& get_label(Accessor const& acc) { return ""; }
};

template <class Accessor>
struct InvokeGetLabelCustomization<
    Accessor, void_t<decltype(std::declval<Accessor const&>().get_label())>> {
static const std::string& get_label(Accessor const& acc) { return acc.get_label(); }
};

template <class Accessor>
KOKKOS_INLINE_FUNCTION const std::string& get_label_from_accessor(
    Accessor const& acc) {
  return InvokeGetLabelCustomization<Accessor>::get_label(acc);
}

// </editor-fold> end get_label is an optional accessor customization }}}1
//==============================================================================

// Managed memory case
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::Unmanaged,
                                         /* FlagIsSet = */ false>
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Unmanaged, false>> {
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Unmanaged, false>>;
  using base_t::base_t;

  using tracker_type = SharedAllocationTracker;

  template <class Record>
  BuildAccessorForMemoryTraitsFlags(Record* arg_record)
      : base_t(), m_tracker() {
    m_tracker.assign_allocated_record_to_uninitialized(arg_record);
  }

  // TODO assignment operator

  KOKKOS_FUNCTION
  auto use_count() const { return m_tracker.use_count(); }

  KOKKOS_FUNCTION
  auto get_label() const {
    return m_tracker.template get_label<typename ViewTraits::memory_space>();
  }

 private:
  tracker_type m_tracker;
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_ACCESSOR_MANAGED_HPP
