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

#ifndef KOKKOS_KOKKOS_MDSPANACCESSOR_HPP
#define KOKKOS_KOKKOS_MDSPANACCESSOR_HPP

#include <Kokkos_Macros.hpp>

#include <View/Accessor/Kokkos_Accessor_fwd.hpp>
#include <View/Accessor/Kokkos_Accessor_Convertibility.hpp>
#include <View/Accessor/Kokkos_Accessor_Managed.hpp>
#include <View/Accessor/Kokkos_Accessor_RandomAccess.hpp>
#include <View/Accessor/Kokkos_Accessor_Atomic.hpp>
#include <View/Accessor/Kokkos_Accessor_Restrict.hpp>
#include <View/Accessor/Kokkos_Accessor_Align.hpp>
#include <View/Accessor/Kokkos_AccessorFromFlags.hpp>

#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Atomic_View.hpp>

#include <experimental/mdspan>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="Call the verify_pointer customization point"> {{{1

// All of this is basically just boilerplate for invoking a customization point
// called `verify_pointer` on accessors that support checking the pointer that
// a View is constructed with (for instance, aligned accessors).
// This can be simplified a bit once we have detection idiom in Kokkos

template <class Accessor, class SFINAESafeDetectionSlot = void>
struct InvokeVerifyPointerCustomization {
  KOKKOS_INLINE_FUNCTION
  static constexpr void verify_pointer(Accessor const& acc,
                                       typename Accessor::pointer const& ptr) {}
};

template <class Accessor>
struct InvokeVerifyPointerCustomization<
    Accessor, void_t<decltype(std::declval<Accessor const&>().verify_pointer(
                  std::declval<typename Accessor::pointer const&>()))>> {
  KOKKOS_INLINE_FUNCTION
  static constexpr void verify_pointer(Accessor const& acc,
                                       typename Accessor::pointer const& ptr) {
    acc.verify_pointer(ptr);
  }
};

template <class Accessor>
KOKKOS_INLINE_FUNCTION void verify_pointer_construction_for_accessor(
    Accessor const& acc, typename Accessor::pointer const& ptr) {
  InvokeVerifyPointerCustomization<Accessor>::verify_pointer(acc, ptr);
}

// TODO @mdspan implement pointer verification for accessors where it makes
//              sense to do so

// </editor-fold> end Call the verify_pointer customization point }}}1
//==============================================================================

template <class T, unsigned Flags>
struct AccessorForMemoryTraitsFlags<T, MemoryTraits<Flags>>
    : BuildAccessorForMemoryTraitsFlags<T, Flags,
                                        MemoryTraitsFlags::Unmanaged> {
 public:
  using base_t =
      BuildAccessorForMemoryTraitsFlags<T, Flags, MemoryTraitsFlags::Unmanaged>;

  using base_t::base_t;
  using pointer       = typename base_t::pointer;
  using reference     = typename base_t::reference;
  using offset_policy = typename base_t::offset_policy;

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr reference access(typename base_t::pointer p, ptrdiff_t i) const
      noexcept {
    // return p[i];
    return base_t::crtp_access_mixin(*this, p, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto offset(typename base_t::pointer p, ptrdiff_t i) const
      noexcept {
    return base_t::crtp_offset_mixin(*this, p, i);
  }
};

template <class T, class MemTraits, class Enable = void>
struct MDSpanAccessorFromKokkosMemoryTraits
    : identity<AccessorForMemoryTraitsFlags<T, MemTraits>> {};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_MDSPANACCESSOR_HPP
