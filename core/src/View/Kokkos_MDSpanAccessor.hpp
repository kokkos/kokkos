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

#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Atomic_View.hpp>

#include <experimental/mdspan>

namespace Kokkos {
namespace Impl {

template <class T, class MemTraits>
struct AccessorForMemoryTraitsFlags;

template <class ViewTraits, unsigned Flags, MemoryTraitsFlags CurrentFlag,
          bool FlagIsSet = CurrentFlag bitand Flags, class Enable = void>
struct BuildAccessorForMemoryTraitsFlags;

template <class Self>
struct MixinAccessorFlagConvertibility;

// Avoid having to rewrite the converting constructors everywhere, since most of
// our memory traits are convertible to and from their negation. There's a lot
// of boilerplate involved in doing this that makes the code less readable,
// but with this mixin and `using base_t::base_t` we don't have to repeat that
// boilerplate for each trait. The mixin is also constructed in such a
// way as to maintain linear inheritance by advancing the current flag in its
// base class specification.
template <class ViewTraits, unsigned Flags, MemoryTraitsFlags CurrentFlag,
          bool FlagIsSet>
struct MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
    ViewTraits, Flags, CurrentFlag, FlagIsSet>>
    : BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, next_flag_v<MemoryTraitsFlags, CurrentFlag>> {
  using base_t = BuildAccessorForMemoryTraitsFlags<
      ViewTraits, Flags, next_flag_v<MemoryTraitsFlags, CurrentFlag>>;
  template <class T, unsigned OtherFlags>
  using analogous_opposite_flag_accessor =
      BuildAccessorForMemoryTraitsFlags<T, OtherFlags, CurrentFlag, !FlagIsSet>;

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // <editor-fold desc="rule of 6 ctors, destructor, and assignment"> {{{3

  MixinAccessorFlagConvertibility() = default;

  MixinAccessorFlagConvertibility(MixinAccessorFlagConvertibility const&) =
      default;

  MixinAccessorFlagConvertibility(MixinAccessorFlagConvertibility&&) = default;

  MixinAccessorFlagConvertibility& operator   =(
      MixinAccessorFlagConvertibility const&) = default;

  MixinAccessorFlagConvertibility& operator=(
      MixinAccessorFlagConvertibility&&) = default;

  ~MixinAccessorFlagConvertibility() = default;

  // </editor-fold> end rule of 6 ctors, destructor, and assignment }}}3
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  // Convertible copy constructor
  template <class T, unsigned OtherFlags,
            //----------------------------------------
            /* requires
             *  std::is_convertible_v<
             *    base_t,
             *    analogous_opposite_flag_accessor<T, OtherFlags>>,
             */
            std::enable_if_t<
                std::is_convertible<base_t, analogous_opposite_flag_accessor<
                                                T, OtherFlags>>::value,
                int> = 0
            //----------------------------------------
            >
  MixinAccessorFlagConvertibility(
      analogous_opposite_flag_accessor<T, OtherFlags> const& other)
      : base_t(other) {}

  // Convertible move constructor
  template <class T, unsigned OtherFlags,
            //----------------------------------------
            /* requires
             *  std::is_convertible_v<
             *    base_t,
             *    analogous_opposite_flag_accessor<T, OtherFlags>>,
             */
            std::enable_if_t<
                std::is_convertible<base_t, analogous_opposite_flag_accessor<
                                                T, OtherFlags>>::value,
                int> = 0
            //----------------------------------------
            >
  MixinAccessorFlagConvertibility(
      analogous_opposite_flag_accessor<T, OtherFlags>&& other)
      : base_t(std::move(other)) {}
};

//==============================================================================
// <editor-fold desc="MDSpanAccessorFromKokkosMemoryTraits"> {{{1

// default case: don't do anything to the accessor
template <class ViewTraits, unsigned Flags, MemoryTraitsFlags CurrentFlag,
          bool FlagIsSet, class Enable>
struct BuildAccessorForMemoryTraitsFlags
    // Some flags don't need to mixin anything for enablement and/or disablement
    // of the flag (i.e., either or both of the false and true FlagIsSet cases
    // are unspecialized), so we can just move on to the next flag for those.
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, CurrentFlag, FlagIsSet>> {
  // Enable convertability from the opposite FlagIsSet version by default
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, CurrentFlag, FlagIsSet>>;
  using base_t::base_t;
};

// Base case (end of flags)
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::END_OF_FLAGS,
                                         /* FlagIsSet = */ false>
    : protected std::experimental::accessor_basic<
          typename ViewTraits::mdspan_element_type> {
  using base_t = std::experimental::accessor_basic<
      typename ViewTraits::mdspan_element_type>;
  using base_t::base_t;

  // Base versions of CRTP-like call for composition purposes
  template <class Self, class Ptr>
  KOKKOS_FORCEINLINE_FUNCTION static auto _do_access(Self const& self, Ptr p,
                                                     ptrdiff_t i) {
    return self.base_t::access(p, i);
  }

  template <class Self, class Ptr>
  KOKKOS_FORCEINLINE_FUNCTION static auto _do_offset(Self const& self, Ptr p,
                                                     ptrdiff_t i) {
    return self.base_t::offset(p, i);
  }
};

//----------------------------------------------------------------------------
// <editor-fold desc="Handle Managed/Unmanaged"> {{{2

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

  // TODO @View constructors for this thing

 private:
  tracker_type m_tracker;
};

// The unmanaged memory case doesn't add any members for now

// </editor-fold> end Handle Managed/Unmanaged }}}2
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// <editor-fold desc="Handle RandomAccess"> {{{2

// RandomAccess case
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::RandomAccess,
                                         /* FlagIsSet = */ true>
    : BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags,
          next_flag_v<MemoryTraitsFlags, MemoryTraitsFlags::RandomAccess>> {
 private:
  using base_t = BuildAccessorForMemoryTraitsFlags<
      ViewTraits, Flags,
      next_flag_v<MemoryTraitsFlags, MemoryTraitsFlags::RandomAccess>>;

 public:
  using base_t::base_t;

  // TODO handle RandomAccess special handle case
};

// </editor-fold> end Handle RandomAccess }}}2
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// <editor-fold desc="Handle Atomic"> {{{2

// Atomic Case
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::Atomic,
                                         /* FlagIsSet = */ true>
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Atomic, true>> {
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Atomic, true>>;
  using base_t::base_t;
  using reference = AtomicReference<typename base_t::pointer>;

  // CRTP-like call for composition purposes
  template <class Self, class Ptr>
  KOKKOS_FORCEINLINE_FUNCTION static auto _do_access(Self const& self, Ptr p,
                                                     ptrdiff_t i) {
    return reference(self.offset(p, i), AtomicViewConstTag{});
  }
};

// </editor-fold> end Handle Atomic }}}2
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// <editor-fold desc="Handle Restrict"> {{{2

// Non-aliasing memory case
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::Restrict,
                                         /* FlagIsSet = */ true>
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Restrict, true>> {
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Restrict, true>>;
  using base_t::base_t;

  using reference = add_restrict_t<typename base_t::reference>;
  using pointer   = add_restrict_t<typename base_t::pointer>;
};

// </editor-fold> end Handle Restrict }}}2
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// <editor-fold desc="Handle Align"> {{{2

// Aligned memory case
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::Aligned,
                                         /* FlagIsSet = */ true>
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Aligned>> {
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, MemoryTraitsFlags::Aligned>>;
  using base_t::base_t;
  using pointer = align_ptr_t<typename base_t::pointer>;
};

// </editor-fold> end Handle Align }}}2
//----------------------------------------------------------------------------

template <class T, unsigned Flags>
struct AccessorForMemoryTraitsFlags<T, MemoryTraits<Flags>>
    : BuildAccessorForMemoryTraitsFlags<T, Flags,
                                        MemoryTraitsFlags::Unmanaged> {
  using base_t =
      BuildAccessorForMemoryTraitsFlags<T, Flags, MemoryTraitsFlags::Unmanaged>;

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto access(typename base_t::pointer p,
                        ptrdiff_t i) const noexcept {
    base_t::_do_access(*this, p, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr auto offset(typename base_t::pointer p,
                        ptrdiff_t i) const noexcept {
    base_t::_do_offset(*this, p, i);
  }
};

template <class T, class MemTraits>
struct MDSpanAccessorFromKokkosMemoryTraits
    : AccessorForMemoryTraitsFlags<T, MemTraits> {
 private:
  using base_t = AccessorForMemoryTraitsFlags<T, MemTraits>;

 public:
  using base_t::base_t;
};

// </editor-fold> end MDSpanAccessorFromKokkosMemoryTraits }}}1
//==============================================================================

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_MDSPANACCESSOR_HPP
