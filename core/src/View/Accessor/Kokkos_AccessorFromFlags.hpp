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

#ifndef KOKKOS_KOKKOS_ACCESSORFROMFLAGS_HPP
#define KOKKOS_KOKKOS_ACCESSORFROMFLAGS_HPP

#include <Kokkos_Macros.hpp>

#include <View/Accessor/Kokkos_Accessor_fwd.hpp>

namespace Kokkos {
namespace Impl {

// default case: don't do anything to the accessor
// Some flags don't need to mixin anything for enablement and/or disablement
// of the flag (i.e., either or both of the false and true FlagIsSet cases
// are unspecialized), so we can just move on to the next flag for those.
// We still want to assume that they're convertible to their negation, so
// we mixin convertibility by default.
template <class ViewTraits, unsigned Flags, MemoryTraitsFlags CurrentFlag,
          bool FlagIsSet /* = Flags bitand CurrentFlag */,
          class Enable /* = void */>
struct BuildAccessorForMemoryTraitsFlags
    : MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, CurrentFlag, FlagIsSet>> {
 private:
  using base_t =
      MixinAccessorFlagConvertibility<BuildAccessorForMemoryTraitsFlags<
          ViewTraits, Flags, CurrentFlag, FlagIsSet>>;

 public:
  using base_t::base_t;
};

// Base case (end of flags)
template <class ViewTraits, unsigned Flags>
struct BuildAccessorForMemoryTraitsFlags<ViewTraits, Flags,
                                         MemoryTraitsFlags::END_OF_FLAGS,
                                         /* FlagIsSet = */ false>
    // Protected to avoid accidental convertibility
    : protected std::experimental::accessor_basic<
          typename ViewTraits::value_type> {
  using base_t = std::experimental::accessor_basic<
      typename ViewTraits::value_type>;
  using base_t::base_t;

  // Allow casting to protected base from within the hierarchy
  template <class, unsigned, MemoryTraitsFlags, bool, class>
  friend struct BuildAccessorForMemoryTraitsFlags;

  // Allow convertibility if it's been passed through this far
  template <class T, unsigned OtherFlags>
  BuildAccessorForMemoryTraitsFlags(
    AccessorForMemoryTraitsFlags<T, MemoryTraits<OtherFlags>> const& other
  ) : base_t(other)
  {}
  template <class T, unsigned OtherFlags>
  BuildAccessorForMemoryTraitsFlags(
      AccessorForMemoryTraitsFlags<T, MemoryTraits<OtherFlags>>&& other
  ) : base_t(std::move(other))
  {}

  // Base versions of CRTP-like call for composition purposes. Basically, if
  // these symbols are not shadowed, they call through to the underlying
  // accessor_basic implementation

  template <class Self, class Ptr>
  KOKKOS_FORCEINLINE_FUNCTION static typename Self::reference crtp_access_mixin(Self const& self,
                                                            Ptr p,
                                                            ptrdiff_t i) {
    return self.base_t::access(p, i);
  }

  template <class Self, class Ptr>
  KOKKOS_FORCEINLINE_FUNCTION static typename Self::reference crtp_offset_mixin(Self const& self,
                                                            Ptr p,
                                                            ptrdiff_t i) {
    return self.base_t::offset(p, i);
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_ACCESSORFROMFLAGS_HPP
