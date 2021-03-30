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

#ifndef KOKKOS_KOKKOS_ACCESSOR_CONVERTIBILITY_HPP
#define KOKKOS_KOKKOS_ACCESSOR_CONVERTIBILITY_HPP

#include <Kokkos_Macros.hpp>

#include <View/Accessor/Kokkos_Accessor_fwd.hpp>
#include <impl/Kokkos_Utilities.hpp>  // next_flag_v

#include <type_traits>

namespace Kokkos {
namespace Impl {

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
 private:
  using base_t = BuildAccessorForMemoryTraitsFlags<
      ViewTraits, Flags, next_flag_v<MemoryTraitsFlags, CurrentFlag>>;
  template <class T, unsigned OtherFlags>
  using analogous_opposite_flag_accessor =
      BuildAccessorForMemoryTraitsFlags<T, OtherFlags, CurrentFlag, !FlagIsSet>;

 public:
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

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_ACCESSOR_CONVERTIBILITY_HPP
