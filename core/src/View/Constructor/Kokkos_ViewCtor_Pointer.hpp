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

#ifndef KOKKOS_KOKKOS_VIEWCTOR_POINTER_HPP
#define KOKKOS_KOKKOS_VIEWCTOR_POINTER_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <View/Constructor/Kokkos_ViewCtor_fwd.hpp>

#include <type_traits>  // std::is_pointer

namespace Kokkos {
namespace Impl {

struct PointerViewCtorTrait {
  template <class T>
  using trait_matches_specification = std::is_pointer<T>;
  struct base_traits {
    static constexpr bool has_pointer = false;
  };
  // PointerType is of the form T*
  template <class PointerType, class AnalyzeNextTrait>
  struct mixin_matching_trait : AnalyzeNextTrait {
    using base_t = AnalyzeNextTrait;

    mixin_matching_trait() = default;

    mixin_matching_trait(mixin_matching_trait const&) = default;

    mixin_matching_trait(mixin_matching_trait&&) = default;

    mixin_matching_trait& operator=(mixin_matching_trait const&) = default;

    mixin_matching_trait& operator=(mixin_matching_trait&&) = default;

    ~mixin_matching_trait() = default;

    // We need this for calling `view_wrap` on the device
    KOKKOS_INLINE_FUNCTION
    mixin_matching_trait(view_ctor_trait_ctor_tag const& tag,
                         PointerType const& pointer)
        : m_pointer(pointer) {}

    template <class... OtherProps>
    mixin_matching_trait(view_ctor_trait_ctor_tag const& tag,
                         PointerType const& pointer, OtherProps const&... other)
        : base_t(tag, other...), m_pointer(pointer) {}

    static constexpr bool has_pointer = true;
    using pointer_type                = PointerType;
    KOKKOS_FUNCTION
    constexpr PointerType get_pointer() const { return m_pointer; }

   private:
    PointerType m_pointer;
  };
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWCTOR_POINTER_HPP
