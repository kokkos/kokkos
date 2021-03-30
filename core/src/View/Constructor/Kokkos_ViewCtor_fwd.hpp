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

#ifndef KOKKOS_KOKKOS_VIEWCTOR_FWD_HPP
#define KOKKOS_KOKKOS_VIEWCTOR_FWD_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core_fwd.hpp>

namespace Kokkos {
namespace Impl {

template <class TraitSpec, class Trait, class Enable = void>
struct ViewCtorTraitMatcher;

struct WithoutInitializingViewCtorTrait;
struct AllowPaddingViewCtorTrait;
struct LabelViewCtorTrait;
struct PointerViewCtorTrait;
struct ExecutionSpaceViewCtorTrait;
struct MemorySpaceViewCtorTrait;

// clang-format off
using view_constructor_trait_specifications =
  type_list<
    WithoutInitializingViewCtorTrait,
    AllowPaddingViewCtorTrait,
    LabelViewCtorTrait,
    PointerViewCtorTrait,
    ExecutionSpaceViewCtorTrait,
    MemorySpaceViewCtorTrait
  >;
// clang-format on

// used to avoid conflicts with rule of 6 constructors
struct view_ctor_trait_ctor_tag {};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWCTOR_FWD_HPP
