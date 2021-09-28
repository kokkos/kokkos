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

#ifndef KOKKOS_KOKKOS_EXTRACTEXTENTS_HPP
#define KOKKOS_KOKKOS_EXTRACTEXTENTS_HPP

#include <Kokkos_Macros.hpp>

#include <experimental/mdspan>

namespace Kokkos {
namespace Impl {

//==============================================================================
// <editor-fold desc="ExtractExtents"> {{{1

template <class T, std::size_t... Exts>
struct ExtractExtents {
  using value_type   = T;
  using extents_type = std::experimental::extents<Exts...>;
};

template <class T, std::size_t... Exts>
struct ExtractExtents<T*, Exts...>
    : ExtractExtents<T, std::experimental::dynamic_extent, Exts...> {};

template <class T, std::size_t N, std::size_t... Exts>
struct ExtractExtents<T[N], Exts...>
    : ExtractExtents<T, size_t{N}, Exts...> {};

// </editor-fold> end ExtractExtents }}}1
//==============================================================================

//==============================================================================
// <editor-fold desc="DataTypeFromExtents"> {{{1

template <class T, class Extents>
struct DataTypeFromExtents;

template <class T, std::size_t Ext, std::size_t... Exts>
struct DataTypeFromExtents<T, std::experimental::extents<Ext, Exts...>>
    : DataTypeFromExtents<T[std::size_t{Ext}],
                          std::experimental::extents<Exts...>> {};

template <class T, std::size_t... Exts>
struct DataTypeFromExtents<
    T, std::experimental::extents<std::experimental::dynamic_extent, Exts...>>
    : DataTypeFromExtents<T*, std::experimental::extents<Exts...>> {};

template <class T>
struct DataTypeFromExtents<T, std::experimental::extents<>> {
  using type = T;
};

// </editor-fold> end DataTypeFromExtents }}}1
//==============================================================================

template <class>
struct RemoveFirstExtent;

template <std::size_t Extent, std::size_t... Extents>
struct RemoveFirstExtent<std::experimental::extents<Extent, Extents...>>
    : identity<std::experimental::extents<Extents...>> {};

template <class>
struct FirstExtentOnly;

template <std::size_t Extent, std::size_t... Extents>
struct FirstExtentOnly<std::experimental::extents<Extent, Extents...>>
    : identity<std::experimental::extents<Extent>> {};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_EXTRACTEXTENTS_HPP
