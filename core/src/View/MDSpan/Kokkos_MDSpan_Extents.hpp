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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_DEPRECATED_CODE_3
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#else
KOKKOS_IMPL_WARNING("Including non-public Kokkos header files is not allowed.")
#endif
#endif

#ifndef KOKKOS_EXPERIMENTAL_MDSPAN_EXTENTS_HPP
#define KOKKOS_EXPERIMENTAL_MDSPAN_EXTENTS_HPP

#include <experimental/mdspan>

namespace Kokkos {
namespace Impl {
template <class DataType>
struct ViewArrayAnalysis;

template <std::size_t... Vals>
struct ViewDimension;

template <class T, class Dim>
struct ViewDataType;
}  // namespace Impl

namespace Experimental::Impl {

/*
 * A few things to note --
 * - mdspan allows for 0-rank extents similarly to View, so we don't need
 * special handling of this case
 * - View dynamic dimensions must be appear before static dimensions. This isn't
 * a requirement in mdspan but won't cause an issue here
 */

template <std::size_t N>
struct ExtentFromDimension {
  static constexpr std::size_t value = N;
};

// Kokkos uses a dimension of '0' to denote a dynamic dimension.
template <>
struct ExtentFromDimension<std::size_t{0}> {
  static constexpr std::size_t value = std::experimental::dynamic_extent;
};

template <std::size_t N>
struct DimensionFromExtent {
  static constexpr std::size_t value = N;
};

template <>
struct DimensionFromExtent<std::experimental::dynamic_extent> {
  static constexpr std::size_t value = std::size_t{0};
};

template <class Dimension, class Indices>
struct ExtentsFromDimension;

template <class Dimension, std::size_t... Indices>
struct ExtentsFromDimension<Dimension, std::index_sequence<Indices...>> {
  using type           = std::experimental::extents<
      std::size_t,  // Mirrors Kokkos::View's size type
      ExtentFromDimension<Dimension::static_extent(Indices)>::value...>;
};

template <class Extents, class Indices>
struct DimensionsFromExtent;

template <class Extents, std::size_t... Indices>
struct DimensionsFromExtent<Extents, std::index_sequence<Indices...>> {
  using type         = ::Kokkos::Impl::ViewDimension<
      DimensionFromExtent<Extents::static_extent(Indices)>::value...>;
};

template <class DataType>
struct ExtentsFromDataType {
  using array_analysis = ::Kokkos::Impl::ViewArrayAnalysis<DataType>;
  using dimension_type = typename array_analysis::dimension;

  using type = typename ExtentsFromDimension<
      dimension_type, std::make_index_sequence<dimension_type::rank>>::type;
};

template <class T, class Extents>
struct DataTypeFromExtents {
  using extents_type   = Extents;
  using dimension_type = typename DimensionsFromExtent<
      Extents, std::make_index_sequence<extents_type::rank()>>::type;

  // Will cause a compile error if it is malformed (i.e. dynamic after static)
  using type = typename ::Kokkos::Impl::ViewDataType<T, dimension_type>::type;
};
}  // namespace Experimental::Impl
}  // namespace Kokkos

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_EXTENTS_HPP
