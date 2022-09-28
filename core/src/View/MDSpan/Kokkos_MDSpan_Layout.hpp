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

#ifndef KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
#define KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP

#include <Kokkos_Core_fwd.hpp>
#include "Kokkos_MDSpan_Extents.hpp"
#include <impl/Kokkos_ViewMapping_fwd.hpp>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

namespace Kokkos {
struct LayoutLeft;
}

namespace Kokkos::Experimental {
template <class Extents>
class MDSpanLayoutLeft {
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = LayoutLeft;

  constexpr MDSpanLayoutLeft(const MDSpanLayoutLeft &)     = default;
  constexpr MDSpanLayoutLeft(MDSpanLayoutLeft &&) noexcept = default;

  constexpr MDSpanLayoutLeft &operator=(const MDSpanLayoutLeft &)     = default;
  constexpr MDSpanLayoutLeft &operator=(MDSpanLayoutLeft &&) noexcept = default;

  template <class OtherExtents>
  friend constexpr bool operator==(
      const MDSpanLayoutLeft &,
      const MDSpanLayoutLeft<OtherExtents> &) noexcept;

  constexpr const extents_type &extents() const noexcept { return m_extents; }

  template <class... Indices>
  constexpr index_type operator()(Indices... idxs) const noexcept {
    // We don't define dimension counts past 8 but hopefully this should
    // give a better error rather than a huge list of overload candidates
    // failing
    static_assert(sizeof...(Indices) <= 8,
                  "Kokkos dimensions are limited to <= 8");
    return m_offset(static_cast<index_type>(idxs)...);
  }

  constexpr index_type required_span_size() const noexcept {
    return m_offset.span();
  }

  constexpr bool is_unique() const noexcept { return true; }
  constexpr bool is_exhaustive() const noexcept {
    return m_offset.span_is_contiguous();
  }

  constexpr bool is_strided() const noexcept {
    // Note: LayoutLeft with >= 0 rank and no dynamic ranks has the comment that it has no padding/striding
    // However according to mdspan's definition of is_strided it still counts as strided
    return true;
  }

  constexpr index_type stride( rank_type r ) const noexcept {
    if ( r == 0 )
      return m_offset.stride_0();
    index_type ret = m_offset.stride_1();
    for ( rank_type i = 1; i < r; ++i )
      ret *= m_extents.extent( i );
    return ret;
  }

  static constexpr bool is_always_unique() noexcept { return true; }
  static constexpr bool is_always_exhaustive() noexcept { return true; }
  static constexpr bool is_always_strided() noexcept { return true; }

 private:
  using dimension_type =
      typename Impl::DimensionsFromExtents<extents_type>::type;
  using offset_type = ::Kokkos::Impl::ViewOffset<dimension_type, layout_type>;

  // This assumption should hold since we are using size_t everywhere in Kokkos
  // but if we ever change this the assert should fail
  static_assert(std::is_same_v<typename offset_type::size_type, index_type>);

  extents_type m_extents;
  offset_type m_offset;
};
}  // namespace Kokkos::Experimental

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN

#endif  // KOKKOS_EXPERIMENTAL_MDSPAN_LAYOUT_HPP
