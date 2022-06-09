/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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

#pragma once

#include "macros.hpp"
#include "trait_backports.hpp"
#include "extents.hpp"
#include <stdexcept>

namespace std {
namespace experimental {

//==============================================================================
struct layout_left;
struct layout_stride;

struct layout_right {
  template <class Extents>
  class mapping {
  private:

    static_assert(detail::__is_extents_v<Extents>, "std::experimental::layout_right::mapping must be instantiated with a specialization of std::experimental::extents.");

    template <class>
    friend class mapping;

    // i0+(i1 + E(1)*(i2 + E(2)*i3))
    template <size_t r, size_t Rank>
    struct __rank_count {};

    template <size_t r, size_t Rank, class I, class... Indices>
    constexpr size_t __compute_offset(
      size_t offset, __rank_count<r,Rank>, const I& i, Indices... idx) const {
      return __compute_offset(offset * __extents.template __extent<r>() + i,__rank_count<r+1,Rank>(),  idx...);
    }

    template<class I, class ... Indices>
    constexpr size_t __compute_offset(
      __rank_count<0,Extents::rank()>, const I& i, Indices... idx) const {
      return __compute_offset(static_cast<size_t>(i),__rank_count<1,Extents::rank()>(),idx...);
    }

    constexpr size_t __compute_offset(size_t offset, __rank_count<Extents::rank(), Extents::rank()>) const {
      return offset;
    }

    constexpr size_t __compute_offset(__rank_count<0,0>) const { return 0; }

  public:

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED ~mapping() noexcept = default;


    using layout_type = layout_right;
    using extents_type = Extents;
    using size_type = typename Extents::size_type;

    constexpr mapping(Extents const& __exts) noexcept
      :__extents(__exts)
    { }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        _MDSPAN_TRAIT(is_constructible, Extents, OtherExtents)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible<OtherExtents, Extents>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    { }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherMapping,
      /* requires */ (
        _MDSPAN_TRAIT(is_constructible, Extents, typename OtherMapping::extents_type) &&
        _MDSPAN_TRAIT(is_same, typename OtherMapping::layout_type, layout_left) &&
        _MDSPAN_TRAIT(is_same, typename OtherMapping::layout_type::template mapping<typename OtherMapping::extents_type>, OtherMapping) &&
        (Extents::rank() <= 1)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible<typename OtherMapping::extents_type, Extents>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(OtherMapping const& other) noexcept // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    { }
    MDSPAN_TEMPLATE_REQUIRES(
      class OtherMapping,
      /* requires */ (
        _MDSPAN_TRAIT(is_constructible, Extents, typename OtherMapping::extents_type) &&
        _MDSPAN_TRAIT(is_same, typename OtherMapping::layout_type, layout_stride) &&
        _MDSPAN_TRAIT(is_same, typename OtherMapping::layout_type::template mapping<typename OtherMapping::extents_type>, OtherMapping)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((Extents::rank()!=0))
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(OtherMapping const& other) // NOLINT(google-explicit-constructor)
      :__extents(other.extents())
    {
       #ifndef __CUDA_ARCH__
       size_t stride = 1;
       for(size_type r=__extents.rank(); r>0; r--) {
         if(stride != other.stride(r-1))
           throw std::runtime_error("Assigning layout_stride to layout_right with invalid strides.");
         stride *= __extents.extent(r-1);
       }
       #endif
    }

    //--------------------------------------------------------------------------------

    template <class... Indices>
    constexpr size_type operator()(Indices... idxs) const noexcept {
      return __compute_offset(__rank_count<0, Extents::rank()>(), idxs...);
    }

    constexpr Extents extents() const noexcept {
      return __extents;
    }

    constexpr size_type stride(size_t i) const noexcept {
      size_type value = 1;
      for(size_type r=Extents::rank()-1; r>i; r--) value*=__extents.extent(r);
      return value;
    }

    constexpr size_type required_span_size() const noexcept {
      size_type value = 1;
      for(size_type r=0; r<Extents::rank(); r++) value*=__extents.extent(r);
      return value;
    }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_contiguous() const noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return true; }

    template<class OtherExtents>
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      return lhs.extents() == rhs.extents();
    }

    // In C++ 20 the not equal exists if equal is found
#if MDSPAN_HAS_CXX_20
    template<class OtherExtents>
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      return lhs.extents() != rhs.extents();
    }
#endif

    // Not really public, but currently needed to implement fully constexpr useable submdspan:
    template<size_t N, size_t ... E, size_t ... Idx>
    constexpr size_type __get_stride(std::experimental::extents<E...>,integer_sequence<size_t, Idx...>) const {
      return _MDSPAN_FOLD_TIMES_RIGHT((Idx>N? __extents.template __extent<Idx>():1),1);
    }
    template<size_t N>
    constexpr size_type __stride() const noexcept {
      return __get_stride<N>(__extents, make_index_sequence<extents_type::rank()>());
    }

private:
   _MDSPAN_NO_UNIQUE_ADDRESS Extents __extents{};

  };
};

} // end namespace experimental
} // end namespace std
