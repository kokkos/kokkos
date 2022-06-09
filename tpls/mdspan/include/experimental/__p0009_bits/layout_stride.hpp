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
#include "static_array.hpp"
#include "extents.hpp"
#include "trait_backports.hpp"
#include "compressed_pair.hpp"

#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
#  include "no_unique_address.hpp"
#endif

#include <algorithm>
#include <numeric>
#include <array>

namespace std {
namespace experimental {

struct layout_stride {
  template <class Extents>
  class mapping
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    : private detail::__no_unique_address_emulation<
        detail::__compressed_pair<
          Extents,
          ::std::experimental::dextents<Extents::rank()>
        >
      >
#endif
  {
  public:
    // This could be a `requires`, but I think it's better and clearer as a `static_assert`.
    static_assert(detail::__is_extents_v<Extents>, "std::experimental::layout_stride::mapping must be instantiated with a specialization of std::experimental::extents.");

    using size_type = typename Extents::size_type;
    using extents_type = Extents;

    using layout_type = layout_stride;

  private:

    //----------------------------------------------------------------------------

    using __strides_storage_t = ::std::experimental::dextents<Extents::rank()>;
    using __member_pair_t = detail::__compressed_pair<extents_type, __strides_storage_t>;

#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    _MDSPAN_NO_UNIQUE_ADDRESS __member_pair_t __members;
#else
    using __base_t = detail::__no_unique_address_emulation<__member_pair_t>;
#endif

    MDSPAN_FORCE_INLINE_FUNCTION constexpr __strides_storage_t const&
    __strides_storage() const noexcept {
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      return __members.__second();
#else
      return this->__base_t::__ref().__second();
#endif
    }
    MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 __strides_storage_t&
    __strides_storage() noexcept {
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      return __members.__second();
#else
      return this->__base_t::__ref().__second();
#endif
    }

    //----------------------------------------------------------------------------

    template <class>
    friend class mapping;

    //----------------------------------------------------------------------------

    // Workaround for non-deducibility of the index sequence template parameter if it's given at the top level
    template <class>
    struct __deduction_workaround;

    template <size_t... Idxs>
    struct __deduction_workaround<index_sequence<Idxs...>>
    {
      template <class OtherExtents>
      MDSPAN_INLINE_FUNCTION
      static constexpr bool _eq_impl(mapping const& self, mapping<OtherExtents> const& other) noexcept {
        return _MDSPAN_FOLD_AND((self.template __stride<Idxs>() == other.template __stride<Idxs>()) /* && ... */);
      }
      template <class OtherExtents>
      MDSPAN_INLINE_FUNCTION
      static constexpr bool _not_eq_impl(mapping const& self, mapping<OtherExtents> const& other) noexcept {
        return _MDSPAN_FOLD_OR((self.template __stride<Idxs>() != other.template __stride<Idxs>()) /* || ... */);
      }

      template <class... Integral>
      MDSPAN_FORCE_INLINE_FUNCTION
      static constexpr size_t _call_op_impl(mapping const& self, Integral... idxs) noexcept {
        return _MDSPAN_FOLD_PLUS_RIGHT((idxs * self.template __stride<Idxs>()), /* + ... + */ 0);
      }

      MDSPAN_INLINE_FUNCTION
      static constexpr size_t _req_span_size_impl(mapping const& self) noexcept {
        // assumes no negative strides; not sure if I'm allowed to assume that or not
        return __impl::_call_op_impl(self, (self.extents().template __extent<Idxs>() - 1)...) + 1;
      }

      template<class OtherMapping>
      static constexpr __strides_storage_t fill_strides(const OtherMapping& map) {
        return __strides_storage_t(map.stride(Idxs)...);
      }
    };

    // Can't use defaulted parameter in the __deduction_workaround template because of a bug in MSVC warning C4348.
    using __impl = __deduction_workaround<make_index_sequence<Extents::rank()>>;


    //----------------------------------------------------------------------------

#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
    MDSPAN_INLINE_FUNCTION constexpr explicit
    mapping(__member_pair_t&& __m) : __members(::std::move(__m)) {}
#else
    MDSPAN_INLINE_FUNCTION constexpr explicit
    mapping(__base_t&& __b) : __base_t(::std::move(__b)) {}
#endif

    //----------------------------------------------------------------------------

  public: // (but not really)

    template <size_t R>
    MDSPAN_INLINE_FUNCTION
    constexpr size_t __stride() const noexcept {
      return __strides_storage().extent(R);
    }

    template <size_t... Idxs>
    MDSPAN_INLINE_FUNCTION
    constexpr array< size_t, Extents::rank() > __strides(std::index_sequence<Idxs...>) const noexcept {
      return {__strides_storage().template __extent<Idxs>()...};
    }

    MDSPAN_INLINE_FUNCTION
    static constexpr mapping
    __make_mapping(
      detail::__extents_to_partially_static_sizes_t<Extents>&& __exts,
      detail::__extents_to_partially_static_sizes_t<
        ::std::experimental::dextents<Extents::rank()>>&& __strs
    ) noexcept {
      // call the private constructor we created for this purpose
      return mapping(
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        __base_t{
#endif
          __member_pair_t(
            extents_type::__make_extents_impl(::std::move(__exts)),
            __strides_storage_t{::std::move(__strs)}
          )
#if !defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#endif
      );
    }

  public:

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED
    mapping& operator=(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED
    mapping& operator=(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED ~mapping() noexcept = default;

    template<class IntegralType>
    MDSPAN_INLINE_FUNCTION
    constexpr
    mapping(
      Extents const& e,
      ::std::array<IntegralType, Extents::rank()> const& strides
    ) noexcept
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          e, __strides_storage_t{strides}
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    { }

    template<class OtherExtents>
    MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible<OtherExtents, Extents>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION
    constexpr
    mapping(
      const mapping<OtherExtents>& rhs
    ) noexcept
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          rhs.extents(), __strides_storage_t{rhs.__strides_storage()}
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    { }


    MDSPAN_TEMPLATE_REQUIRES(
      class OtherMapping,
      /* requires */ (
        _MDSPAN_TRAIT(is_constructible, Extents, typename OtherMapping::extents_type) &&
        _MDSPAN_TRAIT(is_same, typename OtherMapping::layout_type::template mapping<typename OtherMapping::extents_type>, OtherMapping) &&
        OtherMapping::is_always_unique() &&
        OtherMapping::is_always_strided()
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!is_convertible<typename OtherMapping::extents_type, Extents>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14
    mapping(OtherMapping const& other) noexcept // NOLINT(google-explicit-constructor)
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      : __members{
#else
      : __base_t(__base_t{__member_pair_t(
#endif
          other.extents(), __strides_storage_t(__impl::fill_strides(other))
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
        }
#else
        )})
#endif
    {}

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION constexpr extents_type extents() const noexcept {
#if defined(_MDSPAN_USE_ATTRIBUTE_NO_UNIQUE_ADDRESS)
      return __members.__first();
#else
      return this->__base_t::__ref().__first();
#endif
    };

    MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return true; }
    MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 bool is_contiguous() const noexcept {
      // TODO @testing test layout_stride is_contiguous()
// FIXME CUDA
#ifdef __CUDA_ARCH__
      return false;
#else
      auto rem = array<size_t, Extents::rank()>{ };
      std::iota(rem.begin(), rem.end(), size_t(0));
      auto next_idx_iter = std::find_if(
        rem.begin(), rem.end(),
        [&](size_t i) { return this->stride(i) == 1;  }
      );
      if(next_idx_iter != rem.end()) {
        size_t prev_stride_times_prev_extent =
          this->extents().extent(*next_idx_iter) * this->stride(*next_idx_iter);
        // "remove" the index
        constexpr auto removed_index_sentinel = static_cast<size_t>(-1);
        *next_idx_iter = removed_index_sentinel;
        int found_count = 1;
        while (found_count != Extents::rank()) {
          next_idx_iter = std::find_if(
            rem.begin(), rem.end(),
            [&](size_t i) {
              return i != removed_index_sentinel
                && this->extents().extent(i) == prev_stride_times_prev_extent;
            }
          );
          if (next_idx_iter != rem.end()) {
            // "remove" the index
            *next_idx_iter = removed_index_sentinel;
            ++found_count;
            prev_stride_times_prev_extent = stride(*next_idx_iter) * this->extents().extent(*next_idx_iter);
          } else { break; }
        }
        return found_count == Extents::rank();
      }
      return false;
#endif
    }
    MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return true; }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept {
      return false;
    }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }

    MDSPAN_TEMPLATE_REQUIRES(
      class... Indices,
      /* requires */ (
        sizeof...(Indices) == Extents::rank() &&
        _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_constructible, Indices, size_t) /*&& ...*/)
      )
    )
    MDSPAN_FORCE_INLINE_FUNCTION
    constexpr size_t operator()(Indices... idxs) const noexcept {
      return __impl::_call_op_impl(*this, idxs...);
    }

    MDSPAN_INLINE_FUNCTION
    constexpr size_t stride(size_t r) const noexcept {
      return __strides_storage().extent(r);
    }

    MDSPAN_INLINE_FUNCTION
    constexpr array< size_t, Extents::rank() > strides() const noexcept {
      return __strides(std::make_index_sequence<Extents::rank()>());
    }

    MDSPAN_INLINE_FUNCTION
    constexpr size_t required_span_size() const noexcept {
      size_t span_size = 1;
      for(unsigned r = 0; r < Extents::rank(); r++) {
        // Return early if any of the extents are zero
        if(extents().extent(r)==0) return 0;
        span_size = std::max(span_size, extents().extent(r) * __strides_storage().extent(r));
      }
      return span_size;
    }

    template<class OtherExtents>
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      return __impl::_eq_impl(lhs, rhs);
    }

#if MDSPAN_HAS_CXX_20
    template<class OtherExtents>
    MDSPAN_INLINE_FUNCTION
    friend constexpr bool operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept {
      return __impl::_not_eq_impl(lhs, rhs);
    }
#endif

  };
};

} // end namespace experimental
} // end namespace std
