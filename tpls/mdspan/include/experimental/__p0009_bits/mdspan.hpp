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

#include "default_accessor.hpp"
#include "layout_right.hpp"
#include "extents.hpp"
#include "trait_backports.hpp"
#include "compressed_pair.hpp"

namespace std {
namespace experimental {

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = layout_right,
  class AccessorPolicy = default_accessor<ElementType>
>
class mdspan
{
private:
  static_assert(detail::__is_extents_v<Extents>, "std::experimental::mdspan's Extents template parameter must be a specialization of std::experimental::extents.");

  // Workaround for non-deducibility of the index sequence template parameter if it's given at the top level
  template <class>
  struct __deduction_workaround;

  template <size_t... Idxs>
  struct __deduction_workaround<index_sequence<Idxs...>>
  {
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    size_t __size(mdspan const& __self) noexcept {
      return _MDSPAN_FOLD_TIMES_RIGHT((__self.__mapping_ref().extents().template __extent<Idxs>()), /* * ... * */ 1);
    }
    template <class ReferenceType, class SizeType, size_t N>
    MDSPAN_FORCE_INLINE_FUNCTION static constexpr
    ReferenceType __callop(mdspan const& __self, const array<SizeType, N>& indices) noexcept {
      return __self.__accessor_ref().access(__self.__ptr_ref(), __self.__mapping_ref()(indices[Idxs]...));
    }
  };

public:

  //--------------------------------------------------------------------------------
  // Domain and codomain types

  using extents_type = Extents;
  using layout_type = LayoutPolicy;
  using accessor_type = AccessorPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;
  using value_type = remove_cv_t<element_type>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = typename accessor_type::pointer;
  using reference = typename accessor_type::reference;

private:

  // Can't use defaulted parameter in the __deduction_workaround template because of a bug in MSVC warning C4348.
  using __impl = __deduction_workaround<make_index_sequence<extents_type::rank()>>;

  using __map_acc_pair_t = detail::__compressed_pair<mapping_type, accessor_type>;

public:

  //--------------------------------------------------------------------------------
  // [mdspan.basic.cons], mdspan constructors, assignment, and destructor

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan() = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan(const mdspan&) = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mdspan(mdspan&&) = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, SizeTypes...) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type) &&
      _MDSPAN_TRAIT(is_default_constructible, accessor_type)
    )
  )
  MDSPAN_INLINE_FUNCTION
  explicit constexpr mdspan(pointer p, SizeTypes... dynamic_extents)
    // TODO @proposal-bug shouldn't I be allowed to do `move(p)` here?
    : __members(p, __map_acc_pair_t(mapping_type(extents_type(dynamic_extents...)), accessor_type()))
  { }

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, array<SizeType, N>) &&
      _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type) &&
      _MDSPAN_TRAIT(is_default_constructible, accessor_type)
    )
  )
  MDSPAN_CONDITIONAL_EXPLICIT(N != extents_type::rank_dynamic())
  MDSPAN_INLINE_FUNCTION
  constexpr mdspan(pointer p, const array<SizeType, N>& dynamic_extents)
    : __members(p, __map_acc_pair_t(mapping_type(extents_type(dynamic_extents)), accessor_type()))
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdspan, (pointer p, const extents_type& exts), ,
    /* requires */ (_MDSPAN_TRAIT(is_default_constructible, accessor_type) &&
                    _MDSPAN_TRAIT(is_constructible, mapping_type, extents_type))
  ) : __members(p, __map_acc_pair_t(mapping_type(exts), accessor_type()))
  { }

  MDSPAN_FUNCTION_REQUIRES(
    (MDSPAN_INLINE_FUNCTION constexpr),
    mdspan, (pointer p, const mapping_type& m), ,
    /* requires */ (_MDSPAN_TRAIT(is_default_constructible, accessor_type))
  ) : __members(p, __map_acc_pair_t(m, accessor_type()))
  { }

  MDSPAN_INLINE_FUNCTION
  constexpr mdspan(pointer p, const mapping_type& m, const accessor_type& a)
    : __members(p, __map_acc_pair_t(m, a))
  { }

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class OtherAccessor,
    /* requires */ (
      _MDSPAN_TRAIT(is_constructible, mapping_type, typename OtherLayoutPolicy::template mapping<OtherExtents>) &&
      _MDSPAN_TRAIT(is_constructible, accessor_type, OtherAccessor) &&
      _MDSPAN_TRAIT(is_constructible, pointer, typename OtherAccessor::pointer) &&
      _MDSPAN_TRAIT(is_constructible, extents_type, OtherExtents)
    )
  )
  MDSPAN_INLINE_FUNCTION
  constexpr mdspan(const mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>& other)
    : __members(other.__ptr_ref(), __map_acc_pair_t(other.__mapping_ref(), other.__accessor_ref()))
  { }

  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~mdspan() = default;

  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(const mdspan&) = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED mdspan& operator=(mdspan&&) = default;


  //--------------------------------------------------------------------------------
  // [mdspan.basic.mapping], mdspan mapping domain multidimensional index to access codomain element

  #if MDSPAN_USE_BRACKET_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](SizeTypes... indices) const noexcept
  {
    return __accessor_ref().access(__ptr_ref(), __mapping_ref()(size_type(indices)...));
  }
  #endif

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](const array<SizeType, N>& indices) const noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }

  #if !MDSPAN_USE_BRACKET_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class Index,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, Index, size_type) &&
      extents_type::rank() == 1
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator[](Index idx) const noexcept
  {
    return __accessor_ref().access(__ptr_ref(), __mapping_ref()(size_type(idx)));
  }
  #endif

  #if MDSPAN_USE_PAREN_OPERATOR
  MDSPAN_TEMPLATE_REQUIRES(
    class... SizeTypes,
    /* requires */ (
      _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_convertible, SizeTypes, size_type) /* && ... */) &&
      extents_type::rank() == sizeof...(SizeTypes)
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(SizeTypes... indices) const noexcept
  {
    return __accessor_ref().access(__ptr_ref(), __mapping_ref()(size_type(indices)...));
  }

  MDSPAN_TEMPLATE_REQUIRES(
    class SizeType, size_t N,
    /* requires */ (
      _MDSPAN_TRAIT(is_convertible, SizeType, size_type) &&
      N == extents_type::rank()
    )
  )
  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference operator()(const array<SizeType, N>& indices) const noexcept
  {
    return __impl::template __callop<reference>(*this, indices);
  }
  #endif

  MDSPAN_INLINE_FUNCTION constexpr
  accessor_type accessor() const { return __accessor_ref(); };

  //--------------------------------------------------------------------------------
  // [mdspan.basic.domobs], mdspan observers of the domain multidimensional index space

  MDSPAN_INLINE_FUNCTION static constexpr size_t rank() noexcept { return extents_type::rank(); }
  MDSPAN_INLINE_FUNCTION static constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  MDSPAN_INLINE_FUNCTION static constexpr size_type static_extent(size_t r) noexcept { return extents_type::static_extent(r); }

  MDSPAN_INLINE_FUNCTION constexpr extents_type extents() const noexcept { return __mapping_ref().extents(); };
  MDSPAN_INLINE_FUNCTION constexpr size_type extent(size_t r) const noexcept { return __mapping_ref().extents().extent(r); };
  MDSPAN_INLINE_FUNCTION constexpr size_type size() const noexcept {
    return __impl::__size(*this);
  };

  MDSPAN_INLINE_FUNCTION constexpr pointer data() const noexcept { return __ptr_ref(); };

  //--------------------------------------------------------------------------------
  // [mdspan.basic.obs], mdspan observers of the mapping

  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return mapping_type::is_always_unique(); };
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_contiguous() noexcept { return mapping_type::is_always_contiguous(); };
  MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return mapping_type::is_always_strided(); };

  MDSPAN_INLINE_FUNCTION constexpr mapping_type mapping() const noexcept { return __mapping_ref(); };
  MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return __mapping_ref().is_unique(); };
  MDSPAN_INLINE_FUNCTION constexpr bool is_contiguous() const noexcept { return __mapping_ref().is_contiguous(); };
  MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return __mapping_ref().is_strided(); };
  MDSPAN_INLINE_FUNCTION constexpr size_type stride(size_t r) const { return __mapping_ref().stride(r); };

private:

  detail::__compressed_pair<pointer, __map_acc_pair_t> __members{};

  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 pointer& __ptr_ref() noexcept { return __members.__first(); }
  MDSPAN_FORCE_INLINE_FUNCTION constexpr pointer const& __ptr_ref() const noexcept { return __members.__first(); }
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 mapping_type& __mapping_ref() noexcept { return __members.__second().__first(); }
  MDSPAN_FORCE_INLINE_FUNCTION constexpr mapping_type const& __mapping_ref() const noexcept { return __members.__second().__first(); }
  MDSPAN_FORCE_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 accessor_type& __accessor_ref() noexcept { return __members.__second().__second(); }
  MDSPAN_FORCE_INLINE_FUNCTION constexpr accessor_type const& __accessor_ref() const noexcept { return __members.__second().__second(); }

  template <class, class, class, class>
  friend class mdspan;

};

#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
#ifdef _MDSPAN_USE_P2554
MDSPAN_TEMPLATE_REQUIRES(
  class ElementType, class... SizeTypes,
  /* requires */ _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_integral, SizeTypes) /* && ... */) &&
  (sizeof...(SizeTypes) > 0)
)
mdspan(ElementType*, SizeTypes...)
  -> mdspan<ElementType, ::std::experimental::dextents<sizeof...(SizeTypes)>>;

MDSPAN_TEMPLATE_REQUIRES(
  class Pointer,
  (!_MDSPAN_TRAIT(is_array, Pointer))
)
mdspan(const Pointer&) -> mdspan<std::remove_pointer_t<Pointer>, extents<>>;

MDSPAN_TEMPLATE_REQUIRES(
  class CArray,
  _MDSPAN_TRAIT(is_array, CArray)
)
mdspan(CArray&) -> mdspan<std::remove_all_extents_t<CArray>, extents<::std::extent_v<CArray,0>>>;

#else

MDSPAN_TEMPLATE_REQUIRES(
  class ElementType, class... SizeTypes,
  /* requires */ _MDSPAN_FOLD_AND(_MDSPAN_TRAIT(is_integral, SizeTypes) /* && ... */)
)
mdspan(ElementType*, SizeTypes...)
  -> mdspan<ElementType, ::std::experimental::dextents<sizeof...(SizeTypes)>>;

#endif

template <class ElementType, class SizeType, size_t N>
mdspan(ElementType*, const ::std::array<SizeType, N>&)
  -> mdspan<ElementType, ::std::experimental::dextents<N>>;

// This one is necessary because all the constructors take `pointer`s, not
// `ElementType*`s, and `pointer` is taken from `accessor_type::pointer`, which
// seems to throw off automatic deduction guides.
template <class ElementType, size_t... ExtentsPack>
mdspan(ElementType*, const extents<ExtentsPack...>&)
  -> mdspan<ElementType, ::std::experimental::extents<ExtentsPack...>>;

template <class ElementType, class MappingType>
mdspan(ElementType*, const MappingType&)
  -> mdspan<ElementType, typename MappingType::extents_type, typename MappingType::layout_type>;

template <class MappingType, class AccessorType>
mdspan(const typename AccessorType::pointer, const MappingType&, const AccessorType&)
  -> mdspan<typename AccessorType::element_type, typename MappingType::extents_type, typename MappingType::layout_type, AccessorType>;
#endif

} // end namespace experimental
} // end namespace std
