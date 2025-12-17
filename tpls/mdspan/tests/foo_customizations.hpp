//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef _FOO_CUSTOMIZATION_HPP_
#define _FOO_CUSTOMIZATION_HPP_

#include <mdspan/mdspan.hpp>

namespace Foo {
  template<class T>
  struct foo_ptr {
    T* data;
    MDSPAN_INLINE_FUNCTION
    constexpr foo_ptr(T* ptr):data(ptr) {}
  };

  template<class T>
  struct foo_accessor {
    using offset_policy = foo_accessor;
    using element_type = T;
    using reference = T&;
    using data_handle_type = foo_ptr<T>;

    MDSPAN_INLINE_FUNCTION
    constexpr foo_accessor(int* ptr = nullptr) noexcept { flag = ptr; }

    template<class OtherElementType>
    MDSPAN_INLINE_FUNCTION
    constexpr foo_accessor(Kokkos::default_accessor<OtherElementType>) noexcept { flag = nullptr; }

    template<class OtherElementType>
    MDSPAN_INLINE_FUNCTION
    constexpr foo_accessor(foo_accessor<OtherElementType> other) noexcept { flag = other.flag; }


    MDSPAN_INLINE_FUNCTION
    constexpr reference access(data_handle_type p, size_t i) const noexcept {
      return p.data[i];
    }

    MDSPAN_INLINE_FUNCTION
    constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
      return data_handle_type(p.data+i);
    }
    int* flag;

    MDSPAN_INLINE_FUNCTION
    friend constexpr void swap(foo_accessor& x, foo_accessor& y) {
      x.flag[0] = 99;
      y.flag[0] = 77;
      std::swap(x.flag, y.flag);
    }
  };

struct layout_foo {
    template<class Extents>
    class mapping;
};

template <class Extents>
class layout_foo::mapping {
  public:
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = layout_foo;
  private:

    static_assert(Kokkos::detail::impl_is_extents_v<extents_type>,
                  "layout_foo::mapping must be instantiated with a specialization of Kokkos::extents.");
    static_assert(extents_type::rank() < 3, "layout_foo only supports 0D, 1D and 2D");

    template <class>
    friend class mapping;

  public:

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;

    constexpr mapping(extents_type const& exts) noexcept
      :m_extents(exts)
    { }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        MDSPAN_IMPL_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION MDSPAN_IMPL_CONSTEXPR_14
    mapping(mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :m_extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
    }

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        MDSPAN_IMPL_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION MDSPAN_IMPL_CONSTEXPR_14
    mapping(Kokkos::layout_right::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :m_extents(other.extents())
    {}

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        MDSPAN_IMPL_TRAIT(std::is_constructible, extents_type, OtherExtents) &&
        (extents_type::rank() <= 1)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((!std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due to comma
    MDSPAN_INLINE_FUNCTION MDSPAN_IMPL_CONSTEXPR_14
    mapping(Kokkos::layout_left::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      :m_extents(other.extents())
    {}

    MDSPAN_TEMPLATE_REQUIRES(
      class OtherExtents,
      /* requires */ (
        MDSPAN_IMPL_TRAIT(std::is_constructible, extents_type, OtherExtents)
      )
    )
    MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
    MDSPAN_INLINE_FUNCTION MDSPAN_IMPL_CONSTEXPR_14
    mapping(Kokkos::layout_stride::mapping<OtherExtents> const& other) // NOLINT(google-explicit-constructor)
      :m_extents(other.extents())
    {
       /*
        * TODO: check precondition
        * other.required_span_size() is a representable value of type index_type
        */
       #ifndef __CUDA_ARCH__
       size_t stride = 1;
       for(rank_type r=m_extents.rank(); r>0; r--) {
         if(stride != other.stride(r-1))
           throw std::runtime_error("Assigning layout_stride to layout_foo with invalid strides.");
         stride *= m_extents.extent(r-1);
       }
       #endif
    }

    MDSPAN_INLINE_FUNCTION_DEFAULTED MDSPAN_IMPL_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

    MDSPAN_INLINE_FUNCTION
    constexpr const extents_type& extents() const noexcept {
      return m_extents;
    }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type required_span_size() const noexcept {
      index_type value = 1;
      for(rank_type r=0; r != extents_type::rank(); ++r) value*=m_extents.extent(r);
      return value;
    }

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION
    constexpr index_type operator() () const noexcept { return index_type(0); }

    template<class Indx0>
    MDSPAN_INLINE_FUNCTION
    constexpr index_type operator()(Indx0 idx0) const noexcept {
      return static_cast<index_type>(idx0);
    }

    template<class Indx0, class Indx1>
    MDSPAN_INLINE_FUNCTION
    constexpr index_type operator()(Indx0 idx0, Indx1 idx1) const noexcept {
      return static_cast<index_type>(idx0 * m_extents.extent(1) + idx1);
    }

    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_unique() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_exhaustive() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION static constexpr bool is_always_strided() noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_unique() const noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_exhaustive() const noexcept { return true; }
    MDSPAN_INLINE_FUNCTION constexpr bool is_strided() const noexcept { return true; }

    MDSPAN_INLINE_FUNCTION
    constexpr index_type stride(rank_type i) const noexcept {
      index_type value = 1;
      for(rank_type r=extents_type::rank()-1; r>i; r--) value*=m_extents.extent(r);
      return value;
    }

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

#if MDSPAN_HAS_CXX_17
   template<class... SliceSpecifiers>
     friend constexpr auto submdspan_mapping(
       const mapping& src, SliceSpecifiers... slices) {
         // use the fact that layout_foo is layout_right with rank 1 or rank 2
         // i.e. we don't need to implement everything here, we just reuse submdspan_mapping for layout_right
         Kokkos::layout_right::mapping<Extents> compatible_mapping(src.extents());
         auto sub_right = submdspan_mapping(compatible_mapping, slices...);
         if constexpr (std::is_same_v<typename decltype(sub_right.mapping)::layout_type, Kokkos::layout_right>) {
           // NVCC does not like deduction here, so get the extents type explicitly
           using sub_ext_t = std::remove_const_t<std::remove_reference_t<decltype(sub_right.mapping.extents())>>;
           auto sub_mapping = layout_foo::mapping<sub_ext_t>(sub_right.mapping.extents());
           return Kokkos::submdspan_mapping_result<decltype(sub_mapping)>{sub_mapping, sub_right.offset};
         } else {
           return sub_right;
         }
     }
#endif

private:
   MDSPAN_IMPL_NO_UNIQUE_ADDRESS extents_type m_extents{};

};

}
#endif
