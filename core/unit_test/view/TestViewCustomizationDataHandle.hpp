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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

// Test internal ways of customizing the data handle for applications such as resilience

namespace TestDataHandle {
template <class ElementType, class MemorySpace>
class TestCustomDataHandle : public Kokkos::Impl::ReferenceCountedDataHandle<ElementType, MemorySpace>
{
public:

  using base_t = Kokkos::Impl::ReferenceCountedDataHandle<ElementType, MemorySpace>;
  using memory_space = typename base_t::memory_space;
  using value_type = typename base_t::value_type;

  // this only ever works on host
  explicit TestCustomDataHandle(Kokkos::Impl::SharedAllocationRecord<void, void>* rec, int val)
    : base_t(rec), value( val ) {
  }

  // converting ctor
  template <class OtherElementType,
            class = std::enable_if_t<std::is_convertible_v<
                OtherElementType (*)[], value_type (*)[]>>>
  KOKKOS_FUNCTION TestCustomDataHandle(
      const TestCustomDataHandle<OtherElementType, memory_space>& other)
      : base_t(static_cast<base_t &>(*this)), value( other.value ) {}

  template <
      class OtherElementType, class OtherSpace,
      class = std::enable_if_t<
          std::is_convertible_v<OtherElementType (*)[], value_type (*)[]> &&
          Kokkos::SpaceAccessibility<memory_space,
                             typename OtherSpace::memory_space>::assignable>>
  KOKKOS_FUNCTION TestCustomDataHandle(
      const TestCustomDataHandle<OtherElementType, OtherSpace>& other)
      : base_t(static_cast<base_t &>(*this)), value( other.value ) {}

  int value = 0;
};

template< class T >
struct IsTestCustomDataHandle : std::false_type {};

template<class ElementType, class MemorySpace>
struct IsTestCustomDataHandle<TestCustomDataHandle<ElementType, MemorySpace>> : std::true_type {};

struct CustomDataType
{
  double value;
};

inline constexpr struct CustomDataHandleTag_t {} custom_data_handle_tag;

template <class LayoutType, class DeviceType, class MemoryTraits>
constexpr auto customize_view_arguments(
    Kokkos::Impl::ViewArguments<CustomDataType, LayoutType, DeviceType, MemoryTraits>) {
  return custom_data_handle_tag;
}

template <class ElementType, class MDSpanExtents, class MDSpanLayoutType, class MemorySpace, class NestedAccessor>
class TestCustomDataHandleAccessor {
 public:
  using element_type     = ElementType;
  using extents_type = MDSpanExtents;
  using layout_type = MDSpanLayoutType;
  using data_handle_type = TestCustomDataHandle<ElementType, MemorySpace>;
  using reference        = typename NestedAccessor::reference;
  using offset_policy =
      TestCustomDataHandleAccessor<ElementType, MDSpanExtents, MDSpanLayoutType, MemorySpace,
                                   typename NestedAccessor::offset_policy>;
  using memory_space = MemorySpace;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestCustomDataHandleAccessor() noexcept = default;

  template <
      class OtherElementType, class OtherNestedAccessor,
      class = std::enable_if_t<
          std::is_convertible_v<OtherElementType (*)[], element_type (*)[]> &&
          std::is_constructible_v<NestedAccessor, OtherNestedAccessor>>>
  KOKKOS_FUNCTION constexpr TestCustomDataHandleAccessor(
      const TestCustomDataHandleAccessor<OtherElementType, MDSpanExtents,
                                         MDSpanLayoutType, MemorySpace,
                                         OtherNestedAccessor>&) {}

  template <
      class OtherElementType, class OtherSpace, class OtherNestedAccessor,
      class = std::enable_if_t<
          std::is_convertible_v<OtherElementType (*)[], element_type (*)[]> &&
          Kokkos::SpaceAccessibility<
              memory_space, typename OtherSpace::memory_space>::assignable &&
          std::is_constructible_v<NestedAccessor, OtherNestedAccessor>>>
  KOKKOS_FUNCTION constexpr TestCustomDataHandleAccessor(
      const TestCustomDataHandleAccessor<OtherElementType, MDSpanExtents,
                                         MDSpanLayoutType, OtherSpace,
                                         OtherNestedAccessor>&) {}

  template <class OtherElementType,
            class = std::enable_if_t<std::is_convertible_v<
                OtherElementType (*)[], element_type (*)[]>>>
  KOKKOS_FUNCTION constexpr TestCustomDataHandleAccessor(
      const Kokkos::default_accessor<OtherElementType>&) {}

  template <class DstAccessor,
            typename = std::enable_if_t<
                !Kokkos::Impl::IsReferenceCountedAccessor<DstAccessor>::value &&
                std::is_convertible_v<NestedAccessor, DstAccessor>>>
  KOKKOS_FUNCTION operator DstAccessor() const {
    return m_nested_acc;
  }

  KOKKOS_FUNCTION
  constexpr reference access(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OpenACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const {
    return m_nested_acc.access(p.get(), i);
  }

  KOKKOS_FUNCTION
  constexpr data_handle_type offset(
#ifndef KOKKOS_ENABLE_OPENACC
      const data_handle_type& p,
#else
      // FIXME OpenACC: illegal address when passing by reference
      data_handle_type p,
#endif
      size_t i) const {
    return data_handle_type{p, m_nested_acc.offset(p.get(), i)};
  }

  KOKKOS_FUNCTION
  constexpr auto nested_accessor() const { return m_nested_acc; }

 private:
#ifdef MDSPAN_IMPL_NO_UNIQUE_ADDRESS
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS
#else
  [[no_unique_address]]
#endif
  NestedAccessor m_nested_acc;
};

template< typename DataHandleType, typename MDSpanExtents, typename MDSpanLayoutType, typename ElementType, typename MemorySpace >
KOKKOS_INLINE_FUNCTION constexpr DataHandleType data_handle_from_allocation(Kokkos::Impl::SharedAllocationRecord<void, void>* rec,
  const typename MDSpanLayoutType::template mapping< MDSpanExtents > &mapping,
  const Kokkos::Impl::SpaceAwareAccessor<MemorySpace, TestCustomDataHandleAccessor<ElementType, MDSpanExtents, MDSpanLayoutType, MemorySpace, Kokkos::default_accessor<ElementType> > > &/* accessor */) {
  return DataHandleType(rec, 13);
}
}

// Don't try this at home!
namespace Kokkos::Impl {
template <class Traits, class LayoutType>
struct MDSpanViewTraits<Traits, TestDataHandle::CustomDataHandleTag_t, LayoutType> {
  using index_type = typename MDSpanViewTraits<Traits, void, LayoutType>::index_type;
  using extents_type = typename MDSpanViewTraits<Traits, void, LayoutType>::extents_type;
  using mdspan_layout_type = typename MDSpanViewTraits<Traits, void, LayoutType>::mdspan_layout_type;
  using accessor_type      = Kokkos::Impl::SpaceAwareAccessor<typename Traits::memory_space, TestDataHandle::TestCustomDataHandleAccessor<typename Traits::value_type, extents_type, mdspan_layout_type, typename Traits::memory_space, Kokkos::default_accessor<typename Traits::value_type> > >;
  // This will static assert that accessor_type is legal
  using mdspan_type = mdspan<typename Traits::value_type, extents_type,
                             mdspan_layout_type, accessor_type>;
};

template <class ElementType, class MemorySpace>
KOKKOS_INLINE_FUNCTION constexpr auto ptr_from_data_handle(
    const TestDataHandle::TestCustomDataHandle<ElementType, MemorySpace>& handle) {
  return handle.get();
}

template <class ElementType, class MemorySpace>
struct IsReferenceCountedDataHandle<
    TestDataHandle::TestCustomDataHandle<ElementType, MemorySpace>> : std::true_type {};
}


using view_custom_data_handle_type = Kokkos::View<TestDataHandle::CustomDataType *>;

static_assert(TestDataHandle::IsTestCustomDataHandle<view_custom_data_handle_type::data_handle_type>::value, "view data handle type must be a resilient handle");

TEST(TEST_CATEGORY, view_customization_data_handle) {
  auto v = view_custom_data_handle_type("test_v", 17);
  ASSERT_EQ( v.data_handle().value, 13 );
}
