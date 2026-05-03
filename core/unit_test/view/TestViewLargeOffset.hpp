// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace TestViewLargeOffset {

struct OffsetProbe {};

// Test accessor that returns the offset computed by View::operator() instead of
// dereferencing memory. This lets the test cover large offsets without a large
// allocation.
template <class ElementType, class MemorySpace>
struct OffsetAccessor {
  static_assert(std::is_same_v<std::remove_cv_t<ElementType>, OffsetProbe>);

  using element_type     = ElementType;
  using reference        = std::size_t;
  using data_handle_type = ElementType*;
  using offset_policy    = OffsetAccessor;
  using memory_space     = MemorySpace;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr OffsetAccessor() = default;

  template <class OtherElementType, class OtherMemorySpace,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   ElementType (*)[]>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr OffsetAccessor(
      const OffsetAccessor<OtherElementType, OtherMemorySpace>&) noexcept {}

  KOKKOS_FUNCTION
  constexpr reference access(const data_handle_type&,
                             std::size_t offset) const noexcept {
    return offset;
  }

  KOKKOS_FUNCTION
  constexpr data_handle_type offset(const data_handle_type& handle,
                                    std::size_t) const noexcept {
    return handle;
  }
};

template <class LayoutType, class DeviceType, class MemoryTraits>
constexpr auto customize_view_arguments(
    Kokkos::Impl::ViewArguments<OffsetProbe, LayoutType, DeviceType,
                                MemoryTraits>) {
  return Kokkos::Impl::ViewCustomArguments<
      std::size_t,
      OffsetAccessor<OffsetProbe, typename DeviceType::memory_space>>{};
}

constexpr std::uint32_t bx      = 16;
constexpr std::uint32_t nb_octs = 32768;
// With 32 fields the last offset is UINT32_MAX. With 33 fields, the offset no
// longer fits in uint32_t but is still representable by the View mapping type.
constexpr std::uint32_t nb_fields = 33;
constexpr std::uint32_t n0        = bx * bx * bx;

using view_type =
    Kokkos::View<OffsetProbe***, Kokkos::LayoutLeft, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using right_view_type =
    Kokkos::View<OffsetProbe***, Kokkos::LayoutRight, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using strided_rank3_view_type =
    Kokkos::View<OffsetProbe***, Kokkos::LayoutStride, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using strided_rank1_view_type =
    Kokkos::View<OffsetProbe*, Kokkos::LayoutStride, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class IndexType>
void check_layout_left_large_offset_indices() {
  view_type fields(nullptr, Kokkos::LayoutLeft(n0, nb_fields, nb_octs));

  constexpr int index    = nb_fields - 1;
  const auto field_slice = std::pair<int, int>{index, index + 1};
  auto src = Kokkos::subview(fields, Kokkos::ALL, field_slice, Kokkos::ALL);

  constexpr IndexType i  = static_cast<IndexType>(n0 - 1);
  constexpr IndexType j  = static_cast<IndexType>(index);
  constexpr IndexType k  = static_cast<IndexType>(nb_octs / 100 - 1);
  constexpr IndexType kk = k* IndexType{100};

  constexpr std::size_t expected =
      static_cast<std::size_t>(i) + static_cast<std::size_t>(n0) * j +
      static_cast<std::size_t>(n0) * nb_fields * kk;
  static_assert(expected > std::numeric_limits<std::uint32_t>::max());

  EXPECT_EQ(fields(i, j, kk), expected);

  constexpr std::size_t subview_offset = static_cast<std::size_t>(n0) * j;
  EXPECT_EQ(subview_offset + src(i, IndexType{0}, kk), expected);
}

template <class IndexType>
void check_layout_right_large_offset_indices() {
  right_view_type fields(nullptr, Kokkos::LayoutRight(n0, nb_fields, nb_octs));

  constexpr int index    = nb_fields - 1;
  const auto field_slice = std::pair<int, int>{index, index + 1};
  auto src = Kokkos::subview(fields, Kokkos::ALL, field_slice, Kokkos::ALL);

  constexpr IndexType i  = static_cast<IndexType>(n0 - 1);
  constexpr IndexType j  = static_cast<IndexType>(index);
  constexpr IndexType k  = static_cast<IndexType>(nb_octs / 100 - 1);
  constexpr IndexType kk = k* IndexType{100};

  constexpr std::size_t expected =
      static_cast<std::size_t>(kk) + static_cast<std::size_t>(nb_octs) * j +
      static_cast<std::size_t>(nb_octs) * nb_fields * i;
  static_assert(expected > std::numeric_limits<std::uint32_t>::max());

  EXPECT_EQ(fields(i, j, kk), expected);

  constexpr std::size_t subview_offset = static_cast<std::size_t>(nb_octs) * j;
  EXPECT_EQ(subview_offset + src(i, IndexType{0}, kk), expected);
}

template <class IndexType>
void check_layout_stride_large_offset_indices() {
  Kokkos::LayoutStride layout(n0, 1, nb_fields, n0, nb_octs,
                              static_cast<std::size_t>(n0) * nb_fields);
  strided_rank3_view_type fields(nullptr, layout);

  constexpr int index    = nb_fields - 1;
  const auto field_slice = std::pair<int, int>{index, index + 1};
  auto src = Kokkos::subview(fields, Kokkos::ALL, field_slice, Kokkos::ALL);

  constexpr IndexType i  = static_cast<IndexType>(n0 - 1);
  constexpr IndexType j  = static_cast<IndexType>(index);
  constexpr IndexType k  = static_cast<IndexType>(nb_octs / 100 - 1);
  constexpr IndexType kk = k* IndexType{100};

  constexpr std::size_t expected =
      static_cast<std::size_t>(i) + static_cast<std::size_t>(n0) * j +
      static_cast<std::size_t>(n0) * nb_fields * kk;
  static_assert(expected > std::numeric_limits<std::uint32_t>::max());

  EXPECT_EQ(fields(i, j, kk), expected);

  constexpr std::size_t subview_offset = static_cast<std::size_t>(n0) * j;
  EXPECT_EQ(subview_offset + src(i, IndexType{0}, kk), expected);
}

template <class IndexType>
void check_rank1_large_stride_index() {
  constexpr std::size_t large_stride =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  const Kokkos::LayoutStride layout(2, large_stride);
  strided_rank1_view_type view(nullptr, layout);

  constexpr IndexType index = 1;
  EXPECT_EQ(view(index), large_stride);
}

}  // namespace TestViewLargeOffset

TEST(TEST_CATEGORY, view_large_offset_layout_left_uint32_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_left_large_offset_indices<std::uint32_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_layout_left_uint64_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_left_large_offset_indices<std::uint64_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_layout_right_uint32_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_right_large_offset_indices<std::uint32_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_layout_right_uint64_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_right_large_offset_indices<std::uint64_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_layout_stride_uint32_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_stride_large_offset_indices<
      std::uint32_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_layout_stride_uint64_indices) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_layout_stride_large_offset_indices<
      std::uint64_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_rank1_uint32_index) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_rank1_large_stride_index<std::uint32_t>();
#endif
}

TEST(TEST_CATEGORY, view_large_offset_rank1_uint64_index) {
#ifdef KOKKOS_IMPL_32BIT
  GTEST_SKIP() << "skipping for 32-bit builds";
#else
  TestViewLargeOffset::check_rank1_large_stride_index<std::uint64_t>();
#endif
}
