#define MDSPAN_INTERNAL_TEST
#define MDSPAN_DEBUG
#include <cassert>
#include <gtest/gtest.h>
#include <mdspan/mdspan.hpp>

namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

// Compile time tests

// For internal traits
struct fake_mapping {
  using layout_type = Kokkos::Experimental::layout_right_padded<5>;
};

static_assert(!Kokkos::Experimental::detail::is_layout_right_padded_mapping<
              fake_mapping>::value);

static_assert(Kokkos::Experimental::detail::is_layout_right_padded_mapping<
              Kokkos::Experimental::layout_right_padded<4>::mapping<
                  Kokkos::extents<size_t, 7, 4>>>::value);

// layout_right_padded must be trivial
static_assert(std::is_trivial_v<KokkosEx::layout_right_padded<0>>);
static_assert(std::is_trivial_v<KokkosEx::layout_right_padded<4>>);
static_assert(std::is_trivial_v<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>);

// actual padding stride
// If extents_type::rank() equals zero or one, then padding_stride.
static_assert(KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 0>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t, 0>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 0>>::static_padding_stride == 0);
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t>>::static_padding_stride == 0);

// Else, if
// - padding_stride does not equal dynamic_extent and
// - extents_type::static_extent(0) does not equal dynamic_extent,
// then the size_t value which is the least multiple of padding_stride that is greater than or equal to extents_type::static_extent(0).
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t, 7, 3>>::static_padding_stride == 4);
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 3>>::static_padding_stride == 4);

// Otherwise, dynamic_extent.
static_assert(KokkosEx::layout_right_padded<2>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>::static_padding_stride == Kokkos::dynamic_extent);
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>::static_padding_stride == Kokkos::dynamic_extent);
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>::static_padding_stride == Kokkos::dynamic_extent);

namespace
{
template<class LayoutrightPadded, class Extents, class TestExtents>
void test_padding_stride(const Extents &extents, const TestExtents &test_extents)
{
  auto mapping = typename LayoutrightPadded::template mapping<Extents>(extents);
  if constexpr (TestExtents::rank() > 1) {
    ASSERT_EQ(mapping.padded_stride.value(0), test_extents.extent(decltype(mapping)::extent_to_pad_idx));
  } else {
    ASSERT_EQ(mapping.padded_stride.value(0), 0);
  }

  [[maybe_unused]] size_t prod = 1;
  size_t span_size = 1;
  // get rid of NVCC warning "pointless comparison of unsigned integer with zero"
  if constexpr (TestExtents::rank() > 0) {
    auto strs = mapping.strides();
    for (typename decltype(mapping)::rank_type rrev = 0; rrev < TestExtents::rank(); ++rrev)
    {
      auto r = TestExtents::rank() - 1 - rrev;
      ASSERT_EQ(strs[r], prod);
      ASSERT_EQ(mapping.stride(r), prod);
      prod *= test_extents.extent(r);
      span_size += (extents.extent(r) - 1) * strs[r];
    }
  }
  ASSERT_EQ(span_size, mapping.required_span_size());
}

template <class LayoutrightPadded, class Extents, class TestExtents, class Size>
void test_padding_stride(const Extents &extents, const TestExtents &test_extents, Size padding_value)
{
  auto mapping = typename LayoutrightPadded::template mapping<Extents>(extents, padding_value);
  if constexpr (TestExtents::rank() > 1) {
    ASSERT_EQ(mapping.padded_stride.value(0), test_extents.extent(decltype(mapping)::extent_to_pad_idx));
  } else {
    ASSERT_EQ(mapping.padded_stride.value(0), 0);
  }

  [[maybe_unused]] size_t prod = 1;
  size_t span_size = 1;
  // get rid of NVCC warning "pointless comparison of unsigned integer with zero"
  if constexpr (TestExtents::rank() > 0) {
    auto strs = mapping.strides();
    for (typename decltype(mapping)::rank_type rrev = 0; rrev < TestExtents::rank(); ++rrev)
    {
      auto r = TestExtents::rank() - 1 - rrev;
      ASSERT_EQ(strs[r], prod);
      ASSERT_EQ(mapping.stride(r), prod);
      prod *= test_extents.extent(r);
      span_size += (extents.extent(r) - 1) * strs[r];
    }
  }
  ASSERT_EQ(span_size, mapping.required_span_size());
}

template <class LayoutRightPadded, class Extents>
void test_0_or_1_rank_padding_stride(const Extents &extents)
{
  test_padding_stride<LayoutRightPadded>(extents, extents);
}

template <class LayoutRightPadded, class Extents, class Size>
void test_0_or_1_rank_padding_stride(const Extents &extents, Size padding_value)
{
  test_padding_stride<LayoutRightPadded>(extents, extents, padding_value);
}

template <class LayoutRightPadded, class Extents>
void test_default_constructor_equivalence()
{
  using mapping_type = typename LayoutRightPadded::template mapping<Extents>;

  ASSERT_EQ(mapping_type(), mapping_type(Extents{}));
}

template <class LayoutRightPadded, class Extents>
void test_copy_constructor(const Extents &extents)
{
  using mapping_type = typename LayoutRightPadded::template mapping<Extents>;
  auto a = mapping_type(extents);
  auto b = a;

  ASSERT_EQ(a, b);
}

template <class LayoutRightPadded, class Extents, class Size>
void test_copy_constructor(const Extents &extents, Size padding_value)
{
  using mapping_type = typename LayoutRightPadded::template mapping<Extents>;
  auto a = mapping_type(extents, padding_value);
  auto b = a;

  ASSERT_EQ(a, b);
}

template <class LayoutRightPadded, class Extents>
void test_copy_assignment(const Extents &extents)
{
  using mapping_type = typename LayoutRightPadded::template mapping<Extents>;
  auto a = mapping_type(extents);
  mapping_type b;

  b = a;

  ASSERT_EQ(a, b);
}

template <class LayoutRightPadded, class Extents, class Size>
void test_copy_assignment(const Extents &extents, Size padding_value)
{
  using mapping_type = typename LayoutRightPadded::template mapping<Extents>;
  auto a = mapping_type(extents, padding_value);
  mapping_type b;

  b = a;

  ASSERT_EQ(a, b);
}
}

TEST(LayoutrightTests, construction)
{
  // Default Constructor
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, 3>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, 3, 2>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, 3, 2, 1>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>, Kokkos::extents<std::size_t>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>, Kokkos::extents<std::size_t, 3>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>, Kokkos::extents<std::size_t, 3, 2>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>, Kokkos::extents<std::size_t, 3, 2, 1>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, 3, 2, Kokkos::dynamic_extent>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent, Kokkos::dynamic_extent>>();
  test_default_constructor_equivalence<KokkosEx::layout_right_padded<4>, Kokkos::extents<std::size_t, Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>>();

  // Copy constructor
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t>());
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3>());
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2>());
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2, 1>());
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2, Kokkos::dynamic_extent>(10));
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent, Kokkos::dynamic_extent>(9, 10));
  test_copy_constructor<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>(8, 9, 10));

  test_copy_constructor<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t>(), 5);
  test_copy_constructor<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3>(), 5);
  test_copy_constructor<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3, 2>(), 5);
  test_copy_constructor<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3, 2, 1>(), 5);

  // Copy assignment
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t>());
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3>());
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2>());
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2, 1>());
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, 2, Kokkos::dynamic_extent>(10));
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent, Kokkos::dynamic_extent>(9, 10));
  test_copy_assignment<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>(8, 9, 10));

  test_copy_assignment<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t>(), 5);
  test_copy_assignment<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3>(), 5);
  test_copy_assignment<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3, 2>(), 5);
  test_copy_assignment<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 3, 2, 1>(), 5);

  // Constructor only taking an extent
  // Direct-non-list-initializes inner-mapping with:
  // - ext, if extents_type::rank() is zero or one; else,
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 0>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 5>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{7});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 0>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 5>{});
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{7});

  // - ext.extent(0), ext.extent(P_right)..., if padding_stride is dynamic_extent
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 0>{}, Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{ 0 });
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 5>{}, Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{ 5 });
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7});

  // - S_right, ext.extent(P_right)..., where S_right is the least multiple of padding_stride greater than or equal to ext.extent(0)
  test_padding_stride<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 7, 0>{}, Kokkos::extents<std::size_t, 7, 0>{});
  test_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 7, 5>{}, Kokkos::extents<std::size_t, 7, 8>{});
  test_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{8});

  // Constructor taking an extent and a dynamic value
  // Direct-non-list-initializes inner-mapping with:
  // - ext, if extents_type::rank() is zero or one; else,
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t>{}, 4);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 0>{}, 0);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 5>{}, 4);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{7}, 4);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t>{}, 3);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 0>{}, 3255);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 5>{}, 1337);
  test_0_or_1_rank_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{7}, 6323);

  // - S_right, ext.extent(P_right)..., where S_right is the least multiple of padding_value greater than or equal to ext.extent(0)
  test_padding_stride<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 7, 0>{}, Kokkos::extents<std::size_t, 7, 0>{}, 0);
  test_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 7, 5>{}, Kokkos::extents<std::size_t, 7, 8>{}, 4);
  test_padding_stride<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{8}, 4);
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 0>{}, Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{0}, 2);
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 5>{}, Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{8}, 4);
  test_padding_stride<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{8}, 4);

  // Construct layout_right_padded mapping from layout_right mapping
  ASSERT_EQ(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).padded_stride.value(0)), 4);

  ASSERT_EQ(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, 4>>()).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{4})).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{4})).padded_stride.value(0)), 4);

  // Construct layout_right_padded mapping from layout stride
  ASSERT_EQ(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 3>>({}, std::array<std::size_t, 1>{1})).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 3>>({}, std::array<std::size_t, 1>{1})).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).padded_stride.value(0)), 8);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).padded_stride.value(0)), 8);

  ASSERT_EQ(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 3>>({}, std::array<std::size_t, 1>{1})).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 3>>({}, std::array<std::size_t, 1>{1})).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 4>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 4>>({}, std::array<std::size_t, 2>{4, 1})).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).padded_stride.value(0)), 8);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, 7, 5>>({}, std::array<std::size_t, 2>{8, 1})).padded_stride.value(0)), 8);

  // Construct layout_right_padded from another layout_right_padded
  ASSERT_EQ(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>()).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>()).padded_stride.value(0)), 4);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>()).extents()), (Kokkos::extents<std::size_t, 7, 4>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 4>>()).padded_stride.value(0)), 4);

  ASSERT_EQ(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>()).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>()).padded_stride.value(0)), 8);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 5>>(Kokkos::extents<size_t, Kokkos::dynamic_extent, 5>(7))).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 5>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent, 5>>(Kokkos::extents<size_t, Kokkos::dynamic_extent, 5>(7))).padded_stride.value(0)), 8);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>()).extents()), (Kokkos::extents<std::size_t, 7, 5>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>()).padded_stride.value(0)), 8);

  // Construct layout_right_padded mapping from layout_left_padded mapping
  ASSERT_EQ(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>(Kokkos::dextents<size_t, 1>(3))).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>(Kokkos::dextents<size_t, 1>(3))).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>({}, 4)).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>({}, 4)).padded_stride.value(0)), 0);

  ASSERT_EQ(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>()).padded_stride.value(0)), 0);
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>({}, 4)).extents()), (Kokkos::extents<std::size_t, 3>()));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>(KokkosEx::layout_left_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>({}, 4)).padded_stride.value(0)), 0);

  // Construct layout_right mapping from layout_right_padded mapping
  ASSERT_EQ(Kokkos::layout_right::mapping<Kokkos::extents<std::size_t>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>()).extents(), Kokkos::extents<std::size_t>());
  ASSERT_EQ((Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 5>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 5>>()).extents()), (Kokkos::extents<std::size_t, 5>()));
  ASSERT_EQ((Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 4, 8>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 4, 8>>()).extents()), (Kokkos::extents<std::size_t, 4, 8>()));
  ASSERT_EQ((Kokkos::layout_right::mapping<Kokkos::extents<std::size_t, 3, 4, 8>>(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3, 4, 8>>()).extents()), (Kokkos::extents<std::size_t, 3, 4, 8>()));
}

namespace
{
  template <class PaddedLayout, class Extents>
  void test_extent(const Extents &input_extents)
  {
    auto mapping = typename PaddedLayout::template mapping<Extents>(input_extents);
    ASSERT_EQ(mapping.extents(), input_extents);
  }

  template <class PaddedLayout, class Extents, class Size>
  void test_extent(const Extents &input_extents, Size padding_value)
  {
    auto mapping = typename PaddedLayout::template mapping<Extents>(input_extents, padding_value);
    ASSERT_EQ(mapping.extents(), input_extents);
  }
}

TEST(LayoutrightTests, extents)
{
  test_extent<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t>{});
  test_extent<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 7, 0>{});
  test_extent<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 7, 5>{});
  test_extent<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7});
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 0>{});
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 5>{});
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7});

  test_extent<KokkosEx::layout_right_padded<0>>(Kokkos::extents<std::size_t, 7, 0>{}, 0);
  test_extent<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 7, 5>{}, 4);
  test_extent<KokkosEx::layout_right_padded<4>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, 4);
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 0>{}, 1);
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 7, 5>{}, 3);
  test_extent<KokkosEx::layout_right_padded<Kokkos::dynamic_extent>>(Kokkos::extents<std::size_t, 13, Kokkos::dynamic_extent>{7}, 5);
}

// is_always_exhaustive
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{5}}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>{}.is_always_exhaustive());

static_assert(!KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 4, 3>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 4>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 8>>{}.is_always_exhaustive());
static_assert(!KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 4, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 4, Kokkos::dynamic_extent>{5}}.is_always_exhaustive());
static_assert(!KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 5, 4>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 6, 0>>{}.is_always_exhaustive());
static_assert(KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 0>>{}.is_always_exhaustive());

TEST(LayoutrightTests, properties)
{
  // is_exhaustive
  // Sanity check -- if it is always exhaustive it should be exhaustive ^-^
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 3>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 4>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 8>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 6, 0>>{}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 0>>{}.is_exhaustive()));
  ASSERT_FALSE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, 3>>{}.is_exhaustive()));

  // is_exhaustive with dynamic values
  ASSERT_TRUE((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 6, 4>>{Kokkos::extents<std::size_t, 6, 4>{}, 4}.is_exhaustive()));
  ASSERT_TRUE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 6, Kokkos::dynamic_extent>{8}}.is_exhaustive()));
  ASSERT_FALSE((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 6, 6>>{Kokkos::extents<std::size_t, 6, 6>{}, 4}.is_exhaustive()));
  ASSERT_FALSE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 6, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 6, Kokkos::dynamic_extent>{7}}.is_exhaustive()));

  // Equality
  ASSERT_EQ((KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 0>>{}), (KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 0>>{}));
  ASSERT_EQ((KokkosEx::layout_right_padded<0>::mapping<Kokkos::extents<std::size_t, 0>>{}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 0>>{}));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>{}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>{}));
  ASSERT_NE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3>>{}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 5>>{}));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{5}}),
            (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{5}}));
  ASSERT_NE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{3}}),
            (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, Kokkos::dynamic_extent>{5}}));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}));
  ASSERT_NE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 4, 3>>{}));
  ASSERT_NE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}), (KokkosEx::layout_right_padded<8>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}));
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>{5}}),
            (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>{5}}));
  ASSERT_NE((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>{3}}),
            (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>>{Kokkos::extents<std::size_t, 3, Kokkos::dynamic_extent>{5}}));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 4}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 3>>{}));
  ASSERT_EQ((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 4}), (KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 4}));
  ASSERT_NE((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 4}), (KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 4, 3>>{}));
  ASSERT_NE((KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 4}), (KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<Kokkos::extents<std::size_t, 7, 3>>{{}, 8}));
}

TEST(LayoutRightTests, stride)
{
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>().stride(0)), 8);
  ASSERT_EQ((KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>().stride(1)), 1);
}

TEST(LayoutRightTests, access)
{
  auto mapping1 = KokkosEx::layout_right_padded<4>::mapping<Kokkos::extents<std::size_t, 7, 5>>();
  ASSERT_EQ(mapping1(0, 0), 0);
  ASSERT_EQ(mapping1(0, 1), 1);
  ASSERT_EQ(mapping1(0, 4), 4);
  ASSERT_EQ(mapping1(1, 0), 8);
  ASSERT_EQ(mapping1(1, 4), 12);
  ASSERT_EQ(mapping1(6, 0), 48);
  ASSERT_EQ(mapping1(6, 4), 52);

  auto mapping2 = KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
      Kokkos::extents<std::size_t, 7, 5>>({}, 6);
  ASSERT_EQ(mapping2(0, 0), 0);
  ASSERT_EQ(mapping2(0, 1), 1);
  ASSERT_EQ(mapping2(0, 4), 4);
  ASSERT_EQ(mapping2(1, 0), 6);
  ASSERT_EQ(mapping2(1, 4), 10);
  ASSERT_EQ(mapping2(6, 0), 36);
  ASSERT_EQ(mapping2(6, 4), 40);

  auto mapping3 = KokkosEx::layout_right_padded<2>::mapping<
      Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(
      Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{3});
  ASSERT_EQ(mapping3(0, 0), 0);
  ASSERT_EQ(mapping3(0, 1), 1);
  ASSERT_EQ(mapping3(0, 2), 2);
  ASSERT_EQ(mapping3(1, 0), 4);
  ASSERT_EQ(mapping3(1, 2), 6);
  ASSERT_EQ(mapping3(6, 0), 24);
  ASSERT_EQ(mapping3(6, 2), 26);

  auto mapping4 =
      KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
          Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>>(
          Kokkos::extents<std::size_t, 7, Kokkos::dynamic_extent>{7}, 10);
  ASSERT_EQ(mapping4(0, 0), 0);
  ASSERT_EQ(mapping4(0, 1), 1);
  ASSERT_EQ(mapping4(0, 6), 6);
  ASSERT_EQ(mapping4(1, 0), 10);
  ASSERT_EQ(mapping4(1, 6), 16);
  ASSERT_EQ(mapping4(6, 0), 60);
  ASSERT_EQ(mapping4(6, 6), 66);

  auto mapping5 =
      KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
          Kokkos::extents<std::size_t, 7>>({}, 4);
  ASSERT_EQ(mapping5(0), 0);
  ASSERT_EQ(mapping5(1), 1);
  ASSERT_EQ(mapping5(2), 2);
  ASSERT_EQ(mapping5(3), 3);
  ASSERT_EQ(mapping5(4), 4);
  ASSERT_EQ(mapping5(5), 5);
  ASSERT_EQ(mapping5(6), 6);

  auto mapping6 =
      KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
          Kokkos::extents<std::size_t>>({}, 4);
  ASSERT_EQ(mapping6(), 0);
}

// https://github.com/kokkos/mdspan/issues/362
TEST(LayoutRightTests, issue362) {
  auto mapping = KokkosEx::layout_right_padded< 5 >::mapping< Kokkos::extents< std::size_t, 2, 2 > >();
  ASSERT_EQ(mapping.required_span_size(), mapping(1, 1) + 1);
}

TEST(LayoutRightTests, empty_span) {
  {
    auto mapping = KokkosEx::layout_right_padded<16>::mapping<
        Kokkos::extents<std::size_t, 0, 15>>();
    ASSERT_EQ(mapping.required_span_size(), 0);
  }
  {
    auto mapping = KokkosEx::layout_right_padded<16>::mapping<
        Kokkos::extents<std::size_t, 15, 0>>();
    ASSERT_EQ(mapping.required_span_size(), 0);
  }
}

// https://github.com/kokkos/mdspan/issues/393
#define LAYOUT_RIGHT_COMPILE_ISSUE393_DEATH 0
TEST(LayoutRightTests, issue393) {
  // Should not compile
#if LAYOUT_RIGHT_COMPILE_ISSUE393_DEATH
  {
    // static extents size not representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded< 2 >::mapping< Kokkos::extents< std::int8_t, 50, 50 > >();
  }
  {
    // Padding value not representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded< 500 >::mapping< Kokkos::extents< std::int8_t, 2, 2 > >();
  }
  {
    // Padding value product with remaining extents is representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded< 50 >::mapping< Kokkos::extents< std::int8_t, 50, 2 > >();
  }
#endif

  // Valid usage, should compile without narrowing warnings
  {
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded<5>::mapping<
        Kokkos::extents<std::int16_t, 2, 50>>();
  }

  // Test runtime
  // Note GTest deathtest macros are a bit annoying so we just declare the tests
  // in lambdas separately
  auto test_extents_not_representable = []{
    // extents not representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded<2>::mapping<
        Kokkos::dextents<std::int8_t, 2>>{
        Kokkos::dextents<std::int8_t, 2>{50, 50}};
  };
  EXPECT_DEATH(test_extents_not_representable(), "" );

  auto test_padding_value_not_representable = []{
    // Padding value not representable
    [[maybe_unused]] auto mapping =
        KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::int8_t, 2, 2>>{{}, 500};
  };
  EXPECT_DEATH(test_padding_value_not_representable(), "" );

  auto test_padding_value_product_not_representable = []{
    // Padding value product with remaining extents is representable
    [[maybe_unused]] auto mapping =
        KokkosEx::layout_right_padded<Kokkos::dynamic_extent>::mapping<
            Kokkos::extents<std::int8_t, 50, 2>>{{}, 50};
  };
  EXPECT_DEATH(test_padding_value_product_not_representable(), "" );

  auto test_padding_value_product_not_representable2 = []{
    // Padding value product with remaining extents is representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded<
        Kokkos::dynamic_extent>::mapping<Kokkos::dextents<std::int8_t, 2>>{
        Kokkos::dextents<std::int8_t, 2>{50, 2}, 50};
  };
  EXPECT_DEATH(test_padding_value_product_not_representable2(), "" );

  auto test_padding_value_product_not_representable3 = []{
    // Padding value product with remaining extents is representable
    [[maybe_unused]] auto mapping = KokkosEx::layout_right_padded<50>::mapping<
        Kokkos::dextents<std::int8_t, 2>>{
        Kokkos::dextents<std::int8_t, 2>{50, 2}};
  };
  EXPECT_DEATH(test_padding_value_product_not_representable3(), "" );
}
#undef LAYOUT_RIGHT_COMPILE_ISSUE393_DEATH
