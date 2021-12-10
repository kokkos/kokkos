
template <size_t... Ds>
using _sizes = std::integer_sequence<size_t, Ds...>;

template <class>
struct TestViewAPI;
template <class DataType, class Layout, size_t... DynamicSizes,
          size_t... AllSizes>
struct TestViewAPI<
    std::tuple<DataType, Layout, std::integer_sequence<size_t, DynamicSizes...>,
               std::integer_sequence<size_t, AllSizes...>>>
    : public ::testing::Test {
  using data_type   = DataType;
  using layout_type = Layout;
  using space_type  = Kokkos::DefaultExecutionSpace;
  using traits_type =
      Kokkos::MemoryTraits<0>;  // maybe we want to add that later to the matrix
  using view_type =
      Kokkos::View<data_type, layout_type, space_type, traits_type>;
  using alloc_layout_type = typename std::conditional<
      std::is_same<layout_type, Kokkos::LayoutStride>::value,
      Kokkos::LayoutLeft, layout_type>::type;
  using d_alloc_type = Kokkos::View<data_type, alloc_layout_type, space_type>;
  using h_alloc_type = typename Kokkos::View<data_type, alloc_layout_type,
                                             space_type>::HostMirror;

  size_t dyn_sizes[sizeof...(DynamicSizes)] = {DynamicSizes...};
  size_t all_sizes[sizeof...(AllSizes)]     = {AllSizes...};

  constexpr static size_t expected_rank = sizeof...(AllSizes);

  inline view_type create_view() const {
    return d_alloc_type("TestViewAPI", DynamicSizes...);
  }
};

using Kokkos::LayoutLeft;
using Kokkos::LayoutRight;
using Kokkos::LayoutStride;

using compatible_extents_test_types = ::testing::Types<
    // LayoutLeft
    std::tuple<int, LayoutLeft, _sizes<>, _sizes<>>,
    std::tuple<int[5], LayoutLeft, _sizes<>, _sizes<5>>,
    std::tuple<int*, LayoutLeft, _sizes<5>, _sizes<5>>,
    std::tuple<int[5][10], LayoutLeft, _sizes<>, _sizes<5, 10>>,
    std::tuple<int * [10], LayoutLeft, _sizes<5>, _sizes<5, 10>>,
    std::tuple<int**, LayoutLeft, _sizes<5, 10>, _sizes<5, 10>>,
    std::tuple<int[5][10][15], LayoutLeft, _sizes<>, _sizes<5, 10, 15>>,
    std::tuple<int * [10][15], LayoutLeft, _sizes<5>, _sizes<5, 10, 15>>,
    std::tuple<int* * [15], LayoutLeft, _sizes<5, 10>, _sizes<5, 10, 15>>,
    std::tuple<int***, LayoutLeft, _sizes<5, 10, 15>, _sizes<5, 10, 15>>,
    // LayoutRight
    std::tuple<int, LayoutRight, _sizes<>, _sizes<>>,
    std::tuple<int[5], LayoutRight, _sizes<>, _sizes<5>>,
    std::tuple<int*, LayoutRight, _sizes<5>, _sizes<5>>,
    std::tuple<int[5][10], LayoutRight, _sizes<>, _sizes<5, 10>>,
    std::tuple<int * [10], LayoutRight, _sizes<5>, _sizes<5, 10>>,
    std::tuple<int**, LayoutRight, _sizes<5, 10>, _sizes<5, 10>>,
    std::tuple<int[5][10][15], LayoutRight, _sizes<>, _sizes<5, 10, 15>>,
    std::tuple<int * [10][15], LayoutRight, _sizes<5>, _sizes<5, 10, 15>>,
    std::tuple<int* * [15], LayoutRight, _sizes<5, 10>, _sizes<5, 10, 15>>,
    std::tuple<int***, LayoutRight, _sizes<5, 10, 15>, _sizes<5, 10, 15>>,
    // LayoutStride
    std::tuple<int, LayoutStride, _sizes<>, _sizes<>>,
    std::tuple<int[5], LayoutStride, _sizes<>, _sizes<5>>,
    std::tuple<int*, LayoutStride, _sizes<5>, _sizes<5>>,
    std::tuple<int[5][10], LayoutStride, _sizes<>, _sizes<5, 10>>,
    std::tuple<int * [10], LayoutStride, _sizes<5>, _sizes<5, 10>>,
    std::tuple<int**, LayoutStride, _sizes<5, 10>, _sizes<5, 10>>,
    std::tuple<int[5][10][15], LayoutStride, _sizes<>, _sizes<5, 10, 15>>,
    std::tuple<int * [10][15], LayoutStride, _sizes<5>, _sizes<5, 10, 15>>,
    std::tuple<int* * [15], LayoutStride, _sizes<5, 10>, _sizes<5, 10, 15>>,
    std::tuple<int***, LayoutStride, _sizes<5, 10, 15>, _sizes<5, 10, 15>>,
    // Degenerated Sizes
    std::tuple<int*, LayoutLeft, _sizes<0>, _sizes<0>>,
    std::tuple<int * [10], LayoutLeft, _sizes<0>, _sizes<0, 10>>,
    std::tuple<int* * [15], LayoutLeft, _sizes<0, 0>, _sizes<0, 0, 15>>,
    std::tuple<int*, LayoutRight, _sizes<0>, _sizes<0>>,
    std::tuple<int * [10], LayoutRight, _sizes<0>, _sizes<0, 10>>,
    std::tuple<int* * [15], LayoutRight, _sizes<0, 0>, _sizes<0, 0, 15>>,
    std::tuple<int*, LayoutStride, _sizes<0>, _sizes<0>>,
    std::tuple<int * [10], LayoutStride, _sizes<0>, _sizes<0, 10>>,
    std::tuple<int* * [15], LayoutStride, _sizes<0, 0>, _sizes<0, 0, 15>>>;

TYPED_TEST_SUITE(TestViewAPI, compatible_extents_test_types);

TYPED_TEST(TestViewAPI, sizes) {
  using view_type = typename TestFixture::view_type;
  auto a          = this->create_view();
  static_assert(view_type::rank == TestFixture::expected_rank);
  size_t expected_span = 1;
  for (int r = 0; r < view_type::rank; r++) expected_span *= this->all_sizes[r];

  EXPECT_EQ(expected_span, a.span());
  for (int r = 0; r < view_type::rank; r++) {
    EXPECT_EQ(this->all_sizes[r], a.extent(r));
    EXPECT_EQ(this->all_sizes[r], a.extent_int(r));
  }
}
