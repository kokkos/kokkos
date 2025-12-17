
#define ASSERT_SAME_TYPE(...) static_assert((std::is_same_v<__VA_ARGS__>), "Types must be the same")
#define LIBCPP_STATIC_ASSERT(A) static_assert(A);
//#define ASSERT_NOEXCEPT(A) if consteval { static_assert(noexcept(A)); } else { assert(noexcept(A)); }
#define ASSERT_NOEXCEPT(...) static_assert(noexcept(__VA_ARGS__), "Operation must be noexcept");

namespace std {
  using Kokkos::mdspan;
  using Kokkos::extents;
  using Kokkos::dextents;
  using Kokkos::default_accessor;
  using Kokkos::layout_left;
  using Kokkos::layout_right;
  using Kokkos::layout_stride;
}
