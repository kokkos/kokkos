#ifndef TESTVIEWMDSPAN_HPP_
#define TESTVIEWMDSPAN_HPP_

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <type_traits>

#ifdef KOKKOS_ENABLE_CXX20
#include <concepts>
#endif

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

namespace {
// Conversion from DataType to extents
// 0-rank view
static_assert(
    std::is_same_v<
        typename Kokkos::Experimental::Impl::ExtentsFromDataType<double>::type,
        std::experimental::extents<std::size_t>>);
// Only dynamic
static_assert(std::is_same_v<typename Kokkos::Experimental::Impl::
                                 ExtentsFromDataType<double***>::type,
                             std::experimental::extents<
                                 std::size_t, std::experimental::dynamic_extent,
                                 std::experimental::dynamic_extent,
                                 std::experimental::dynamic_extent>>);
// Only static
static_assert(std::is_same_v<
              typename Kokkos::Experimental::Impl::ExtentsFromDataType<
                  double[2][3][17]>::type,
              std::experimental::extents<std::size_t, std::size_t{2},
                                         std::size_t{3}, std::size_t{17}>>);
// Both dynamic and static
static_assert(
    std::is_same_v<typename Kokkos::Experimental::Impl::ExtentsFromDataType<
                       double** [3][2][8]>::type,
                   std::experimental::extents<
                       std::size_t, std::experimental::dynamic_extent,
                       std::experimental::dynamic_extent, std::size_t { 3 },
                       std::size_t{2}, std::size_t{8}>>);

// Conversion from extents to DataType
// 0-rank extents
static_assert(
    std::is_same_v<double,
                   typename Kokkos::Experimental::Impl::DataTypeFromExtents<
                       double, std::experimental::extents<std::size_t>>::type>);

// only dynamic
static_assert(std::is_same_v<
              double****,
              typename Kokkos::Experimental::Impl::DataTypeFromExtents<
                  double, std::experimental::extents<
                              std::size_t, std::experimental::dynamic_extent,
                              std::experimental::dynamic_extent,
                              std::experimental::dynamic_extent,
                              std::experimental::dynamic_extent>>::type>);

// only static
static_assert(
    std::is_same_v<
        double[7][5][3],
        typename Kokkos::Experimental::Impl::DataTypeFromExtents<
            double, std::experimental::extents<std::size_t, 7, 5, 3>>::type>);

// both dynamic and static
static_assert(
    std::is_same_v<
        double*** [20][45],
        typename Kokkos::Experimental::Impl::DataTypeFromExtents<
            double, std::experimental::extents<
                        std::size_t, std::experimental::dynamic_extent,
                        std::experimental::dynamic_extent,
                        std::experimental::dynamic_extent, 20, 45>>::type>);

#ifdef KOKKOS_ENABLE_CXX20
#if 0
// These asserts are C++20 only since mdspan requirements are expressed in terms
// of C++20 requirements
template <typename T>
concept satisfies_layout_requirements = std::copyable<T>
    && std::equality_comparable<T>
    && std::is_nothrow_constructible_v< T >
    && std::is_nothrow_move_assignable_v< T >
    && std::is_nothrow_swappable_v< T >;

using test_extent_type =
    std::experimental::extents<std::size_t, std::experimental::dynamic_extent,
                               std::size_t{6}>;
static_assert(
    satisfies_layout_requirements<Kokkos::Experimental::MDSpanLayoutLeft<test_extent_type>>);
#endif
#endif
}  // namespace Test

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN

#endif  // TESTVIEWMDSPAN_HPP_
