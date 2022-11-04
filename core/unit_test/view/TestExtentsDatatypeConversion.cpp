#include <Kokkos_Core.hpp>
#include <type_traits>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN

namespace {

// Helper to make static tests more succinct
template <typename DataType, typename Extent>
constexpr bool datatype_matches_extent =
    std::is_same_v<typename Kokkos::Experimental::Impl::ExtentsFromDataType<
                       std::size_t, DataType>::type,
                   Extent>;

template <typename DataType, typename BaseType, typename Extents>
constexpr bool extent_matches_datatype =
    std::is_same_v<DataType, typename Kokkos::Experimental::Impl::
                   DataTypeFromExtents<BaseType, Extents>::type>;

// Conversion from DataType to extents
// 0-rank view
static_assert(
    datatype_matches_extent<double, std::experimental::extents<std::size_t>>);

// Only dynamic
static_assert(datatype_matches_extent<
              double***, std::experimental::extents<
                             std::size_t, std::experimental::dynamic_extent,
                             std::experimental::dynamic_extent,
                             std::experimental::dynamic_extent>>);
// Only static
static_assert(datatype_matches_extent<double[2][3][17],
              std::experimental::extents < std::size_t, std::size_t{2},
              std::size_t{3}, std::size_t{17} >>);

// Both dynamic and static
static_assert(
    datatype_matches_extent<double** [3][2][8],
    std::experimental::extents < std::size_t, std::experimental::dynamic_extent,
    std::experimental::dynamic_extent, std::size_t { 3 }, std::size_t{2},
    std::size_t{8} >>);

// Conversion from extents to DataType
// 0-rank extents
static_assert(extent_matches_datatype<double, double,
                                      std::experimental::extents<std::size_t>>);

// only dynamic
static_assert(
    extent_matches_datatype<double****, double,
                            std::experimental::extents<
                                std::size_t, std::experimental::dynamic_extent,
                                std::experimental::dynamic_extent,
                                std::experimental::dynamic_extent,
                                std::experimental::dynamic_extent>>);

// only static
static_assert(
    extent_matches_datatype<double[7][5][3], double,
                            std::experimental::extents<std::size_t, 7, 5, 3>>);

// both dynamic and static
static_assert(
    extent_matches_datatype<double*** [20][45], double,
                            std::experimental::extents<
                                std::size_t, std::experimental::dynamic_extent,
                                std::experimental::dynamic_extent,
                                std::experimental::dynamic_extent, 20, 45>>);
}  // namespace

#endif  // KOKKOS_ENABLE_IMPL_MDSPAN
