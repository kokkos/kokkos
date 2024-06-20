#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

constexpr size_t size = 5;

using device_type     = Kokkos::View<int*>::device_type;
using execution_space = typename device_type::execution_space;
using view_t          = Kokkos::View<double[size], execution_space>;

static const view_t my_view; // needed otherwise I get incomplete type for ViewTracker... wtf

using view_tracker_t  = Kokkos::Impl::ViewTracker<view_t>;
using track_t         = typename view_tracker_t::track_type;
using map_t           = Kokkos::Impl::ViewMapping<typename view_t::traits, typename view_t::traits::specialize>;

/// Can be replaced by lambda defined inside of a @c static_assert once @c c++20 is available.
/// @todo Do something meaningful with the @c track object.
constexpr auto test_SharedAllocationTracker()
{
    track_t track;
    return true;
}

TEST(Constexprness, SharedAllocationTracker)
{
    static_assert(test_SharedAllocationTracker());
}

/// Can be replaced by lambda defined inside of a @c static_assert once @c c++20 is available.
/// @todo Do something meaningful with the @c view_tracker object.
constexpr auto test_ViewTracker()
{
    view_tracker_t view_tracker;
    return true;
}

TEST(Constexpr, ViewTracker)
{
    static_assert(test_ViewTracker());
}

/// Can be replaced by lambda defined inside of a @c static_assert once @c c++20 is available.
/// @todo Do something meaningful with the @c map object.
constexpr auto test_ViewMapping()
{
    map_t map;
    return map.m_impl_handle == nullptr;
}

TEST(Constexpr, ViewMapping)
{
    static_assert(test_ViewMapping());
}

/// Can be replaced by lambda defined inside of a @c static_assert once @c c++20 is available.
/// @todo Do something meaningful with the @c view object.
constexpr auto test_View()
{
    view_t my_view;
    return my_view.size() == my_view.extent(0) && my_view.size() == size;
}

TEST(Constexpr, View)
{
    static_assert(test_View());
}
