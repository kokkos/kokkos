// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SYCL_MDRANGEPOLICY_HPP_
#define KOKKOS_SYCL_MDRANGEPOLICY_HPP_

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {

template <>
struct default_outer_direction<Kokkos::SYCL> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct default_inner_direction<Kokkos::SYCL> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

namespace Impl {

template <typename... Properties>
struct MDRangePolicyInternal;

// Specialization for SYCL execution space
template <typename P, typename... Properties>
struct MDRangePolicyInternal<Kokkos::SYCL, P, Properties...>
    : public PolicyTraits<P, Properties...> {
 public:
  using traits          = Impl::PolicyTraits<P, Properties...>;
  using execution_space = Kokkos::SYCL;
  using range_policy    = RangePolicy<Properties...>;

  using iteration_pattern = typename traits::iteration_pattern;
  using work_tag          = typename traits::work_tag;
  using launch_bounds     = typename traits::launch_bounds;
  using member_type       = typename range_policy::member_type;

  template <typename... OtherProperties>
  friend struct MDRangePolicyInternal;

  static constexpr int rank = iteration_pattern::rank;

  using index_type       = typename traits::index_type;
  using array_index_type = std::make_signed_t<index_type>;
  using point_type       = Kokkos::Array<array_index_type, rank>;
  using tile_type        = Kokkos::Array<array_index_type, rank>;

  execution_space m_space;

 public:
  int m_max_total_tile_size                      = 1;
  Kokkos::Array<int, 3> m_max_threads_dimensions = {};

  point_type m_lower          = {};
  point_type m_upper          = {};
  tile_type m_tile            = {};
  point_type m_tile_end       = {};
  index_type m_num_tiles      = 1;
  index_type m_prod_tile_dims = 1;
  bool m_tune_tile_size       = false;

  static constexpr auto outer_direction =
      (iteration_pattern::outer_direction != Iterate::Default)
          ? iteration_pattern::outer_direction
          : default_outer_direction<typename traits::execution_space>::value;

  static constexpr auto inner_direction =
      iteration_pattern::inner_direction != Iterate::Default
          ? iteration_pattern::inner_direction
          : default_inner_direction<typename traits::execution_space>::value;

  static constexpr auto Right = Iterate::Right;
  static constexpr auto Left  = Iterate::Left;

 public:
  MDRangePolicyInternal() {
    m_max_total_tile_size =
        space.impl_internal_space_instance()->m_maxWorkgroupSize;
    auto device = space.sycl_queue().get_device();
    auto max_work_item_sizes =
        device.get_info<sycl::info::device::max_work_item_sizes<3>>();
    m_max_threads_dimensions[0] = max_work_item_sizes[0];
    m_max_threads_dimensions[1] = max_work_item_sizes[1];
    m_max_threads_dimensions[2] = max_work_item_sizes[2];
    if constexpr (launch_bounds::maxTperB != 0) {
      m_max_total_tile_size =
          std::min<index_type>(launch_bounds::maxTperB, m_max_total_tile_size);
    }
  }

  template <typename OtherExecSpace, typename OtherP,
            typename... OtherProperties>
  MDRangePolicyInternal(const MDRangePolicyInternal<OtherExecSpace, OtherP,
                                                    OtherProperties...>& p)
      : traits(p),  // base class may contain data such as desired occupancy
        m_space(p.m_space),
        m_max_total_tile_size(p.m_max_total_tile_size),
        m_max_threads_dimensions(p.m_max_threads_dimensions),
        m_lower(p.m_lower),
        m_upper(p.m_upper),
        m_tile(p.m_tile),
        m_tile_end(p.m_tile_end),
        m_num_tiles(p.m_num_tiles),
        m_prod_tile_dims(p.m_prod_tile_dims),
        m_tune_tile_size(p.m_tune_tile_size) {}

  // Default constructor and assignment operators
  MDRangePolicyInternal(const MDRangePolicyInternal&)            = default;
  MDRangePolicyInternal(MDRangePolicyInternal&&)                 = default;
  MDRangePolicyInternal& operator=(const MDRangePolicyInternal&) = default;
  MDRangePolicyInternal& operator=(MDRangePolicyInternal&&)      = default;
  ~MDRangePolicyInternal()                                       = default;

 public:
  tile_type tile_size_recommended() const {
    tile_type tile_sizes = {};
    if (inner_direction == Iterate::Left) {
      if constexpr (rank == 2) {
        tile_sizes = {64, 4};
      } else if constexpr (rank == 3) {
        tile_sizes = {32, 2, 4};
      } else if constexpr (rank == 4) {
        tile_sizes = {32, 2, 2, 2};
      } else if constexpr (rank == 5) {
        tile_sizes = {32, 2, 2, 1, 2};
      } else if constexpr (rank == 6) {
        tile_sizes = {32, 2, 2, 1, 2, 1};
      } else {
        for (int i = 0; i < rank; ++i) {
          tile_sizes[i] = 2;
        }
        tile_sizes[0] = 16;
      }
    } else {
      if constexpr (rank == 2) {
        tile_sizes = {4, 64};
      } else if constexpr (rank == 3) {
        tile_sizes = {4, 2, 32};
      } else if constexpr (rank == 4) {
        tile_sizes = {2, 2, 2, 32};
      } else if constexpr (rank == 5) {
        tile_sizes = {2, 1, 2, 2, 32};
      } else if constexpr (rank == 6) {
        tile_sizes = {2, 1, 2, 1, 2, 32};
      } else {
        for (int i = 0; i < rank; ++i) {
          tile_sizes[i] = 2;
        }
        tile_sizes[rank - 1] = 16;
      }
    }
    return tile_sizes;
  }
};

// Settings for TeamMDRangePolicy
template <typename Rank, TeamMDRangeThreadAndVector ThreadAndVector>
struct ThreadAndVectorNestLevel<Rank, Kokkos::SYCL, ThreadAndVector>
    : AcceleratorBasedNestLevel<Rank, ThreadAndVector> {};

}  // namespace Impl
}  // Namespace Kokkos
#endif
