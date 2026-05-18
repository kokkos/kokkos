// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_HIP_PARALLEL_FOR_MDRANGE_HPP
#define KOKKOS_HIP_PARALLEL_FOR_MDRANGE_HPP

#include <Kokkos_Parallel.hpp>

#include <HIP/Kokkos_HIP_BlockSize_Deduction.hpp>
#include <HIP/Kokkos_HIP_KernelLaunch.hpp>
#include <KokkosExp_MDRangePolicy.hpp>
#include <impl/KokkosExp_IterateTileGPU.hpp>

namespace Kokkos::Impl {

// Device closure for MDRange ParallelFor on HIP.
// Selects between stride and no-stride iteration patterns at compile time.
template <typename FunctorType, bool UseStride, typename... Traits>
class ParallelForMDRange;

template <typename FunctorType, bool UseStride, typename... Traits>
class ParallelForMDRange<FunctorType, UseStride,
                         Kokkos::MDRangePolicy<Traits...>> {
 public:
  using Policy       = Kokkos::MDRangePolicy<Traits...>;
  using functor_type = FunctorType;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using array_type       = typename Policy::point_type;

  using DeviceIteratePattern = std::conditional_t<
      UseStride,
      Kokkos::Impl::DeviceIterate<Policy::rank, array_index_type, index_type,
                                  FunctorType, Policy::inner_direction,
                                  typename Policy::work_tag>,
      Kokkos::Impl::DeviceIterateNoStride<
          Policy::rank, array_index_type, index_type, FunctorType,
          Policy::inner_direction, typename Policy::work_tag>>;

  const FunctorType m_functor;
  const array_type m_lower;
  const array_type m_upper;
  const array_type m_extent;  // tile_size * num_tiles

 public:
  ParallelForMDRange() = delete;

  inline __device__ void operator()() const {
    DeviceIteratePattern(m_lower, m_upper, m_extent, m_functor).exec_range();
  }

  ParallelForMDRange(FunctorType const& arg_functor, const array_type& lower,
                     const array_type& upper, const array_type& extent)
      : m_functor(arg_functor),
        m_lower(lower),
        m_upper(upper),
        m_extent(extent) {}
};

// ParallelFor
template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, HIP> {
 public:
  using Policy       = Kokkos::MDRangePolicy<Traits...>;
  using functor_type = FunctorType;

 private:
  using array_index_type = typename Policy::array_index_type;
  using index_type       = typename Policy::index_type;
  using LaunchBounds     = typename Policy::launch_bounds;
  using MaxGridSize      = Kokkos::Array<index_type, 3>;
  using array_type       = typename Policy::point_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const MaxGridSize m_max_grid_size;

  array_type m_lower;
  array_type m_upper;
  array_type m_extent;  // tile_size * num_tiles

 public:
  ParallelFor() = delete;

  inline __device__ void operator()() const {
    Kokkos::Impl::DeviceIterate<Policy::rank, array_index_type, index_type,
                                FunctorType, Policy::inner_direction,
                                typename Policy::work_tag>(m_lower, m_upper,
                                                           m_extent, m_functor)
        .exec_range();
  }

  inline void execute() const {
    if (m_policy.m_num_tiles == 0) return;

    const auto [grid, block] =
        Kokkos::Impl::compute_device_launch_params(m_policy, m_max_grid_size);

    // Check if the grid covers the full iteration space (no stride needed).
    using comp_t = std::common_type_t<index_type, array_index_type>;

    const comp_t max_grid_x = static_cast<comp_t>(m_max_grid_size[0]);
    const comp_t max_grid_y = static_cast<comp_t>(m_max_grid_size[1]);
    const comp_t max_grid_z = static_cast<comp_t>(m_max_grid_size[2]);
    const comp_t bx         = static_cast<comp_t>(block.x);
    const comp_t by         = static_cast<comp_t>(block.y);
    const comp_t bz         = static_cast<comp_t>(block.z);

    bool need_grid_stride = true;
    if constexpr (Policy::rank == 1) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0])) {
        need_grid_stride = false;
      }
    } else if constexpr (Policy::rank == 2) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0]) &&
          (max_grid_y * by) >= static_cast<comp_t>(m_extent[1])) {
        need_grid_stride = false;
      }
    } else if constexpr (Policy::rank == 3) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0]) &&
          (max_grid_y * by) >= static_cast<comp_t>(m_extent[1]) &&
          (max_grid_z * bz) >= static_cast<comp_t>(m_extent[2])) {
        need_grid_stride = false;
      }
    } else if constexpr (Policy::rank == 4) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0]) *
                                   static_cast<comp_t>(m_extent[1]) &&
          (max_grid_y * by) >= static_cast<comp_t>(m_extent[2]) &&
          (max_grid_z * bz) >= static_cast<comp_t>(m_extent[3])) {
        need_grid_stride = false;
      }
    } else if constexpr (Policy::rank == 5) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0]) *
                                   static_cast<comp_t>(m_extent[1]) &&
          (max_grid_y * by) >= static_cast<comp_t>(m_extent[2]) *
                                   static_cast<comp_t>(m_extent[3]) &&
          (max_grid_z * bz) >= static_cast<comp_t>(m_extent[4])) {
        need_grid_stride = false;
      }
    } else if constexpr (Policy::rank == 6) {
      if ((max_grid_x * bx) >= static_cast<comp_t>(m_extent[0]) *
                                   static_cast<comp_t>(m_extent[1]) &&
          (max_grid_y * by) >= static_cast<comp_t>(m_extent[2]) *
                                   static_cast<comp_t>(m_extent[3]) &&
          (max_grid_z * bz) >= static_cast<comp_t>(m_extent[4]) *
                                   static_cast<comp_t>(m_extent[5])) {
        need_grid_stride = false;
      }
    }

    if constexpr (Policy::is_graph_kernel::value) {
      hip_parallel_launch<ParallelFor, LaunchBounds>(
          *this, grid, block, 0,
          m_policy.space().impl_internal_space_instance(), false);
    } else {
      // launch the kernel
      if (need_grid_stride) {
        using ClosureType = ParallelForMDRange<FunctorType, true, Policy>;
        ClosureType closure(m_functor, m_lower, m_upper, m_extent);
        hip_parallel_launch<ClosureType, LaunchBounds>(
            closure, grid, block, 0,
            m_policy.space().impl_internal_space_instance(), false);
      } else {
        using ClosureType = ParallelForMDRange<FunctorType, false, Policy>;
        ClosureType closure(m_functor, m_lower, m_upper, m_extent);
        hip_parallel_launch<ClosureType, LaunchBounds>(
            closure, grid, block, 0,
            m_policy.space().impl_internal_space_instance(), false);
      }
    }
  }  // end execute

  ParallelFor(FunctorType const& arg_functor, Policy const& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_max_grid_size({
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[0]),
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[1]),
            static_cast<index_type>(
                m_policy.space().hip_device_prop().maxGridSize[2]),
        }) {
    // Initialize begins and ends based on layout
    // Swap the fastest indexes to x dimension
    for (array_index_type i = 0; i < Policy::rank; ++i) {
      if constexpr (Policy::inner_direction == Iterate::Left) {
        m_lower[i]  = m_policy.m_lower[i];
        m_upper[i]  = m_policy.m_upper[i];
        m_extent[i] = m_policy.m_tile[i] * m_policy.m_tile_end[i];
      } else {
        m_lower[i]  = m_policy.m_lower[Policy::rank - 1 - i];
        m_upper[i]  = m_policy.m_upper[Policy::rank - 1 - i];
        m_extent[i] = m_policy.m_tile[Policy::rank - 1 - i] *
                      m_policy.m_tile_end[Policy::rank - 1 - i];
      }
    }
  }

  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    using closure_type =
        ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>, HIP>;
    unsigned block_size = hip_get_max_blocksize<closure_type, LaunchBounds>();
    if (block_size == 0)
      Kokkos::Impl::throw_runtime_exception(
          std::string("Kokkos::Impl::ParallelFor< HIP > could not find a valid "
                      "tile size."));
    return block_size;
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_HIP_PARALLEL_FOR_MDRANGE_HPP
