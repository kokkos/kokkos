// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_EXP_ITERATE_TILE_GPU_HPP
#define KOKKOS_EXP_ITERATE_TILE_GPU_HPP

#include <Kokkos_Macros.hpp>

#include <algorithm>

#include <utility>

#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_ENABLE_SYCL
template <typename index_type>
struct EmulateCUDADim3 {
  index_type x;
  index_type y;
  index_type z;
};
#endif

template <class Tag, class Functor, class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION std::enable_if_t<std::is_void_v<Tag>>
_tag_invoke(Functor const& f, Args&&... args) {
  f(std::forward<Args>(args)...);
}

template <class Tag, class Functor, class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION std::enable_if_t<!std::is_void_v<Tag>>
_tag_invoke(Functor const& f, Args&&... args) {
  f(Tag{}, std::forward<Args>(args)...);
}

template <class Tag, class Functor, class T, size_t N, size_t... Idxs,
          class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION void _tag_invoke_array_helper(
    Functor const& f, T (&vals)[N], std::integer_sequence<size_t, Idxs...>,
    Args&&... args) {
  _tag_invoke<Tag>(f, vals[Idxs]..., std::forward<Args>(args)...);
}

template <class Tag, class Functor, class T, size_t N, class... Args>
KOKKOS_IMPL_FORCEINLINE_FUNCTION void _tag_invoke_array(Functor const& f,
                                                        T (&vals)[N],
                                                        Args&&... args) {
  _tag_invoke_array_helper<Tag>(f, vals, std::make_index_sequence<N>{},
                                std::forward<Args>(args)...);
}

// ------------------------------------------------------------------ //
// ParallelFor iteration pattern
template <int N, typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile;

// Rank 2
template <typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile<2, PolicyType, Functor, MaxGridSize, Tag> {
  using index_type = typename PolicyType::index_type;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_max_grid_size(max_grid_size_),
        gridDim(gridDim_),
        blockDim(blockDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_)
      : m_policy(policy_), m_func(f_), m_max_grid_size(max_grid_size_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    const index_type stride_0 = gridDim.x * blockDim.x;
    const index_type stride_1 = gridDim.y * blockDim.y;

    const index_type start_0 =
        blockIdx.x * blockDim.x + threadIdx.x + m_policy.m_lower[0];
    const index_type start_1 =
        blockIdx.y * blockDim.y + threadIdx.y + m_policy.m_lower[1];

    if constexpr (PolicyType::inner_direction == Iterate::Left) {
      for (index_type idx_1 = start_1;
           idx_1 < static_cast<index_type>(m_policy.m_upper[1]);
           idx_1 += stride_1) {
        for (index_type idx_0 = start_0;
             idx_0 < static_cast<index_type>(m_policy.m_upper[0]);
             idx_0 += stride_0) {
          Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1);
        }
      }
    } else {
      for (index_type idx_0 = start_0;
           idx_0 < static_cast<index_type>(m_policy.m_upper[0]);
           idx_0 += stride_0) {
        for (index_type idx_1 = start_1;
             idx_1 < static_cast<index_type>(m_policy.m_upper[1]);
             idx_1 += stride_1) {
          Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1);
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  const MaxGridSize& m_max_grid_size;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

// Rank 3
template <typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile<3, PolicyType, Functor, MaxGridSize, Tag> {
  using index_type = typename PolicyType::index_type;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_max_grid_size(max_grid_size_),
        gridDim(gridDim_),
        blockDim(blockDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_)
      : m_policy(policy_), m_func(f_), m_max_grid_size(max_grid_size_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    const index_type stride_0 = gridDim.x * blockDim.x;
    const index_type stride_1 = gridDim.y * blockDim.y;
    const index_type stride_2 = gridDim.z * blockDim.z;

    const index_type start_0 =
        blockIdx.x * blockDim.x + threadIdx.x + m_policy.m_lower[0];
    const index_type start_1 =
        blockIdx.y * blockDim.y + threadIdx.y + m_policy.m_lower[1];
    const index_type start_2 =
        blockIdx.z * blockDim.z + threadIdx.z + m_policy.m_lower[2];

    if constexpr (PolicyType::inner_direction == Iterate::Left) {
      for (index_type idx_2 = start_2;
           idx_2 < static_cast<index_type>(m_policy.m_upper[2]);
           idx_2 += stride_2) {
        for (index_type idx_1 = start_1;
             idx_1 < static_cast<index_type>(m_policy.m_upper[1]);
             idx_1 += stride_1) {
          for (index_type idx_0 = start_0;
               idx_0 < static_cast<index_type>(m_policy.m_upper[0]);
               idx_0 += stride_0) {
            Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2);
          }
        }
      }
    } else {
      for (index_type idx_0 = start_0;
           idx_0 < static_cast<index_type>(m_policy.m_upper[0]);
           idx_0 += stride_0) {
        for (index_type idx_1 = start_1;
             idx_1 < static_cast<index_type>(m_policy.m_upper[1]);
             idx_1 += stride_1) {
          for (index_type idx_2 = start_2;
               idx_2 < static_cast<index_type>(m_policy.m_upper[2]);
               idx_2 += stride_2) {
            Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2);
          }
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  const MaxGridSize& m_max_grid_size;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

// Rank 4
template <typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile<4, PolicyType, Functor, MaxGridSize, Tag> {
  using index_type = typename PolicyType::index_type;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_max_grid_size(max_grid_size_),
        gridDim(gridDim_),
        blockDim(blockDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_)
      : m_policy(policy_), m_func(f_), m_max_grid_size(max_grid_size_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    const index_type stride_2 = gridDim.y * blockDim.y;
    const index_type stride_3 = gridDim.z * blockDim.z;

    const index_type start_2 =
        blockIdx.y * blockDim.y + threadIdx.y + m_policy.m_lower[2];
    const index_type start_3 =
        blockIdx.z * blockDim.z + threadIdx.z + m_policy.m_lower[3];

    const index_type max_tiles_01 =
        m_policy.m_tile_end[0] * m_policy.m_tile_end[1];

    if constexpr (PolicyType::inner_direction == Iterate::Left) {
      const index_type thread_id_1 = threadIdx.x / m_policy.m_tile[0];
      const index_type thread_id_0 = threadIdx.x % m_policy.m_tile[0];

      for (index_type idx_3 = start_3;
           idx_3 < static_cast<index_type>(m_policy.m_upper[3]);
           idx_3 += stride_3) {
        for (index_type idx_2 = start_2;
             idx_2 < static_cast<index_type>(m_policy.m_upper[2]);
             idx_2 += stride_2) {
          // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
          for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
               tile_x += gridDim.x) {
            const index_type tile_1 = tile_x / m_policy.m_tile_end[0];
            const index_type tile_0 = tile_x % m_policy.m_tile_end[0];

            const index_type idx_1 =
                tile_1 * m_policy.m_tile[1] + thread_id_1 + m_policy.m_lower[1];
            const index_type idx_0 =
                tile_0 * m_policy.m_tile[0] + thread_id_0 + m_policy.m_lower[0];

            if (idx_1 < static_cast<index_type>(m_policy.m_upper[1]) &&
                idx_0 < static_cast<index_type>(m_policy.m_upper[0])) {
              Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3);
            }
          }
        }
      }

    } else {  // Iterate::Right

      const index_type thread_id_0 = threadIdx.x / m_policy.m_tile[1];
      const index_type thread_id_1 = threadIdx.x % m_policy.m_tile[1];

      // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
      for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
           tile_x += gridDim.x) {
        const index_type tile_0 = tile_x / m_policy.m_tile_end[1];
        const index_type tile_1 = tile_x % m_policy.m_tile_end[1];

        const index_type idx_0 =
            tile_0 * m_policy.m_tile[0] + thread_id_0 + m_policy.m_lower[0];
        const index_type idx_1 =
            tile_1 * m_policy.m_tile[1] + thread_id_1 + m_policy.m_lower[1];

        if (idx_0 < static_cast<index_type>(m_policy.m_upper[0]) &&
            idx_1 < static_cast<index_type>(m_policy.m_upper[1])) {
          for (index_type idx_2 = start_2;
               idx_2 < static_cast<index_type>(m_policy.m_upper[2]);
               idx_2 += stride_2) {
            for (index_type idx_3 = start_3;
                 idx_3 < static_cast<index_type>(m_policy.m_upper[3]);
                 idx_3 += stride_3) {
              Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3);
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  const MaxGridSize& m_max_grid_size;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

// Rank 5
template <typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile<5, PolicyType, Functor, MaxGridSize, Tag> {
  using index_type = typename PolicyType::index_type;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_max_grid_size(max_grid_size_),
        gridDim(gridDim_),
        blockDim(blockDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_)
      : m_policy(policy_), m_func(f_), m_max_grid_size(max_grid_size_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    const index_type start_4 =
        blockIdx.z * blockDim.z + threadIdx.z + m_policy.m_lower[4];
    const index_type stride_4 = gridDim.z * blockDim.z;

    const index_type max_tiles_01 =
        m_policy.m_tile_end[0] * m_policy.m_tile_end[1];
    const index_type max_tiles_23 =
        m_policy.m_tile_end[2] * m_policy.m_tile_end[3];

    if (PolicyType::inner_direction == Iterate::Left) {
      const index_type thread_id_3 = threadIdx.y / m_policy.m_tile[2];
      const index_type thread_id_2 = threadIdx.y % m_policy.m_tile[2];

      const index_type thread_id_1 = threadIdx.x / m_policy.m_tile[0];
      const index_type thread_id_0 = threadIdx.x % m_policy.m_tile[0];

      for (index_type idx_4 = start_4;
           idx_4 < static_cast<index_type>(m_policy.m_upper[4]);
           idx_4 += stride_4) {
        // Reconstruct tile 2 and tile 3 from blockIdx.y and threadIdx.y
        for (index_type tile_y = blockIdx.y; tile_y < max_tiles_23;
             tile_y += gridDim.y) {
          const index_type tile_3 = tile_y / m_policy.m_tile_end[2];
          const index_type tile_2 = tile_y % m_policy.m_tile_end[2];

          const index_type idx_3 =
              tile_3 * m_policy.m_tile[3] + thread_id_3 + m_policy.m_lower[3];
          const index_type idx_2 =
              tile_2 * m_policy.m_tile[2] + thread_id_2 + m_policy.m_lower[2];

          if (idx_3 < static_cast<index_type>(m_policy.m_upper[3]) &&
              idx_2 < static_cast<index_type>(m_policy.m_upper[2])) {
            // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
            for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
                 tile_x += gridDim.x) {
              const index_type tile_1 = tile_x / m_policy.m_tile_end[0];
              const index_type tile_0 = tile_x % m_policy.m_tile_end[0];

              const index_type idx_1 = tile_1 * m_policy.m_tile[1] +
                                       thread_id_1 + m_policy.m_lower[1];
              const index_type idx_0 = tile_0 * m_policy.m_tile[0] +
                                       thread_id_0 + m_policy.m_lower[0];

              if (idx_1 < static_cast<index_type>(m_policy.m_upper[1]) &&
                  idx_0 < static_cast<index_type>(m_policy.m_upper[0])) {
                Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3,
                                       idx_4);
              }
            }
          }
        }
      }

    } else {  // Iterate::Right

      const index_type thread_id_0 = threadIdx.x / m_policy.m_tile[1];
      const index_type thread_id_1 = threadIdx.x % m_policy.m_tile[1];

      const index_type thread_id_2 = threadIdx.y / m_policy.m_tile[3];
      const index_type thread_id_3 = threadIdx.y % m_policy.m_tile[3];

      // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
      for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
           tile_x += gridDim.x) {
        const index_type tile_0 = tile_x / m_policy.m_tile_end[1];
        const index_type tile_1 = tile_x % m_policy.m_tile_end[1];

        const index_type idx_1 =
            tile_1 * m_policy.m_tile[1] + thread_id_1 + m_policy.m_lower[1];
        const index_type idx_0 =
            tile_0 * m_policy.m_tile[0] + thread_id_0 + m_policy.m_lower[0];

        if (idx_0 < static_cast<index_type>(m_policy.m_upper[0]) &&
            idx_1 < static_cast<index_type>(m_policy.m_upper[1])) {
          // Reconstruct tile 2 and tile 3 from blockIdx.y and threadIdx.y
          for (index_type tile_y = blockIdx.y; tile_y < max_tiles_23;
               tile_y += gridDim.y) {
            const index_type tile_2 = tile_y / m_policy.m_tile_end[3];
            const index_type tile_3 = tile_y % m_policy.m_tile_end[3];

            const index_type idx_2 =
                tile_2 * m_policy.m_tile[2] + thread_id_2 + m_policy.m_lower[2];
            const index_type idx_3 =
                tile_3 * m_policy.m_tile[3] + thread_id_3 + m_policy.m_lower[3];

            if (idx_2 < static_cast<index_type>(m_policy.m_upper[2]) &&
                idx_3 < static_cast<index_type>(m_policy.m_upper[3])) {
              for (index_type idx_4 = start_4;
                   idx_4 < static_cast<index_type>(m_policy.m_upper[4]);
                   idx_4 += stride_4) {
                Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3,
                                       idx_4);
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  const MaxGridSize& m_max_grid_size;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

// Rank 6
template <typename PolicyType, typename Functor, typename MaxGridSize,
          typename Tag>
struct DeviceIterateTile<6, PolicyType, Functor, MaxGridSize, Tag> {
  using index_type = typename PolicyType::index_type;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_max_grid_size(max_grid_size_),
        gridDim(gridDim_),
        blockDim(blockDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_,
      const MaxGridSize& max_grid_size_)
      : m_policy(policy_), m_func(f_), m_max_grid_size(max_grid_size_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    const index_type max_tiles_01 =
        m_policy.m_tile_end[0] * m_policy.m_tile_end[1];
    const index_type max_tiles_23 =
        m_policy.m_tile_end[2] * m_policy.m_tile_end[3];
    const index_type max_tiles_45 =
        m_policy.m_tile_end[4] * m_policy.m_tile_end[5];

    if (PolicyType::inner_direction == Iterate::Left) {
      const index_type thread_id_5 = threadIdx.z / m_policy.m_tile[4];
      const index_type thread_id_4 = threadIdx.z % m_policy.m_tile[4];

      const index_type thread_id_3 = threadIdx.y / m_policy.m_tile[2];
      const index_type thread_id_2 = threadIdx.y % m_policy.m_tile[2];

      const index_type thread_id_1 = threadIdx.x / m_policy.m_tile[0];
      const index_type thread_id_0 = threadIdx.x % m_policy.m_tile[0];

      // Reconstruct tile 4 and tile 5 from blockIdx.z and threadIdx.z
      for (index_type tile_z = blockIdx.z; tile_z < max_tiles_45;
           tile_z += gridDim.z) {
        const index_type tile_5 = tile_z / m_policy.m_tile_end[4];
        const index_type tile_4 = tile_z % m_policy.m_tile_end[4];

        const index_type idx_5 =
            tile_5 * m_policy.m_tile[5] + thread_id_5 + m_policy.m_lower[5];
        const index_type idx_4 =
            tile_4 * m_policy.m_tile[4] + thread_id_4 + m_policy.m_lower[4];

        if (idx_5 < static_cast<index_type>(m_policy.m_upper[5]) &&
            idx_4 < static_cast<index_type>(m_policy.m_upper[4])) {
          // Reconstruct tile 2 and tile 3 from blockIdx.y and threadIdx.y
          for (index_type tile_y = blockIdx.y; tile_y < max_tiles_23;
               tile_y += gridDim.y) {
            const index_type tile_3 = tile_y / m_policy.m_tile_end[2];
            const index_type tile_2 = tile_y % m_policy.m_tile_end[2];

            const index_type idx_3 =
                tile_3 * m_policy.m_tile[3] + thread_id_3 + m_policy.m_lower[3];
            const index_type idx_2 =
                tile_2 * m_policy.m_tile[2] + thread_id_2 + m_policy.m_lower[2];

            if (idx_3 < static_cast<index_type>(m_policy.m_upper[3]) &&
                idx_2 < static_cast<index_type>(m_policy.m_upper[2])) {
              // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
              for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
                   tile_x += gridDim.x) {
                const index_type tile_1 = tile_x / m_policy.m_tile_end[0];
                const index_type tile_0 = tile_x % m_policy.m_tile_end[0];

                const index_type idx_1 = tile_1 * m_policy.m_tile[1] +
                                         thread_id_1 + m_policy.m_lower[1];
                const index_type idx_0 = tile_0 * m_policy.m_tile[0] +
                                         thread_id_0 + m_policy.m_lower[0];

                if (idx_1 < static_cast<index_type>(m_policy.m_upper[1]) &&
                    idx_0 < static_cast<index_type>(m_policy.m_upper[0])) {
                  Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3,
                                         idx_4, idx_5);
                }
              }
            }
          }
        }
      }

    } else {  // Iterate::Right

      const index_type thread_id_0 = threadIdx.x / m_policy.m_tile[1];
      const index_type thread_id_1 = threadIdx.x % m_policy.m_tile[1];

      const index_type thread_id_2 = threadIdx.y / m_policy.m_tile[3];
      const index_type thread_id_3 = threadIdx.y % m_policy.m_tile[3];

      const index_type thread_id_4 = threadIdx.z / m_policy.m_tile[5];
      const index_type thread_id_5 = threadIdx.z % m_policy.m_tile[5];

      // Reconstruct tile 0 and tile 1 from blockIdx.x and threadIdx.x
      for (index_type tile_x = blockIdx.x; tile_x < max_tiles_01;
           tile_x += gridDim.x) {
        const index_type tile_0 = tile_x / m_policy.m_tile_end[1];
        const index_type tile_1 = tile_x % m_policy.m_tile_end[1];

        const index_type idx_0 =
            tile_0 * m_policy.m_tile[0] + thread_id_0 + m_policy.m_lower[0];
        const index_type idx_1 =
            tile_1 * m_policy.m_tile[1] + thread_id_1 + m_policy.m_lower[1];

        if (idx_0 < static_cast<index_type>(m_policy.m_upper[0]) &&
            idx_1 < static_cast<index_type>(m_policy.m_upper[1])) {
          // Reconstruct tile 2 and tile 3 from blockIdx.y and threadIdx.y
          for (index_type tile_y = blockIdx.y; tile_y < max_tiles_23;
               tile_y += gridDim.y) {
            const index_type tile_2 = tile_y / m_policy.m_tile_end[3];
            const index_type tile_3 = tile_y % m_policy.m_tile_end[3];

            const index_type idx_2 =
                tile_2 * m_policy.m_tile[2] + thread_id_2 + m_policy.m_lower[2];
            const index_type idx_3 =
                tile_3 * m_policy.m_tile[3] + thread_id_3 + m_policy.m_lower[3];

            if (idx_2 < static_cast<index_type>(m_policy.m_upper[2]) &&
                idx_3 < static_cast<index_type>(m_policy.m_upper[3])) {
              // Reconstruct tile 4 and tile 5 from blockIdx.z and threadIdx.z
              for (index_type tile_z = blockIdx.z; tile_z < max_tiles_45;
                   tile_z += gridDim.z) {
                const index_type tile_4 = tile_z / m_policy.m_tile_end[5];
                const index_type tile_5 = tile_z % m_policy.m_tile_end[5];

                const index_type idx_4 = tile_4 * m_policy.m_tile[4] +
                                         thread_id_4 + m_policy.m_lower[4];
                const index_type idx_5 = tile_5 * m_policy.m_tile[5] +
                                         thread_id_5 + m_policy.m_lower[5];

                if (idx_4 < static_cast<index_type>(m_policy.m_upper[4]) &&
                    idx_5 < static_cast<index_type>(m_policy.m_upper[5])) {
                  Impl::_tag_invoke<Tag>(m_func, idx_0, idx_1, idx_2, idx_3,
                                         idx_4, idx_5);
                }
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  const MaxGridSize& m_max_grid_size;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

// ----------------------------------------------------------------------------------

namespace Reduce {

template <typename T>
struct is_array_type : std::false_type {
  using value_type = T;
};

template <typename T>
struct is_array_type<T*> : std::true_type {
  using value_type = T;
};

template <typename T>
struct is_array_type<T[]> : std::true_type {
  using value_type = T;
};

// ------------------------------------------------------------------ //

template <typename T>
using value_type_storage_t =
    std::conditional_t<is_array_type<T>::value, std::decay_t<T>,
                       std::add_lvalue_reference_t<T>>;

// ParallelReduce iteration pattern
// Scalar reductions

// num_blocks = min( num_tiles, max_num_blocks ); //i.e. determined by number of
// tiles and reduction algorithm constraints extract n-dim tile offsets (i.e.
// tile's global starting mulit-index) from the tileid = blockid using tile
// dimensions local indices within a tile extracted from (index_type)threadIdx_x
// using tile dims, constrained by blocksize combine tile and local id info for
// multi-dim global ids

// Pattern:
// Each block+thread is responsible for a tile+local_id combo (additional when
// striding by num_blocks)
// 1. create offset arrays
// 2. loop over number of tiles, striding by griddim (equal to num tiles, or max
// num blocks)
// 3. temps set for tile_idx and thrd_idx, which will be modified
// 4. if LL vs LR:
//      determine tile starting point offsets (multidim)
//      determine local index offsets (multidim)
//      concatentate tile offset + local offset for global multi-dim index
//    if offset withinin range bounds AND local offset within tile bounds, call
//    functor

template <int N, typename PolicyType, typename Functor, typename Tag,
          typename ValueType, typename Enable = void>
struct DeviceIterateTile {
  using index_type         = typename PolicyType::index_type;
  using value_type_storage = value_type_storage_t<ValueType>;

#ifdef KOKKOS_ENABLE_SYCL
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(
      const PolicyType& policy_, const Functor& f_, value_type_storage v_,
      const EmulateCUDADim3<index_type> gridDim_,
      const EmulateCUDADim3<index_type> blockIdx_,
      const EmulateCUDADim3<index_type> threadIdx_)
      : m_policy(policy_),
        m_func(f_),
        m_v(v_),
        gridDim(gridDim_),
        blockIdx(blockIdx_),
        threadIdx(threadIdx_) {}
#else
  KOKKOS_IMPL_DEVICE_FUNCTION DeviceIterateTile(const PolicyType& policy_,
                                                const Functor& f_,
                                                value_type_storage v_)
      : m_policy(policy_), m_func(f_), m_v(v_) {}
#endif

  KOKKOS_IMPL_DEVICE_FUNCTION
  void exec_range() const {
    if (static_cast<index_type>(blockIdx.x) < m_policy.m_num_tiles &&
        static_cast<index_type>(threadIdx.y) < m_policy.m_prod_tile_dims) {
      index_type m_offset[PolicyType::rank];  // tile starting global id offset
      index_type
          m_local_offset[PolicyType::rank];  // tile starting global id offset

      for (index_type tileidx = static_cast<index_type>(blockIdx.x);
           tileidx < m_policy.m_num_tiles; tileidx += gridDim.x) {
        index_type tile_idx =
            tileidx;  // temp because tile_idx will be modified while
                      // determining tile starting point offsets
        index_type thrd_idx = static_cast<index_type>(threadIdx.y);
        bool in_bounds      = true;

        // LL
        if (PolicyType::inner_direction == Iterate::Left) {
          for (int i = 0; i < PolicyType::rank; ++i) {
            m_offset[i] =
                (tile_idx % m_policy.m_tile_end[i]) * m_policy.m_tile[i] +
                m_policy.m_lower[i];
            tile_idx /= m_policy.m_tile_end[i];

            // tile-local indices identified with (index_type)threadIdx_y
            m_local_offset[i] = (thrd_idx % m_policy.m_tile[i]);
            thrd_idx /= m_policy.m_tile[i];

            m_offset[i] += m_local_offset[i];
            if (!(m_offset[i] < m_policy.m_upper[i] &&
                  m_local_offset[i] < m_policy.m_tile[i])) {
              in_bounds = false;
            }
          }
          if (in_bounds) {
            Impl::_tag_invoke_array<Tag>(m_func, m_offset, m_v);
          }
        }
        // LR
        else {
          for (int i = PolicyType::rank - 1; i >= 0; --i) {
            m_offset[i] =
                (tile_idx % m_policy.m_tile_end[i]) * m_policy.m_tile[i] +
                m_policy.m_lower[i];
            tile_idx /= m_policy.m_tile_end[i];

            // tile-local indices identified with (index_type)threadIdx_y
            m_local_offset[i] =
                (thrd_idx %
                 m_policy.m_tile[i]);  // Move this to first computation,
                                       // add to m_offset right away
            thrd_idx /= m_policy.m_tile[i];

            m_offset[i] += m_local_offset[i];
            if (!(m_offset[i] < m_policy.m_upper[i] &&
                  m_local_offset[i] < m_policy.m_tile[i])) {
              in_bounds = false;
            }
          }
          if (in_bounds) {
            Impl::_tag_invoke_array<Tag>(m_func, m_offset, m_v);
          }
        }
      }
    }
  }  // end exec_range

 private:
  const PolicyType& m_policy;
  const Functor& m_func;
  value_type_storage m_v;
#ifdef KOKKOS_ENABLE_SYCL
  const EmulateCUDADim3<index_type> gridDim;
  const EmulateCUDADim3<index_type> blockIdx;
  const EmulateCUDADim3<index_type> threadIdx;
#endif
};

}  // namespace Reduce
}  // namespace Impl
}  // namespace Kokkos
#endif
