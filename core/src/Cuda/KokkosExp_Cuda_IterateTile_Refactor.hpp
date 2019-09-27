/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CUDA_EXP_ITERATE_TILE_REFACTOR_HPP
#define KOKKOS_CUDA_EXP_ITERATE_TILE_REFACTOR_HPP

#include <Kokkos_Macros.hpp>
#if defined(__CUDACC__) && defined(KOKKOS_ENABLE_CUDA)

#include <iostream>
#include <algorithm>
#include <cstdio>

#include <utility>

// #include<Cuda/Kokkos_CudaExec.hpp>
// Including the file above leads to following type of errors:
// /home/ndellin/kokkos/core/src/Cuda/Kokkos_CudaExec.hpp(84): error: incomplete
// type is not allowed use existing Kokkos functionality, e.g. max blocks, once
// resolved

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>
#endif

namespace Kokkos {
namespace Impl {

template <class Tag, class Functor, class... Args>
__device__ __inline__ typename std::enable_if<std::is_void<Tag>::value>::type
_tag_invoke(Functor const& f, Args&&... args) {
  f((Args &&) args...);
}

template <class Tag, class Functor, class... Args>
__device__ __inline__ typename std::enable_if<!std::is_void<Tag>::value>::type
_tag_invoke(Functor const& f, Args&&... args) {
  f(Tag{}, (Args &&) args...);
}

template <class Tag, class Functor, class T, size_t N, size_t... Idxs, class... Args>
__device__ __inline__ void
_tag_invoke_array_helper(Functor const& f, T(& vals)[N], integer_sequence<size_t, Idxs...>, Args&&... args) {
  _tag_invoke<Tag>(f, vals[Idxs]..., (Args&&)args...);
}

template <class Tag, class Functor, class T, size_t N, class... Args>
__device__ __inline__ void
_tag_invoke_array(Functor const& f, T(& vals)[N], Args&&... args) {
  _tag_invoke_array_helper<Tag>(f, vals, make_index_sequence<N>{}, (Args &&) args...);
}

namespace Refactor {

// ------------------------------------------------------------------ //
// ParallelFor iteration pattern
template <int N, typename RP, typename Functor, typename Tag>
struct DeviceIterateTile;

// Rank 2
// Specializations for void tag type
template <typename RP, typename Functor, typename Tag>
struct DeviceIterateTile<2, RP, Functor, Tag> {
  using index_type = typename RP::index_type;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_)
      : m_rp(rp_), m_func(f_) {}

  inline __device__ void exec_range() const {
    // LL
    if (RP::inner_direction == RP::Left) {
      for (auto tile_id1 = (index_type)blockIdx.y;
           tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y) {
        const index_type offset_1 = tile_id1 * m_rp.m_tile[1] +
                                    (index_type)threadIdx.y +
                                    (index_type)m_rp.m_lower[1];
        if (offset_1 < m_rp.m_upper[1] &&
            (index_type)threadIdx.y < m_rp.m_tile[1]) {
          for (auto tile_id0 = (index_type)blockIdx.x;
               tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x) {
            const index_type offset_0 = tile_id0 * m_rp.m_tile[0] +
                                        (index_type)threadIdx.x +
                                        (index_type)m_rp.m_lower[0];
            if (offset_0 < m_rp.m_upper[0] &&
                (index_type)threadIdx.x < m_rp.m_tile[0]) {
              Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1);
            }
          }
        }
      }
    }
    // LR
    else {
      for (auto tile_id0 = (index_type)blockIdx.x;
           tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x) {
        const index_type offset_0 = tile_id0 * m_rp.m_tile[0] +
                                    (index_type)threadIdx.x +
                                    (index_type)m_rp.m_lower[0];
        if (offset_0 < m_rp.m_upper[0] &&
            (index_type)threadIdx.x < m_rp.m_tile[0]) {
          for (auto tile_id1 = (index_type)blockIdx.y;
               tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y) {
            const index_type offset_1 = tile_id1 * m_rp.m_tile[1] +
                                        (index_type)threadIdx.y +
                                        (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] &&
                (index_type)threadIdx.y < m_rp.m_tile[1]) {
              Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1);
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const RP& m_rp;
  const Functor& m_func;
};

// Rank 3
template <typename RP, typename Functor, typename Tag>
struct DeviceIterateTile<3, RP, Functor, Tag> {
  using index_type = typename RP::index_type;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_)
      : m_rp(rp_), m_func(f_) {}

  inline __device__ void exec_range() const {
    // LL
    if (RP::inner_direction == RP::Left) {
      for (auto tile_id2 = (index_type)blockIdx.z;
           tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z) {
        const index_type offset_2 = tile_id2 * m_rp.m_tile[2] +
                                    (index_type)threadIdx.z +
                                    (index_type)m_rp.m_lower[2];
        if (offset_2 < m_rp.m_upper[2] &&
            (index_type)threadIdx.z < m_rp.m_tile[2]) {
          for (auto tile_id1 = (index_type)blockIdx.y;
               tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y) {
            const index_type offset_1 = tile_id1 * m_rp.m_tile[1] +
                                        (index_type)threadIdx.y +
                                        (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] &&
                (index_type)threadIdx.y < m_rp.m_tile[1]) {
              for (auto tile_id0 = (index_type)blockIdx.x;
                   tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x) {
                const index_type offset_0 = tile_id0 * m_rp.m_tile[0] +
                                            (index_type)threadIdx.x +
                                            (index_type)m_rp.m_lower[0];
                if (offset_0 < m_rp.m_upper[0] &&
                    (index_type)threadIdx.x < m_rp.m_tile[0]) {
                  Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1, offset_2);
                }
              }
            }
          }
        }
      }
    }
    // LR
    else {
      for (auto tile_id0 = (index_type)blockIdx.x;
           tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x) {
        const index_type offset_0 = tile_id0 * m_rp.m_tile[0] +
                                    (index_type)threadIdx.x +
                                    (index_type)m_rp.m_lower[0];
        if (offset_0 < m_rp.m_upper[0] &&
            (index_type)threadIdx.x < m_rp.m_tile[0]) {
          for (auto tile_id1 = (index_type)blockIdx.y;
               tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y) {
            const index_type offset_1 = tile_id1 * m_rp.m_tile[1] +
                                        (index_type)threadIdx.y +
                                        (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] &&
                (index_type)threadIdx.y < m_rp.m_tile[1]) {
              for (auto tile_id2 = (index_type)blockIdx.z;
                   tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z) {
                const index_type offset_2 = tile_id2 * m_rp.m_tile[2] +
                                            (index_type)threadIdx.z +
                                            (index_type)m_rp.m_lower[2];
                if (offset_2 < m_rp.m_upper[2] &&
                    (index_type)threadIdx.z < m_rp.m_tile[2]) {
                  Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1, offset_2);
                }
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const RP& m_rp;
  const Functor& m_func;
};

// Rank 4
template <typename RP, typename Functor, typename Tag>
struct DeviceIterateTile<4, RP, Functor, Tag> {
  using index_type = typename RP::index_type;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_)
      : m_rp(rp_), m_func(f_) {}

  static constexpr index_type max_blocks = 65535;
  // static constexpr index_type max_blocks =
  // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount);

  inline __device__ void exec_range() const {
    // enum { max_blocks =
    // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount) };
    // const index_type max_blocks = static_cast<index_type>(
    // Kokkos::Impl::cuda_internal_maximum_grid_count() );
    // LL
    if (RP::inner_direction == RP::Left) {
      const index_type temp0  = m_rp.m_tile_end[0];
      const index_type temp1  = m_rp.m_tile_end[1];
      const index_type numbl0 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl1 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl0)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x % numbl0;
      const index_type tile_id1 = (index_type)blockIdx.x / numbl0;
      const index_type thr_id0  = (index_type)threadIdx.x % m_rp.m_tile[0];
      const index_type thr_id1  = (index_type)threadIdx.x / m_rp.m_tile[0];

      for (index_type tile_id3 = (index_type)blockIdx.z;
           tile_id3 < m_rp.m_tile_end[3]; tile_id3 += gridDim.z) {
        const index_type offset_3 = tile_id3 * m_rp.m_tile[3] +
                                    (index_type)threadIdx.z +
                                    (index_type)m_rp.m_lower[3];
        if (offset_3 < m_rp.m_upper[3] &&
            (index_type)threadIdx.z < m_rp.m_tile[3]) {
          for (index_type tile_id2 = (index_type)blockIdx.y;
               tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.y) {
            const index_type offset_2 = tile_id2 * m_rp.m_tile[2] +
                                        (index_type)threadIdx.y +
                                        (index_type)m_rp.m_lower[2];
            if (offset_2 < m_rp.m_upper[2] &&
                (index_type)threadIdx.y < m_rp.m_tile[2]) {
              for (index_type j = tile_id1; j < m_rp.m_tile_end[1];
                   j += numbl1) {
                const index_type offset_1 =
                    j * m_rp.m_tile[1] + thr_id1 + (index_type)m_rp.m_lower[1];
                if (offset_1 < m_rp.m_upper[1] && thr_id1 < m_rp.m_tile[1]) {
                  for (index_type i = tile_id0; i < m_rp.m_tile_end[0];
                       i += numbl0) {
                    const index_type offset_0 = i * m_rp.m_tile[0] + thr_id0 +
                                                (index_type)m_rp.m_lower[0];
                    if (offset_0 < m_rp.m_upper[0] &&
                        thr_id0 < m_rp.m_tile[0]) {
                      Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                             offset_2, offset_3);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // LR
    else {
      const index_type temp0  = m_rp.m_tile_end[0];
      const index_type temp1  = m_rp.m_tile_end[1];
      const index_type numbl1 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl0 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl1)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x / numbl1;
      const index_type tile_id1 = (index_type)blockIdx.x % numbl1;
      const index_type thr_id0  = (index_type)threadIdx.x / m_rp.m_tile[1];
      const index_type thr_id1  = (index_type)threadIdx.x % m_rp.m_tile[1];

      for (index_type i = tile_id0; i < m_rp.m_tile_end[0]; i += numbl0) {
        const index_type offset_0 =
            i * m_rp.m_tile[0] + thr_id0 + (index_type)m_rp.m_lower[0];
        if (offset_0 < m_rp.m_upper[0] && thr_id0 < m_rp.m_tile[0]) {
          for (index_type j = tile_id1; j < m_rp.m_tile_end[1]; j += numbl1) {
            const index_type offset_1 =
                j * m_rp.m_tile[1] + thr_id1 + (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] && thr_id1 < m_rp.m_tile[1]) {
              for (index_type tile_id2 = (index_type)blockIdx.y;
                   tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.y) {
                const index_type offset_2 = tile_id2 * m_rp.m_tile[2] +
                                            (index_type)threadIdx.y +
                                            (index_type)m_rp.m_lower[2];
                if (offset_2 < m_rp.m_upper[2] &&
                    (index_type)threadIdx.y < m_rp.m_tile[2]) {
                  for (index_type tile_id3 = (index_type)blockIdx.z;
                       tile_id3 < m_rp.m_tile_end[3]; tile_id3 += gridDim.z) {
                    const index_type offset_3 = tile_id3 * m_rp.m_tile[3] +
                                                (index_type)threadIdx.z +
                                                (index_type)m_rp.m_lower[3];
                    if (offset_3 < m_rp.m_upper[3] &&
                        (index_type)threadIdx.z < m_rp.m_tile[3]) {
                      Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                             offset_2, offset_3);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const RP& m_rp;
  const Functor& m_func;
};

// Rank 5
template <typename RP, typename Functor, typename Tag>
struct DeviceIterateTile<5, RP, Functor, Tag> {
  using index_type = typename RP::index_type;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_)
      : m_rp(rp_), m_func(f_) {}

  static constexpr index_type max_blocks = 65535;
  // static constexpr index_type max_blocks =
  // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount);

  inline __device__ void exec_range() const {
    // enum { max_blocks =
    // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount) };
    // const index_type max_blocks = static_cast<index_type>(
    // Kokkos::Impl::cuda_internal_maximum_grid_count() );
    // LL
    if (RP::inner_direction == RP::Left) {
      index_type temp0        = m_rp.m_tile_end[0];
      index_type temp1        = m_rp.m_tile_end[1];
      const index_type numbl0 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl1 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl0)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x % numbl0;
      const index_type tile_id1 = (index_type)blockIdx.x / numbl0;
      const index_type thr_id0  = (index_type)threadIdx.x % m_rp.m_tile[0];
      const index_type thr_id1  = (index_type)threadIdx.x / m_rp.m_tile[0];

      temp0                   = m_rp.m_tile_end[2];
      temp1                   = m_rp.m_tile_end[3];
      const index_type numbl2 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl3 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl2)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id2 = (index_type)blockIdx.y % numbl2;
      const index_type tile_id3 = (index_type)blockIdx.y / numbl2;
      const index_type thr_id2  = (index_type)threadIdx.y % m_rp.m_tile[2];
      const index_type thr_id3  = (index_type)threadIdx.y / m_rp.m_tile[2];

      for (index_type tile_id4 = (index_type)blockIdx.z;
           tile_id4 < m_rp.m_tile_end[4]; tile_id4 += gridDim.z) {
        const index_type offset_4 = tile_id4 * m_rp.m_tile[4] +
                                    (index_type)threadIdx.z +
                                    (index_type)m_rp.m_lower[4];
        if (offset_4 < m_rp.m_upper[4] &&
            (index_type)threadIdx.z < m_rp.m_tile[4]) {
          for (index_type l = tile_id3; l < m_rp.m_tile_end[3]; l += numbl3) {
            const index_type offset_3 =
                l * m_rp.m_tile[3] + thr_id3 + (index_type)m_rp.m_lower[3];
            if (offset_3 < m_rp.m_upper[3] && thr_id3 < m_rp.m_tile[3]) {
              for (index_type k = tile_id2; k < m_rp.m_tile_end[2];
                   k += numbl2) {
                const index_type offset_2 =
                    k * m_rp.m_tile[2] + thr_id2 + (index_type)m_rp.m_lower[2];
                if (offset_2 < m_rp.m_upper[2] && thr_id2 < m_rp.m_tile[2]) {
                  for (index_type j = tile_id1; j < m_rp.m_tile_end[1];
                       j += numbl1) {
                    const index_type offset_1 = j * m_rp.m_tile[1] + thr_id1 +
                                                (index_type)m_rp.m_lower[1];
                    if (offset_1 < m_rp.m_upper[1] &&
                        thr_id1 < m_rp.m_tile[1]) {
                      for (index_type i = tile_id0; i < m_rp.m_tile_end[0];
                           i += numbl0) {
                        const index_type offset_0 = i * m_rp.m_tile[0] +
                                                    thr_id0 +
                                                    (index_type)m_rp.m_lower[0];
                        if (offset_0 < m_rp.m_upper[0] &&
                            thr_id0 < m_rp.m_tile[0]) {
                          Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                                 offset_2, offset_3, offset_4);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // LR
    else {
      index_type temp0        = m_rp.m_tile_end[0];
      index_type temp1        = m_rp.m_tile_end[1];
      const index_type numbl1 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl0 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl1)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x / numbl1;
      const index_type tile_id1 = (index_type)blockIdx.x % numbl1;
      const index_type thr_id0  = (index_type)threadIdx.x / m_rp.m_tile[1];
      const index_type thr_id1  = (index_type)threadIdx.x % m_rp.m_tile[1];

      temp0                   = m_rp.m_tile_end[2];
      temp1                   = m_rp.m_tile_end[3];
      const index_type numbl3 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl2 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl3)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id2 = (index_type)blockIdx.y / numbl3;
      const index_type tile_id3 = (index_type)blockIdx.y % numbl3;
      const index_type thr_id2  = (index_type)threadIdx.y / m_rp.m_tile[3];
      const index_type thr_id3  = (index_type)threadIdx.y % m_rp.m_tile[3];

      for (index_type i = tile_id0; i < m_rp.m_tile_end[0]; i += numbl0) {
        const index_type offset_0 =
            i * m_rp.m_tile[0] + thr_id0 + (index_type)m_rp.m_lower[0];
        if (offset_0 < m_rp.m_upper[0] && thr_id0 < m_rp.m_tile[0]) {
          for (index_type j = tile_id1; j < m_rp.m_tile_end[1]; j += numbl1) {
            const index_type offset_1 =
                j * m_rp.m_tile[1] + thr_id1 + (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] && thr_id1 < m_rp.m_tile[1]) {
              for (index_type k = tile_id2; k < m_rp.m_tile_end[2];
                   k += numbl2) {
                const index_type offset_2 =
                    k * m_rp.m_tile[2] + thr_id2 + (index_type)m_rp.m_lower[2];
                if (offset_2 < m_rp.m_upper[2] && thr_id2 < m_rp.m_tile[2]) {
                  for (index_type l = tile_id3; l < m_rp.m_tile_end[3];
                       l += numbl3) {
                    const index_type offset_3 = l * m_rp.m_tile[3] + thr_id3 +
                                                (index_type)m_rp.m_lower[3];
                    if (offset_3 < m_rp.m_upper[3] &&
                        thr_id3 < m_rp.m_tile[3]) {
                      for (index_type tile_id4 = (index_type)blockIdx.z;
                           tile_id4 < m_rp.m_tile_end[4];
                           tile_id4 += gridDim.z) {
                        const index_type offset_4 = tile_id4 * m_rp.m_tile[4] +
                                                    (index_type)threadIdx.z +
                                                    (index_type)m_rp.m_lower[4];
                        if (offset_4 < m_rp.m_upper[4] &&
                            (index_type)threadIdx.z < m_rp.m_tile[4]) {
                          Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                                 offset_2, offset_3, offset_4);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const RP& m_rp;
  const Functor& m_func;
};

// Rank 6
// Specializations for void tag type
template <typename RP, typename Functor, typename Tag>
struct DeviceIterateTile<6, RP, Functor, Tag> {
  using index_type = typename RP::index_type;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_)
      : m_rp(rp_), m_func(f_) {}

  static constexpr index_type max_blocks = 65535;
  // static constexpr index_type max_blocks =
  // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount);

  inline __device__ void exec_range() const {
    // enum { max_blocks =
    // static_cast<index_type>(Kokkos::Impl::CudaTraits::UpperBoundGridCount) };
    // const index_type max_blocks = static_cast<index_type>(
    // Kokkos::Impl::cuda_internal_maximum_grid_count() );
    // LL
    if (RP::inner_direction == RP::Left) {
      index_type temp0        = m_rp.m_tile_end[0];
      index_type temp1        = m_rp.m_tile_end[1];
      const index_type numbl0 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl1 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl0)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x % numbl0;
      const index_type tile_id1 = (index_type)blockIdx.x / numbl0;
      const index_type thr_id0  = (index_type)threadIdx.x % m_rp.m_tile[0];
      const index_type thr_id1  = (index_type)threadIdx.x / m_rp.m_tile[0];

      temp0                   = m_rp.m_tile_end[2];
      temp1                   = m_rp.m_tile_end[3];
      const index_type numbl2 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl3 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl2)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id2 = (index_type)blockIdx.y % numbl2;
      const index_type tile_id3 = (index_type)blockIdx.y / numbl2;
      const index_type thr_id2  = (index_type)threadIdx.y % m_rp.m_tile[2];
      const index_type thr_id3  = (index_type)threadIdx.y / m_rp.m_tile[2];

      temp0                   = m_rp.m_tile_end[4];
      temp1                   = m_rp.m_tile_end[5];
      const index_type numbl4 = (temp0 <= max_blocks ? temp0 : max_blocks);
      const index_type numbl5 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl4)
               : (temp1 <= max_blocks ? temp1 : max_blocks));

      const index_type tile_id4 = (index_type)blockIdx.z % numbl4;
      const index_type tile_id5 = (index_type)blockIdx.z / numbl4;
      const index_type thr_id4  = (index_type)threadIdx.z % m_rp.m_tile[4];
      const index_type thr_id5  = (index_type)threadIdx.z / m_rp.m_tile[4];

      for (index_type n = tile_id5; n < m_rp.m_tile_end[5]; n += numbl5) {
        const index_type offset_5 =
            n * m_rp.m_tile[5] + thr_id5 + (index_type)m_rp.m_lower[5];
        if (offset_5 < m_rp.m_upper[5] && thr_id5 < m_rp.m_tile[5]) {
          for (index_type m = tile_id4; m < m_rp.m_tile_end[4]; m += numbl4) {
            const index_type offset_4 =
                m * m_rp.m_tile[4] + thr_id4 + (index_type)m_rp.m_lower[4];
            if (offset_4 < m_rp.m_upper[4] && thr_id4 < m_rp.m_tile[4]) {
              for (index_type l = tile_id3; l < m_rp.m_tile_end[3];
                   l += numbl3) {
                const index_type offset_3 =
                    l * m_rp.m_tile[3] + thr_id3 + (index_type)m_rp.m_lower[3];
                if (offset_3 < m_rp.m_upper[3] && thr_id3 < m_rp.m_tile[3]) {
                  for (index_type k = tile_id2; k < m_rp.m_tile_end[2];
                       k += numbl2) {
                    const index_type offset_2 = k * m_rp.m_tile[2] + thr_id2 +
                                                (index_type)m_rp.m_lower[2];
                    if (offset_2 < m_rp.m_upper[2] &&
                        thr_id2 < m_rp.m_tile[2]) {
                      for (index_type j = tile_id1; j < m_rp.m_tile_end[1];
                           j += numbl1) {
                        const index_type offset_1 = j * m_rp.m_tile[1] +
                                                    thr_id1 +
                                                    (index_type)m_rp.m_lower[1];
                        if (offset_1 < m_rp.m_upper[1] &&
                            thr_id1 < m_rp.m_tile[1]) {
                          for (index_type i = tile_id0; i < m_rp.m_tile_end[0];
                               i += numbl0) {
                            const index_type offset_0 =
                                i * m_rp.m_tile[0] + thr_id0 +
                                (index_type)m_rp.m_lower[0];
                            if (offset_0 < m_rp.m_upper[0] &&
                                thr_id0 < m_rp.m_tile[0]) {
                              Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                                     offset_2, offset_3,
                                                     offset_4, offset_5);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // LR
    else {
      index_type temp0        = m_rp.m_tile_end[0];
      index_type temp1        = m_rp.m_tile_end[1];
      const index_type numbl1 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl0 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl1)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id0 = (index_type)blockIdx.x / numbl1;
      const index_type tile_id1 = (index_type)blockIdx.x % numbl1;
      const index_type thr_id0  = (index_type)threadIdx.x / m_rp.m_tile[1];
      const index_type thr_id1  = (index_type)threadIdx.x % m_rp.m_tile[1];

      temp0                   = m_rp.m_tile_end[2];
      temp1                   = m_rp.m_tile_end[3];
      const index_type numbl3 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl2 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl3)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id2 = (index_type)blockIdx.y / numbl3;
      const index_type tile_id3 = (index_type)blockIdx.y % numbl3;
      const index_type thr_id2  = (index_type)threadIdx.y / m_rp.m_tile[3];
      const index_type thr_id3  = (index_type)threadIdx.y % m_rp.m_tile[3];

      temp0                   = m_rp.m_tile_end[4];
      temp1                   = m_rp.m_tile_end[5];
      const index_type numbl5 = (temp1 <= max_blocks ? temp1 : max_blocks);
      const index_type numbl4 =
          (temp0 * temp1 > max_blocks
               ? index_type(max_blocks / numbl5)
               : (temp0 <= max_blocks ? temp0 : max_blocks));

      const index_type tile_id4 = (index_type)blockIdx.z / numbl5;
      const index_type tile_id5 = (index_type)blockIdx.z % numbl5;
      const index_type thr_id4  = (index_type)threadIdx.z / m_rp.m_tile[5];
      const index_type thr_id5  = (index_type)threadIdx.z % m_rp.m_tile[5];

      for (index_type i = tile_id0; i < m_rp.m_tile_end[0]; i += numbl0) {
        const index_type offset_0 =
            i * m_rp.m_tile[0] + thr_id0 + (index_type)m_rp.m_lower[0];
        if (offset_0 < m_rp.m_upper[0] && thr_id0 < m_rp.m_tile[0]) {
          for (index_type j = tile_id1; j < m_rp.m_tile_end[1]; j += numbl1) {
            const index_type offset_1 =
                j * m_rp.m_tile[1] + thr_id1 + (index_type)m_rp.m_lower[1];
            if (offset_1 < m_rp.m_upper[1] && thr_id1 < m_rp.m_tile[1]) {
              for (index_type k = tile_id2; k < m_rp.m_tile_end[2];
                   k += numbl2) {
                const index_type offset_2 =
                    k * m_rp.m_tile[2] + thr_id2 + (index_type)m_rp.m_lower[2];
                if (offset_2 < m_rp.m_upper[2] && thr_id2 < m_rp.m_tile[2]) {
                  for (index_type l = tile_id3; l < m_rp.m_tile_end[3];
                       l += numbl3) {
                    const index_type offset_3 = l * m_rp.m_tile[3] + thr_id3 +
                                                (index_type)m_rp.m_lower[3];
                    if (offset_3 < m_rp.m_upper[3] &&
                        thr_id3 < m_rp.m_tile[3]) {
                      for (index_type m = tile_id4; m < m_rp.m_tile_end[4];
                           m += numbl4) {
                        const index_type offset_4 = m * m_rp.m_tile[4] +
                                                    thr_id4 +
                                                    (index_type)m_rp.m_lower[4];
                        if (offset_4 < m_rp.m_upper[4] &&
                            thr_id4 < m_rp.m_tile[4]) {
                          for (index_type n = tile_id5; n < m_rp.m_tile_end[5];
                               n += numbl5) {
                            const index_type offset_5 =
                                n * m_rp.m_tile[5] + thr_id5 +
                                (index_type)m_rp.m_lower[5];
                            if (offset_5 < m_rp.m_upper[5] &&
                                thr_id5 < m_rp.m_tile[5]) {
                              Impl::_tag_invoke<Tag>(m_func, offset_0, offset_1,
                                                     offset_2, offset_3,
                                                     offset_4, offset_5);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }  // end exec_range

 private:
  const RP& m_rp;
  const Functor& m_func;
};

}  // namespace Refactor

// ----------------------------------------------------------------------------------

namespace Reduce {

template <typename T>
using is_void = std::is_same<T, void>;

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
    typename std::conditional<is_array_type<T>::value, std::decay<T>,
                              std::add_lvalue_reference<T> >::type::type;

// ParallelReduce iteration pattern
// Scalar reductions

// num_blocks = min( num_tiles, max_num_blocks ); //i.e. determined by number of
// tiles and reduction algorithm constraints extract n-dim tile offsets (i.e.
// tile's global starting mulit-index) from the tileid = blockid using tile
// dimensions local indices within a tile extracted from (index_type)threadIdx.x
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

template <int N, typename RP, typename Functor, typename Tag,
    typename ValueType, typename Enable = void>
struct DeviceIterateTile {
  using index_type         = typename RP::index_type;
  using value_type_storage = value_type_storage_t<ValueType>;

  __device__ DeviceIterateTile(const RP& rp_, const Functor& f_,
                               value_type_storage v_)
      : m_rp(rp_), m_func(f_), m_v(v_) {}

  inline __device__ void exec_range() const {
    if ((index_type)blockIdx.x < m_rp.m_num_tiles &&
        (index_type)threadIdx.y < m_rp.m_prod_tile_dims) {
      index_type m_offset[RP::rank];        // tile starting global id offset
      index_type m_local_offset[RP::rank];  // tile starting global id offset

      for (index_type tileidx = (index_type)blockIdx.x;
           tileidx < m_rp.m_num_tiles; tileidx += gridDim.x) {
        // temp because tile_idx will be modified while
        // determining tile starting point offsets
        index_type tile_idx = tileidx;
        index_type thrd_idx = (index_type)threadIdx.y;
        bool in_bounds      = true;

        // LL
        if (RP::inner_direction == RP::Left) {
          for (int i = 0; i < RP::rank; ++i) {
            // Deduce this blocks tile_id
            m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] +
                          m_rp.m_lower[i];
            tile_idx /= m_rp.m_tile_end[i];

            m_local_offset[i] = (thrd_idx % m_rp.m_tile[i]);
            thrd_idx /= m_rp.m_tile[i];

            m_offset[i] += m_local_offset[i];
            if (!(m_offset[i] < m_rp.m_upper[i] &&
                  m_local_offset[i] < m_rp.m_tile[i])) {
              in_bounds = false;
            }
          }
          if (in_bounds) {
            Impl::_tag_invoke_array<Tag>(m_func, m_offset, m_v);
          }
        }
        // LR
        else {
          for (int i = RP::rank - 1; i >= 0; --i) {
            m_offset[i] = (tile_idx % m_rp.m_tile_end[i]) * m_rp.m_tile[i] +
                          m_rp.m_lower[i];
            tile_idx /= m_rp.m_tile_end[i];

            m_local_offset[i] = (thrd_idx % m_rp.m_tile[i]);
            thrd_idx /= m_rp.m_tile[i];

            m_offset[i] += m_local_offset[i];
            if (!(m_offset[i] < m_rp.m_upper[i] &&
                  m_local_offset[i] < m_rp.m_tile[i])) {
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
  const RP& m_rp;
  const Functor& m_func;
  value_type_storage m_v;
};

}  // namespace Reduce

// ----------------------------------------------------------------------------------

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
