/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
#define KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

#include <initializer_list>

#include <Kokkos_Layout.hpp>
#include <Kokkos_Array.hpp>
#include <impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>
#include <type_traits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/KokkosExp_Cuda_IterateTile.hpp>
#include <Cuda/KokkosExp_Cuda_IterateTile_Refactor.hpp>
#endif

#if defined(__HIPCC__) && defined(KOKKOS_ENABLE_HIP)
#include <HIP/KokkosExp_HIP_IterateTile.hpp>
#endif

namespace Kokkos {

// ------------------------------------------------------------------ //
// Moved to Kokkos_Layout.hpp for more general accessibility
/*
enum class Iterate
{
  Default, // Default for the device
  Left,    // Left indices stride fastest
  Right,   // Right indices stride fastest
};
*/

template <typename ExecSpace>
struct default_outer_direction {
  using type = Iterate;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static constexpr Iterate value = Iterate::Left;
#else
  static constexpr Iterate value = Iterate::Right;
#endif
};

template <typename ExecSpace>
struct default_inner_direction {
  using type = Iterate;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static constexpr Iterate value = Iterate::Left;
#else
  static constexpr Iterate value = Iterate::Right;
#endif
};

// Iteration Pattern
template <unsigned N, Iterate OuterDir = Iterate::Default,
          Iterate InnerDir = Iterate::Default>
struct Rank {
  static_assert(N != 0u, "Kokkos Error: rank 0 undefined");
  static_assert(N != 1u,
                "Kokkos Error: rank 1 is not a multi-dimensional range");
  static_assert(N < 7u, "Kokkos Error: Unsupported rank...");

  using iteration_pattern = Rank<N, OuterDir, InnerDir>;

  static constexpr int rank                = N;
  static constexpr Iterate outer_direction = OuterDir;
  static constexpr Iterate inner_direction = InnerDir;
};

namespace Impl {
// NOTE prefer C array U[M] to std::initalizer_list<U> so that the number of
// elements can be deduced (https://stackoverflow.com/q/40241370)
template <class Array, class U, std::size_t M>
constexpr Array to_array_potentially_narrowing(const U (&init)[M]) {
  using T = typename Array::value_type;
  Array a{};
  constexpr std::size_t N = a.size();
  static_assert(M <= N, "");
  auto* ptr = a.data();
  // NOTE equivalent to
  // std::transform(std::begin(init), std::end(init), a.data(),
  //                [](U x) { return static_cast<T>(x); });
  // except that std::transform is not constexpr.
  for (auto x : init)
    *ptr++ = static_cast<T>(x);  // allow narrowing conversions
  return a;
}
}  // namespace Impl

// multi-dimensional iteration pattern
template <typename... Properties>
struct MDRangePolicy : public Kokkos::Impl::PolicyTraits<Properties...> {
  using traits       = Kokkos::Impl::PolicyTraits<Properties...>;
  using range_policy = RangePolicy<Properties...>;

  typename traits::execution_space m_space;

  using impl_range_policy =
      RangePolicy<typename traits::execution_space,
                  typename traits::schedule_type, typename traits::index_type>;

  using execution_policy =
      MDRangePolicy<Properties...>;  // needed for is_execution_space
                                     // interrogation

  template <class... OtherProperties>
  friend struct MDRangePolicy;

  static_assert(!std::is_same<typename traits::iteration_pattern, void>::value,
                "Kokkos Error: MD iteration pattern not defined");

  using iteration_pattern = typename traits::iteration_pattern;
  using work_tag          = typename traits::work_tag;
  using launch_bounds     = typename traits::launch_bounds;
  using member_type       = typename range_policy::member_type;

  enum { rank = static_cast<int>(iteration_pattern::rank) };

  using index_type       = typename traits::index_type;
  using array_index_type = long;
  using point_type = Kokkos::Array<array_index_type, rank>;  // was index_type
  using tile_type  = Kokkos::Array<array_index_type, rank>;
  // If point_type or tile_type is not templated on a signed integral type (if
  // it is unsigned), then if user passes in intializer_list of
  // runtime-determined values of signed integral type that are not const will
  // receive a compiler error due to an invalid case for implicit conversion -
  // "conversion from integer or unscoped enumeration type to integer type that
  // cannot represent all values of the original, except where source is a
  // constant expression whose value can be stored exactly in the target type"
  // This would require the user to either pass a matching index_type parameter
  // as template parameter to the MDRangePolicy or static_cast the individual
  // values

  point_type m_lower          = {};
  point_type m_upper          = {};
  tile_type m_tile            = {};
  point_type m_tile_end       = {};
  index_type m_num_tiles      = 1;
  index_type m_prod_tile_dims = 1;
  bool m_tune_tile_size       = false;

  /*
    // NDE enum impl definition alternative - replace static constexpr int ?
    enum { outer_direction = static_cast<int> (
        (iteration_pattern::outer_direction != Iterate::Default)
      ? iteration_pattern::outer_direction
      : default_outer_direction< typename traits::execution_space>::value ) };

    enum { inner_direction = static_cast<int> (
        iteration_pattern::inner_direction != Iterate::Default
      ? iteration_pattern::inner_direction
      : default_inner_direction< typename traits::execution_space>::value ) };

    enum { Right = static_cast<int>( Iterate::Right ) };
    enum { Left  = static_cast<int>( Iterate::Left ) };
  */
  // static constexpr int rank = iteration_pattern::rank;

  static constexpr int outer_direction = static_cast<int>(
      (iteration_pattern::outer_direction != Iterate::Default)
          ? iteration_pattern::outer_direction
          : default_outer_direction<typename traits::execution_space>::value);

  static constexpr int inner_direction = static_cast<int>(
      iteration_pattern::inner_direction != Iterate::Default
          ? iteration_pattern::inner_direction
          : default_inner_direction<typename traits::execution_space>::value);

  // Ugly ugly workaround intel 14 not handling scoped enum correctly
  static constexpr int Right = static_cast<int>(Iterate::Right);
  static constexpr int Left  = static_cast<int>(Iterate::Left);

  KOKKOS_INLINE_FUNCTION const typename traits::execution_space& space() const {
    return m_space;
  }

  MDRangePolicy() = default;

  template <typename LT, std::size_t LN, typename UT, std::size_t UN,
            typename TT = array_index_type, std::size_t TN = rank,
            typename = std::enable_if_t<std::is_integral<LT>::value &&
                                        std::is_integral<UT>::value &&
                                        std::is_integral<TT>::value>>
  MDRangePolicy(const LT (&lower)[LN], const UT (&upper)[UN],
                const TT (&tile)[TN] = {})
      : MDRangePolicy(
            Impl::to_array_potentially_narrowing<decltype(m_lower)>(lower),
            Impl::to_array_potentially_narrowing<decltype(m_upper)>(upper),
            Impl::to_array_potentially_narrowing<decltype(m_tile)>(tile)) {
    static_assert(
        LN == rank && UN == rank && TN <= rank,
        "MDRangePolicy: Constructor initializer lists have wrong size");
  }

  template <typename LT, std::size_t LN, typename UT, std::size_t UN,
            typename TT = array_index_type, std::size_t TN = rank,
            typename = std::enable_if_t<std::is_integral<LT>::value &&
                                        std::is_integral<UT>::value &&
                                        std::is_integral<TT>::value>>
  MDRangePolicy(const typename traits::execution_space& work_space,
                const LT (&lower)[LN], const UT (&upper)[UN],
                const TT (&tile)[TN] = {})
      : MDRangePolicy(
            work_space,
            Impl::to_array_potentially_narrowing<decltype(m_lower)>(lower),
            Impl::to_array_potentially_narrowing<decltype(m_upper)>(upper),
            Impl::to_array_potentially_narrowing<decltype(m_tile)>(tile)) {
    static_assert(
        LN == rank && UN == rank && TN <= rank,
        "MDRangePolicy: Constructor initializer lists have wrong size");
  }

  MDRangePolicy(point_type const& lower, point_type const& upper,
                tile_type const& tile = tile_type{})
      : MDRangePolicy(typename traits::execution_space(), lower, upper, tile) {}

  MDRangePolicy(const typename traits::execution_space& work_space,
                point_type const& lower, point_type const& upper,
                tile_type const& tile = tile_type{})
      : m_space(work_space), m_lower(lower), m_upper(upper), m_tile(tile) {
    init();
  }

  template <class... OtherProperties>
  MDRangePolicy(const MDRangePolicy<OtherProperties...> p)
      : m_space(p.m_space),
        m_lower(p.m_lower),
        m_upper(p.m_upper),
        m_tile(p.m_tile),
        m_tile_end(p.m_tile_end),
        m_num_tiles(p.m_num_tiles),
        m_prod_tile_dims(p.m_prod_tile_dims) {}

  void impl_change_tile_size(const point_type& tile) {
    m_tile = tile;
    init(m_lower, m_upper, m_tile);
  }
  bool impl_tune_tile_size() const { return m_tune_tile_size; }

 private:
  void init() {
    // Host
    if (true
#if defined(KOKKOS_ENABLE_CUDA)
        && !std::is_same<typename traits::execution_space, Kokkos::Cuda>::value
#endif
#if defined(KOKKOS_ENABLE_HIP)
        && !std::is_same<typename traits::execution_space,
                         Kokkos::Experimental::HIP>::value
#endif
    ) {
      index_type span;
      for (int i = 0; i < rank; ++i) {
        span = m_upper[i] - m_lower[i];
        if (m_tile[i] <= 0) {
          m_tune_tile_size = true;
          if (((int)inner_direction == (int)Right && (i < rank - 1)) ||
              ((int)inner_direction == (int)Left && (i > 0))) {
            m_tile[i] = 2;
          } else {
            m_tile[i] = (span == 0 ? 1 : span);
          }
        }
        m_tile_end[i] =
            static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    else  // Cuda or HIP
    {
      index_type span;
      int increment  = 1;
      int rank_start = 0;
      int rank_end   = rank;
      if ((int)inner_direction == (int)Right) {
        increment  = -1;
        rank_start = rank - 1;
        rank_end   = -1;
      }
      bool is_cuda_exec_space =
#if defined(KOKKOS_ENABLE_CUDA)
          std::is_same<typename traits::execution_space, Kokkos::Cuda>::value;
#else
          false;
#endif
      for (int i = rank_start; i != rank_end; i += increment) {
        span = m_upper[i] - m_lower[i];
        if (m_tile[i] <= 0) {
          // TODO: determine what is a good default tile size for Cuda and HIP
          // may be rank dependent
          m_tune_tile_size = true;
          if (((int)inner_direction == (int)Right && (i < rank - 1)) ||
              ((int)inner_direction == (int)Left && (i > 0))) {
            if (m_prod_tile_dims < 256) {
              m_tile[i] = (is_cuda_exec_space) ? 2 : 4;
            } else {
              m_tile[i] = 1;
            }
          } else {
            m_tile[i] = 16;
          }
        }
        m_tile_end[i] =
            static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
      if (m_prod_tile_dims >
          1024) {  // Match Cuda restriction for ParallelReduce; 1024,1024,64
                   // max per dim (Kepler), but product num_threads < 1024
        if (is_cuda_exec_space) {
          printf(" Tile dimensions exceed Cuda limits\n");
          Kokkos::abort(
              "Cuda ExecSpace Error: MDRange tile dims exceed maximum number "
              "of threads per block - choose smaller tile dims");
        } else {
          printf(" Tile dimensions exceed HIP limits\n");
          Kokkos::abort(
              "HIP ExecSpace Error: MDRange tile dims exceed maximum number of "
              "threads per block - choose smaller tile dims");
        }
      }
    }
#endif
  }
};

}  // namespace Kokkos

// For backward compatibility
namespace Kokkos {
namespace Experimental {
using Kokkos::Iterate;
using Kokkos::MDRangePolicy;
using Kokkos::Rank;
}  // namespace Experimental
}  // namespace Kokkos
// ------------------------------------------------------------------ //

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <unsigned long P, class... Properties>
struct PolicyPropertyAdaptor<WorkItemProperty::ImplWorkItemProperty<P>,
                             MDRangePolicy<Properties...>> {
  using policy_in_t = MDRangePolicy<Properties...>;
  using policy_out_t =
      MDRangePolicy<typename policy_in_t::traits::execution_space,
                    typename policy_in_t::traits::schedule_type,
                    typename policy_in_t::traits::work_tag,
                    typename policy_in_t::traits::index_type,
                    typename policy_in_t::traits::iteration_pattern,
                    typename policy_in_t::traits::launch_bounds,
                    WorkItemProperty::ImplWorkItemProperty<P>>;
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
