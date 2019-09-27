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

#ifndef KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
#define KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

#include <initializer_list>

#include <Kokkos_Layout.hpp>

#include <impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>

#if defined(__CUDACC__) && defined(KOKKOS_ENABLE_CUDA)
#include <Cuda/KokkosExp_Cuda_IterateTile.hpp>
#include <Cuda/KokkosExp_Cuda_IterateTile_Refactor.hpp>
#endif

#if defined(__HCC__) && defined(KOKKOS_ENABLE_ROCM)
//#include<ROCm/KokkosExp_ROCm_IterateTile.hpp>
#include <ROCm/KokkosExp_ROCm_IterateTile_Refactor.hpp>
#endif

namespace Kokkos {
namespace Impl {

// Workaround for narrowing conversion warnings
template <class IndexType>
struct _ignore_narrowing_index_wrapper {
  IndexType value;

  template <class T,
            int = typename std::enable_if<
                std::is_convertible<T, IndexType>::value, int>::type{0}>
  /* intentionally implicit */
  KOKKOS_FORCEINLINE_FUNCTION _ignore_narrowing_index_wrapper(
      T arg_value) noexcept
      : value(arg_value) {}

  // workaround for Cuda linker bug??!?
  KOKKOS_INLINE_FUNCTION _ignore_narrowing_index_wrapper() noexcept
      : value(IndexType{}) {}
  KOKKOS_INLINE_FUNCTION _ignore_narrowing_index_wrapper(
      _ignore_narrowing_index_wrapper const&) noexcept = default;
  KOKKOS_INLINE_FUNCTION _ignore_narrowing_index_wrapper(
      _ignore_narrowing_index_wrapper&&) noexcept = default;

  KOKKOS_INLINE_FUNCTION _ignore_narrowing_index_wrapper& operator=(
      _ignore_narrowing_index_wrapper const&) noexcept = default;
  KOKKOS_INLINE_FUNCTION _ignore_narrowing_index_wrapper& operator=(
      _ignore_narrowing_index_wrapper&&) noexcept = default;

  KOKKOS_INLINE_FUNCTION ~_ignore_narrowing_index_wrapper() noexcept = default;

  KOKKOS_INLINE_FUNCTION operator IndexType() noexcept { return value; }
  KOKKOS_INLINE_FUNCTION operator IndexType const() const noexcept {
    return value;
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

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
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_ROCM)
  static constexpr Iterate value = Iterate::Left;
#else
  static constexpr Iterate value = Iterate::Right;
#endif
};

template <typename ExecSpace>
struct default_inner_direction {
  using type = Iterate;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_ROCM)
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

// multi-dimensional iteration pattern
template <typename... Properties>
struct MDRangePolicy : public Kokkos::Impl::PolicyTraits<Properties...> {
 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Public member types and constexpr data members"> {{{2

  using traits       = Kokkos::Impl::PolicyTraits<Properties...>;
  using range_policy = RangePolicy<Properties...>;
  using impl_range_policy =
      RangePolicy<typename traits::execution_space,
                  typename traits::schedule_type, typename traits::index_type>;
  // needed for is_execution_policy interrogation
  using execution_policy = MDRangePolicy<Properties...>;

  using iteration_pattern = typename traits::iteration_pattern;
  using work_tag          = typename traits::work_tag;
  using launch_bounds     = typename traits::launch_bounds;
  using member_type       = typename range_policy::member_type;

  enum { rank = static_cast<int>(iteration_pattern::rank) };

  using index_type       = typename traits::index_type;
  using array_index_type = typename std::make_signed<index_type>::type;
  using execution_space  = typename traits::execution_space;
  using point_type =
      Kokkos::Array<Impl::_ignore_narrowing_index_wrapper<index_type>, rank>;
  using tile_type = point_type;

  static constexpr int outer_direction = static_cast<int>(
      (iteration_pattern::outer_direction != Iterate::Default)
          ? iteration_pattern::outer_direction
          : default_outer_direction<typename traits::execution_space>::value);

  static constexpr int inner_direction = static_cast<int>(
      iteration_pattern::inner_direction != Iterate::Default
          ? iteration_pattern::inner_direction
          : default_inner_direction<typename traits::execution_space>::value);

  static constexpr int Right = static_cast<int>(Iterate::Right);
  static constexpr int Left  = static_cast<int>(Iterate::Left);

  // </editor-fold> end Public member types and constexpr data members }}}2
  //----------------------------------------------------------------------------

 private:
  //------------------------------------------------------------------------------
  // <editor-fold desc="Private member types"> {{{2

  // If point_type or tile_type is not templated on a signed integral type (if
  // it is unsigned), then if user passes in intializer_list of
  // runtime-determined values of signed integral type that are not a constant
  // expression will receive a compiler error due to an invalid case for
  // implicit conversion - "conversion from integer or unscoped enumeration type
  // to integer type that cannot represent all values of the original, except
  // where source is a constant expression whose value can be stored exactly in
  // the target type"  This would require the user to either pass a matching
  // index_type parameter as template parameter to the MDRangePolicy or
  // static_cast the individual values

  // TODO this doesn't need to be replicated for every MDRangePolicy
  //      specialization; consider moving it to a free function or something

  // </editor-fold> end Private member types }}}2
  //------------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // <editor-fold desc="Friends"> {{{2

  template <class... OtherProperties>
  friend struct MDRangePolicy;

  // for now, make Impl::ParallelFor, etc., friends
  // TODO design and implement a public interface for the things ParallelFor
  //      and friends need

  template <class, class, class>
  friend class Kokkos::Impl::ParallelFor;

  template <class, class, class, class>
  friend class Kokkos::Impl::ParallelReduce;

  template <class, class, class>
  friend class Kokkos::Impl::ParallelScan;

  template <class, class, class, class>
  friend class Kokkos::Impl::ParallelScanWithTotal;

  template <class, class, class, class, class>
  friend struct Kokkos::Impl::HostIterateTile;

#if defined(KOKKOS_ENABLE_CUDA)
  template <int, class, class, class>
  friend struct Kokkos::Impl::Refactor::DeviceIterateTile;
  template <int, class, class, class, class, class>
  friend struct Kokkos::Impl::Reduce::DeviceIterateTile;
#endif

  // </editor-fold> end Friends }}}2
  //----------------------------------------------------------------------------

  static_assert(!std::is_same<typename traits::iteration_pattern, void>::value,
                "Kokkos Error: MD iteration pattern not defined");

 public:
 private:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Private data members"> {{{2

  using _point_type = Kokkos::Array<index_type, rank>;
  using _tile_type  = _point_type;

  execution_space m_space;
  _point_type m_lower;
  _point_type m_upper;
  _tile_type m_tile;
  _point_type m_tile_end;
  index_type m_num_tiles;
  index_type m_prod_tile_dims;

  // </editor-fold> end Private data members }}}2
  //----------------------------------------------------------------------------

 public:
  //----------------------------------------------------------------------------
  // <editor-fold desc="Constructors, destructor, and assignment"> {{{2

  MDRangePolicy(point_type const& lower, point_type const& upper,
                tile_type const& tile = {
                    /* workaround for nvcc compiler/linker bug */ 0})
      : MDRangePolicy(execution_space{}, lower, upper, tile) {
    /* forwarding ctor, must be empty */
  }

  MDRangePolicy(execution_space const& work_space, point_type const& lower,
                point_type const& upper,
                tile_type const& tile = {
                    /* workaround for nvcc compiler/linker bug */ 0})
      : m_space(work_space),
        m_lower(_get_narrowed_values(lower)),
        m_upper(_get_narrowed_values(upper)),
        m_tile(_get_narrowed_values(tile)),
        m_num_tiles(1),
        m_prod_tile_dims(1) {
    init();
  }

  template <class... OtherProperties>
  MDRangePolicy(const MDRangePolicy<OtherProperties...> p) noexcept
      : m_space(p.m_space),
        m_lower(p.m_lower),
        m_upper(p.m_upper),
        m_tile(p.m_tile),
        m_tile_end(p.m_tile_end),
        m_num_tiles(p.m_num_tiles),
        m_prod_tile_dims(p.m_prod_tile_dims) {}

  // </editor-fold> end Constructors, destructor, and assignment }}}2
  //----------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION const execution_space& space() const {
    return m_space;
  }

 private:
  // TODO this doesn't need to be replicated for every MDRangePolicy
  //      specialization; consider moving it to a free function or something
  static _point_type _get_narrowed_values(point_type const& a) noexcept {
    // We don't have access to std::integer_sequence, so we have to do this
    _point_type rv = {/* workaround for nvcc compiler/linker bug */ 0};
    for (int i = 0; i < rank; ++i) {
      // TODO should be a safe narrowing_cast template like in GSL
      rv[i] = static_cast<index_type>(a[i].value);
    }
    return rv;
  }

  void init() {
    // Host
    if (!std::is_same<execution_space, Kokkos::Cuda>::value ||
        !std::is_same<execution_space, Kokkos::Experimental::ROCm>::value) {
      index_type span;
      for (int i = 0; i < rank; ++i) {
        span = m_upper[i] - m_lower[i];
        if (m_tile[i] <= 0) {
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
    } else  // Cuda or ROCm
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
      for (int i = rank_start; i != rank_end; i += increment) {
        span = m_upper[i] - m_lower[i];
        if (m_tile[i] <= 0) {
          // TODO: determine what is a good default tile size for cuda/rocm
          // may be rank dependent
          if (((int)inner_direction == (int)Right && (i < rank - 1)) ||
              ((int)inner_direction == (int)Left && (i > 0))) {
            if (m_prod_tile_dims < 256) {
#if defined(KOKKOS_ENABLE_CUDA)
              m_tile[i] = 2;
#elif defined(KOKKOS_ENABLE_ROCM)
              m_tile[i] = 4;
#endif
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
        printf(" Tile dimensions exceed Cuda limits\n");
        Kokkos::abort(
#if defined(KOKKOS_ENABLE_CUDA)
            " Cuda"
#elif defined(KOKKOS_ENABLE_ROCM)
            " ROCm"
#endif
            " ExecSpace Error: MDRange tile dims exceed maximum number of "
            "threads per block - choose smaller tile dims");
      }
    }
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

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
// ------------------------------------------------------------------ //
// md_parallel_for - deprecated use parallel_for
// ------------------------------------------------------------------ //

namespace Kokkos {
namespace Experimental {

template <typename MDRange, typename Functor, typename Enable = void>
void md_parallel_for(
    MDRange const& range, Functor const& f, const std::string& str = "",
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Cuda>::value
#endif
#if defined(KOKKOS_ENABLE_ROCM)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Experimental::ROCm>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for(range_policy(0, range.m_num_tiles).set_chunk_size(1), g,
                       str);
}

template <typename MDRange, typename Functor>
void md_parallel_for(
    const std::string& str, MDRange const& range, Functor const& f,
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Cuda>::value
#endif
#if defined(KOKKOS_ENABLE_ROCM)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Experimental::ROCm>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for(range_policy(0, range.m_num_tiles).set_chunk_size(1), g,
                       str);
}

// Cuda specialization
#if defined(__CUDACC__) && defined(KOKKOS_ENABLE_CUDA)
template <typename MDRange, typename Functor>
void md_parallel_for(
    const std::string& str, MDRange const& range, Functor const& f,
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && std::is_same<typename MDRange::range_policy::execution_space,
                         Kokkos::Cuda>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag>
      closure(range, f);
  closure.execute();
}

template <typename MDRange, typename Functor>
void md_parallel_for(
    MDRange const& range, Functor const& f, const std::string& str = "",
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && std::is_same<typename MDRange::range_policy::execution_space,
                         Kokkos::Cuda>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag>
      closure(range, f);
  closure.execute();
}
#endif
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
// md_parallel_reduce - deprecated use parallel_reduce
// ------------------------------------------------------------------ //
template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce(
    MDRange const& range, Functor const& f, ValueType& v,
    const std::string& str = "",
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Cuda>::value
#endif
#if defined(KOKKOS_ENABLE_ROCM)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Experimental::ROCm>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, ValueType> g(range,
                                                                       f);

  using range_policy = typename MDRange::impl_range_policy;
  Kokkos::parallel_reduce(
      str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v);
}

template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce(
    const std::string& str, MDRange const& range, Functor const& f,
    ValueType& v,
    typename std::enable_if<
        (true
#if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Cuda>::value
#endif
#if defined(KOKKOS_ENABLE_ROCM)
         && !std::is_same<typename MDRange::range_policy::execution_space,
                          Kokkos::Experimental::ROCm>::value
#endif
         )>::type* = 0) {
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, ValueType> g(range,
                                                                       f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_reduce(
      str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v);
}

// Cuda - md_parallel_reduce not implemented - use parallel_reduce

}  // namespace Experimental
}  // namespace Kokkos
#endif

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <unsigned long P, class... Properties>
struct PolicyPropertyAdaptor<WorkItemProperty::ImplWorkItemProperty<P>,
                             MDRangePolicy<Properties...>> {
  typedef MDRangePolicy<Properties...> policy_in_t;
  typedef MDRangePolicy<typename policy_in_t::traits::execution_space,
                        typename policy_in_t::traits::schedule_type,
                        typename policy_in_t::traits::work_tag,
                        typename policy_in_t::traits::index_type,
                        typename policy_in_t::traits::iteration_pattern,
                        typename policy_in_t::traits::launch_bounds,
                        WorkItemProperty::ImplWorkItemProperty<P>>
      policy_out_t;
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
