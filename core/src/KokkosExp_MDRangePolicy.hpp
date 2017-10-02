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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP
#define KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

#include <initializer_list>

#include<impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>

#if defined( __CUDACC__ ) && defined( KOKKOS_ENABLE_CUDA )
#include<Cuda/KokkosExp_Cuda_IterateTile.hpp>
#include <Cuda/KokkosExp_Cuda_IterateTile_Refactor.hpp>
#endif

namespace Kokkos {

// ------------------------------------------------------------------ //

enum class Iterate
{
  Default, // Default for the device
  Left,    // Left indices stride fastest
  Right,   // Right indices stride fastest
};

template <typename ExecSpace>
struct default_outer_direction
{
  using type = Iterate;
  #if defined( KOKKOS_ENABLE_CUDA)
  static constexpr Iterate value = Iterate::Left;
  #else
  static constexpr Iterate value = Iterate::Right;
  #endif
};

template <typename ExecSpace>
struct default_inner_direction
{
  using type = Iterate;
  #if defined( KOKKOS_ENABLE_CUDA)
  static constexpr Iterate value = Iterate::Left;
  #else
  static constexpr Iterate value = Iterate::Right;
  #endif
};


// Iteration Pattern
template < unsigned N
         , Iterate OuterDir = Iterate::Default
         , Iterate InnerDir = Iterate::Default
         >
struct Rank
{
  static_assert( N != 0u, "Kokkos Error: rank 0 undefined");
  static_assert( N != 1u, "Kokkos Error: rank 1 is not a multi-dimensional range");
  static_assert( N < 7u, "Kokkos Error: Unsupported rank...");

  using iteration_pattern = Rank<N, OuterDir, InnerDir>;

  static constexpr int rank = N;
  static constexpr Iterate outer_direction = OuterDir;
  static constexpr Iterate inner_direction = InnerDir;
};


// multi-dimensional iteration pattern
template <typename... Properties>
struct MDRangePolicy
  : public Kokkos::Impl::PolicyTraits<Properties ...>
{
  using traits = Kokkos::Impl::PolicyTraits<Properties ...>;
  using range_policy = RangePolicy<Properties...>;

  using impl_range_policy = RangePolicy< typename traits::execution_space
                                       , typename traits::schedule_type
                                       , typename traits::index_type
                                       > ;

  typedef MDRangePolicy execution_policy; // needed for is_execution_space interrogation

  static_assert( !std::is_same<typename traits::iteration_pattern,void>::value
               , "Kokkos Error: MD iteration pattern not defined" );

  using iteration_pattern   = typename traits::iteration_pattern;
  using work_tag            = typename traits::work_tag;
  using launch_bounds       = typename traits::launch_bounds;
  using member_type = typename range_policy::member_type;

  enum { rank = static_cast<int>(iteration_pattern::rank) };

  using index_type  = typename traits::index_type;
  using array_index_type = long;
  using point_type  = Kokkos::Array<array_index_type,rank>; //was index_type
  using tile_type   = Kokkos::Array<array_index_type,rank>;
  // If point_type or tile_type is not templated on a signed integral type (if it is unsigned), 
  // then if user passes in intializer_list of runtime-determined values of 
  // signed integral type that are not const will receive a compiler error due 
  // to an invalid case for implicit conversion - 
  // "conversion from integer or unscoped enumeration type to integer type that cannot represent all values of the original, except where source is a constant expression whose value can be stored exactly in the target type"
  // This would require the user to either pass a matching index_type parameter
  // as template parameter to the MDRangePolicy or static_cast the individual values

  point_type m_lower;
  point_type m_upper;
  tile_type  m_tile;
  point_type m_tile_end;
  index_type m_num_tiles;
  index_type m_prod_tile_dims;

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
  //static constexpr int rank = iteration_pattern::rank;

  static constexpr int outer_direction = static_cast<int> (
      (iteration_pattern::outer_direction != Iterate::Default)
    ? iteration_pattern::outer_direction
    : default_outer_direction< typename traits::execution_space>::value );

  static constexpr int inner_direction = static_cast<int> (
      iteration_pattern::inner_direction != Iterate::Default
    ? iteration_pattern::inner_direction
    : default_inner_direction< typename traits::execution_space>::value ) ;

  // Ugly ugly workaround intel 14 not handling scoped enum correctly
  static constexpr int Right = static_cast<int>( Iterate::Right );
  static constexpr int Left  = static_cast<int>( Iterate::Left );

  MDRangePolicy( point_type const& lower, point_type const& upper, tile_type const& tile = tile_type{} )
    : m_lower(lower)
    , m_upper(upper)
    , m_tile(tile)
    , m_num_tiles(1)
    , m_prod_tile_dims(1)
  {
    // Host
    if ( true
       #if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same< typename traits::execution_space, Kokkos::Cuda >::value
       #endif
       )
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = upper[i] - lower[i];
        if ( m_tile[i] <= 0 ) {
          if (  ((int)inner_direction == (int)Right && (i < rank-1))
              || ((int)inner_direction == (int)Left && (i > 0)) )
          {
            m_tile[i] = 2;
          }
          else {
            m_tile[i] = span;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
    }
    #if defined(KOKKOS_ENABLE_CUDA)
    else // Cuda
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = m_upper[i] - m_lower[i];
        if ( m_tile[i] <= 0 ) {
          // TODO: determine what is a good default tile size for cuda
          // may be rank dependent
          if (  ((int)inner_direction == (int)Right && (i < rank-1))
              || ((int)inner_direction == (int)Left && (i > 0)) )
          {
            if ( m_prod_tile_dims < 512 ) {
              m_tile[i] = 2;
            } else {
              m_tile[i] = 1;
            }
          }
          else {
            m_tile[i] = 16;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
      if ( m_prod_tile_dims > 512 ) { // Match Cuda restriction for ParallelReduce; 1024,1024,64 max per dim (Kepler), but product num_threads < 1024
        printf(" Tile dimensions exceed Cuda limits\n");
        Kokkos::abort(" Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
        //Kokkos::Impl::throw_runtime_exception( " Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
      }
    }
    #endif
  }


  template < typename LT , typename UT , typename TT = array_index_type >
  MDRangePolicy( std::initializer_list<LT> const& lower, std::initializer_list<UT> const& upper, std::initializer_list<TT> const& tile = {} )
  {

    if(static_cast<int>(m_lower.size()) != rank || static_cast<int>(m_upper.size()) != rank)
      Kokkos::abort("MDRangePolicy: Constructor initializer lists have wrong size");

    for ( auto i = 0; i < rank; ++i ) {
      m_lower[i] = static_cast<array_index_type>(lower.begin()[i]);
      m_upper[i] = static_cast<array_index_type>(upper.begin()[i]);
      if(static_cast<int>(tile.size())==rank)
        m_tile[i] = static_cast<array_index_type>(tile.begin()[i]);
      else
        m_tile[i] = 0;
    }

    m_num_tiles = 1;
    m_prod_tile_dims = 1;

    // Host
    if ( true
       #if defined(KOKKOS_ENABLE_CUDA)
         && !std::is_same< typename traits::execution_space, Kokkos::Cuda >::value
       #endif
       )
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = m_upper[i] - m_lower[i];
        if ( m_tile[i] <= 0 ) {
          if (  ((int)inner_direction == (int)Right && (i < rank-1))
              || ((int)inner_direction == (int)Left && (i > 0)) )
          {
            m_tile[i] = 2;
          }
          else {
            m_tile[i] = span;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
    }
    #if defined(KOKKOS_ENABLE_CUDA)
    else // Cuda
    {
      index_type span;
      for (int i=0; i<rank; ++i) {
        span = m_upper[i] - m_lower[i];
        if ( m_tile[i] <= 0 ) {
          // TODO: determine what is a good default tile size for cuda
          // may be rank dependent
          if (  ((int)inner_direction == (int)Right && (i < rank-1))
              || ((int)inner_direction == (int)Left && (i > 0)) )
          {
            if ( m_prod_tile_dims < 512 ) {
              m_tile[i] = 2;
            } else {
              m_tile[i] = 1;
            }
          }
          else {
            m_tile[i] = 16;
          }
        }
        m_tile_end[i] = static_cast<index_type>((span + m_tile[i] - 1) / m_tile[i]);
        m_num_tiles *= m_tile_end[i];
        m_prod_tile_dims *= m_tile[i];
      }
      if ( m_prod_tile_dims > 512 ) { // Match Cuda restriction for ParallelReduce; 1024,1024,64 max per dim (Kepler), but product num_threads < 1024
        printf(" Tile dimensions exceed Cuda limits\n");
        Kokkos::abort(" Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
        //Kokkos::Impl::throw_runtime_exception( " Cuda ExecSpace Error: MDRange tile dims exceed maximum number of threads per block - choose smaller tile dims");
      }
    }
    #endif
  }

};

} // namespace Kokkos

// For backward compatibility
namespace Kokkos { namespace Experimental {
  using Kokkos::MDRangePolicy;
  using Kokkos::Rank;
  using Kokkos::Iterate;
} } // end Kokkos::Experimental
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
//md_parallel_for - deprecated use parallel_for
// ------------------------------------------------------------------ //

namespace Kokkos { namespace Experimental {

template <typename MDRange, typename Functor, typename Enable = void>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

// Cuda specialization
#if defined( __CUDACC__ ) && defined( KOKKOS_ENABLE_CUDA )
template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f);
  closure.execute();
}

template <typename MDRange, typename Functor>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f);
  closure.execute();
}
#endif
// ------------------------------------------------------------------ //

// ------------------------------------------------------------------ //
//md_parallel_reduce - deprecated use parallel_reduce
// ------------------------------------------------------------------ //
template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, ValueType> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;
  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_ENABLE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Kokkos::Impl::Experimental::MDFunctor<MDRange, Functor, ValueType> g(range, f);

  using range_policy = typename MDRange::impl_range_policy;

  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

// Cuda - md_parallel_reduce not implemented - use parallel_reduce

} } // namespace Kokkos::Experimental

#endif //KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

