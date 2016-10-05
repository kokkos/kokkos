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

#include <initializer_list> //unnecessary now?

#if defined(KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION) && defined(KOKKOS_HAVE_PRAGMA_IVDEP) && !defined(__CUDA_ARCH__)
#define KOKKOS_MDRANGE_IVDEP
#endif

#include<impl/KokkosExp_Host_IterateTile.hpp>
#include <Kokkos_ExecPolicy.hpp>
#include <Kokkos_Parallel.hpp>

#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )
#include<Cuda/KokkosExp_Cuda_IterateTile.hpp>
#endif

namespace Kokkos { namespace Experimental {

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
  static constexpr Iterate value = Iterate::Right;
};

template <typename ExecSpace>
struct default_inner_direction
{
  using type = Iterate;
  static constexpr Iterate value = Iterate::Right;
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
  static_assert( N < 4u, "Kokkos Error: Unsupported rank...");

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

  static_assert( !std::is_same<typename traits::iteration_pattern,void>::value
               , "Kokkos Error: MD iteration pattern not defined" );

  using iteration_pattern   = typename traits::iteration_pattern;
  using work_tag            = typename traits::work_tag;

  static constexpr int rank = iteration_pattern::rank;

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

  using index_type  = typename traits::index_type;
  using point_type  = Kokkos::Array<index_type,rank>;
  using tile_type   = Kokkos::Array<int,rank>;


  MDRangePolicy( point_type const& lower, point_type const& upper, tile_type const& tile = tile_type{} )
    : m_lower(lower)
    , m_upper(upper)
    , m_tile(tile)
    , m_num_tiles{1}
  {
    index_type end;
    for (int i=0; i<rank; ++i) {
      end = upper[i] - lower[i];
      m_tile_end[i] = static_cast<index_type>((end + m_tile[i] -1) / m_tile[i]);
      m_num_tiles *= m_tile_end[i];
    }
  }

  point_type m_lower;
  point_type m_upper;
  tile_type  m_tile;
  point_type m_tile_end;
  index_type m_num_tiles;
};

#if 0
namespace Impl {

template < typename RP
         , typename Functor
         , typename Dim 
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile 
  : protected HostIterateTile<RP, Functor, std::integral_constant<int,Dim::value-1>, Tag, ValueType>
{
  static_assert( Dim::value <= RP::rank, "Error: greater than rank");
  static_assert( Dim::value > 0, "Error: less than 0");

  using base_type = HostIterateTile<RP, Functor, std::integral_constant<int,Dim::value-1>, Tag, ValueType>;
  using final_type = HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, ValueType>;

  using index_type = typename RP::index_type;

  template< typename... Args >
  KOKKOS_INLINE_FUNCTION
  HostIterateTile( Args &&... args )
    : base_type( std::forward<Args>(args)... )
  {}

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = final_type::m_begin[Dim::value]; i < final_type::m_end[Dim::value]; ++i) {
      base_type::apply_left(i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = final_type::m_begin[RP::rank-Dim::value-1]; i < final_type::m_end[RP::rank-Dim::value-1]; ++i) {
      base_type::apply_right(std::forward<Args>(args)...,i);
    }
  }

};

template < typename RP
         , typename Functor
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, void, void>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx )
    : m_rp{rp}
    , m_func{func}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(std::forward<Args>(args)...,i);
    }
  }

  RP        const& m_rp;
  Functor   const& m_func;
  point_type m_begin;
  point_type m_end;
};


template < typename RP
         , typename Functor
         , typename Tag
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, void>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx )
    : m_rp{rp}
    , m_func{func}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(m_tag, i, std::forward<Args>(args)...);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(m_tag, std::forward<Args>(args)...,i);
    }
  }

  RP         const& m_rp;
  Functor    const& m_func;
  point_type m_begin;
  point_type m_end;
  Tag        m_tag;
};

template < typename RP
         , typename Functor
         , typename ValueType
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, void, ValueType>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx, value_type & v )
    : m_rp{rp}
    , m_func{func}
    , m_v{v}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] =  (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(i, std::forward<Args>(args)..., m_v);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(std::forward<Args>(args)...,i, m_v);
    }
  }

  RP        const& m_rp;
  Functor   const& m_func;
  value_type     & m_v;
  point_type m_begin;
  point_type m_end;
};

template < typename RP
         , typename Functor
         , typename Tag
         , typename ValueType
         >
struct HostIterateTile<RP, Functor, std::integral_constant<int,0>, Tag, ValueType>
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  using value_type = ValueType;

  KOKKOS_INLINE_FUNCTION
  HostIterateTile( RP const& rp, Functor const& func, index_type tile_idx, value_type & v )
    : m_rp{rp}
    , m_func{func}
    , m_v{v}
  {
    if (RP::outer_direction == RP::Left) {
      for (int i=0; i<RP::rank; ++i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
    else {
      for (int i=RP::rank-1; i>=0; --i) {
        m_begin[i] = (tile_idx % rp.m_tile_end[i]) * rp.m_tile[i] + rp.m_lower[i] ;
        m_end[i] = ((m_begin[i] + rp.m_tile[i]) <= rp.m_upper[i]) ? (m_begin[i] + rp.m_tile[i]) : rp.m_upper[i] ;
        tile_idx /= rp.m_tile_end[i];
      }
    }
  }

  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left();
    } else {
      apply_right();
    }
  }

protected:

  template< typename... Args>
  void apply_left( Args &&... args )
  {
    for (index_type i = m_begin[0]; i < m_end[0]; ++i) {
      m_func(m_tag, i, std::forward<Args>(args)..., m_v);
    }
  }

  template< typename... Args>
  void apply_right( Args &&... args )
  {
    for (index_type i = m_begin[RP::rank-1]; i < m_end[RP::rank-1]; ++i) {
      m_func(m_tag, std::forward<Args>(args)...,i, m_v);
    }
  }

  RP        const& m_rp;
  Functor   const& m_func;
  value_type     & m_v;
  point_type m_begin;
  point_type m_end;
  Tag        m_tag;
};


// ------------------------------------------------------------------ //
#if 0
// Cuda IterateTile
#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )

template < int N , typename RP >
struct BlockGridSizes;

template< typename RP >
struct BlockGridSizes<2 , RP>
{
  BlockGridSizes( RP const& _rp, dim3 & _blocksize, dim3 & _gridsize )
    : m_rp(_rp)
    , block(_blocksize)
    , grid(_gridsize)
  {}

  void apply()
  {
    // max blocks 65535 per blockdim or total?
    // when should tile dims be checked to be < 1024 and power of two (or add padding)?
    if (RP::outer_direction == RP::Left) {
      blocksize( m_rp.m_tile[0] , m_rp.m_tile[1] , 1); //pad for mult of 16? check within max num threads bounds? 
      gridsize( std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , 65535 ) , std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1 ) / block.y , 65535 ) , 1);
    }
    else
    {
      blocksize( m_rp.m_tile[1] , m_rp.m_tile[0] , 1); 
      gridsize( std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.x - 1 ) / block.x , 65535 ) , std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.y - 1 ) / block.y , 65535 ) , 1);
    }
  }

  dim3 &block;
  dim3 &grid;
  RP const& m_rp;
};

// ------------------------------------------------------------------ //

template < int N , typename RP , typename Tag >
struct apply_left;
template < int N , typename RP , typename Tag >
struct apply_right;

// Specializations for void tag type

// Rank2
template< typename RP >
struct apply_left<2,RP,void >
{
  using index_type = typename RP::index_type;

  __device__
  apply_left( RP const& _rp )
  : m_rp(_rp)
  {}

  inline __device__
  void exec_range()
  {
    // Loop over size maxnumblocks until full range covered
    for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
      //Execute kernel with extracted tile_id0
      const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
      if ( offset_1 < m_rp.m_upper[1] ) {
        for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
          const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
          if ( offset_0 < m_rp.m_upper[0] ) {
            m_func(offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  } //end exec_range

private:
  RP const& m_rp;
};

template< typename RP >
struct apply_right<2,RP,void>
{
  using index_type = typename RP::index_type;

  __device__ 
  apply_right( RP const& _rp )
  : m_rp(_rp)
  {}

  inline __device__
  void exec_range()
  {
    for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
      const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
      if ( offset_0 < m_rp.m_upper[0] ) {
        for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
          //Execute kernel with extracted tile_id0
          const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
          if ( offset_1 < m_rp.m_upper[1] ) {
            m_func(offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }

private:
  RP const& m_rp;
};

template< typename RP , typename Tag >
struct apply_left<2,RP,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_left( RP const& _rp )
  : m_rp(_rp)
  {}

  inline __device__
  void exec_range()
  {
    // Loop over size maxnumblocks until full range covered
    for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
      //Execute kernel with extracted tile_id0
      const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
      if ( offset_1 < m_rp.m_upper[1] ) {
        for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
          const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
          if ( offset_0 < m_rp.m_upper[0] ) {
            m_func(Tag(), offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }

private:
  RP const& m_rp;
};

template< typename RP , typename Tag >
struct apply_right<2,RP,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_right( RP const& _rp )
  : m_rp(_rp)
  {}

  inline __device__
  void exec_range()
  {
    for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
      const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
      if ( offset_0 < m_rp.m_upper[0] ) {
        for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
          //Execute kernel with extracted tile_id0
          const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
          if ( offset_1 < m_rp.m_upper[1] ) {
            m_func(Tag(), offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }

private:
  RP const& m_rp;
};

// ----------------------------------------------------------------------------------

template < typename RP
         , typename Functor
         , typename Tag
//         , typename std::enable_if< std::is_same<Tag,void>::value >::type*
         >
struct DeviceIterateTile
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  struct VoidDummy {};
  typedef typename std::conditional< std::is_same<Tag, void>::value, VoidDummy, Tag>::type usable_tag;

  DeviceIterateTile( RP const& rp, Functor const& func )
    : m_rp{rp}
    , m_func{func}
  {}

private:
  inline __device__  
  void apply() const
  {
    if (RP::inner_direction == RP::Left) {
      apply_left<RP::rank,RP,Tag>(m_rp).exec_range();
    } else {
      apply_right<RP::rank,RP,Tag>(m_rp).exec_range();
    }
  }

public:

  inline
  __device__
  void operator()(void) const
  {
    this-> apply();
  }

  inline
  void execute() const
  {
    dim3 block, grid;
    BlockGridSizes<RP::rank,RP>(m_rp,block,grid).apply();

    CudaParallelLaunch< DeviceIterateTile >( *this , grid , block , 0 );
  }

protected:

  RP        const& m_rp;
  Functor   const& m_func;
};

template < typename RP
         , typename Functor
         , typename Tag
         , typename std::enable_if< !std::is_same<Tag,void>::value >::type*
         >
struct DeviceIterateTile
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

//FIX: Tiles are determined by GPU blockIds
  DeviceIterateTile( RP const& rp, Functor const& func )
    : m_rp{rp}
    , m_func{func}
  {}

public:

  inline
  __device__
  void operator()(void) const
  {
    this-> apply();
  }

  inline
  void execute() const
  {
    dim3 block, grid;
    BlockGridSizes<RP::rank>(m_rp,block,grid);
//    get_block_thread_sizes<RP::rank>(block,grid);

    CudaParallelLaunch< DeviceIterateTile >( *this , grid , block , 0 );
  }


private:
  KOKKOS_INLINE_FUNCTION
  void apply()
  {
    if (RP::inner_direction == RP::Left) {
      apply_left<RP::rank>(m_rp).exec_range();
    } else {
      apply_right<RP::rank>(m_rp).exec_range();;
    }
  }

protected:

  RP        const& m_rp;
  Functor   const& m_func;
};
#endif

#endif // KOKKOS_HAVE_CUDA

// MDFunctor - wraps the range_policy and functor to pass to IterateTile
// Serial, Threads, OpenMP
// Cuda uses DeviceIterateTile directly within md_parallel_for
template < typename MDRange, typename Functor, typename ValueType = void >
struct MDFunctor
{
  using range_policy = MDRange;
  using functor_type = Functor;
  using value_type   = ValueType;
  using work_tag     = typename range_policy::work_tag;
  using index_type   = typename range_policy::index_type;
  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRange
                                                                           , Functor
                                                                           , std::integral_constant<int,MDRange::rank - 1>
                                                                           , work_tag
                                                                           , value_type  
                                                                           >;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDRange const& range, Functor const& f, ValueType & v )
    : m_range( range )
    , m_func( f )
    , m_v( v )
  {}


  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor && ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor && ) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(index_type t, value_type & v) const
  {
    iterate_type(m_range, m_func, t, v).apply();
  }
  
  MDRange   m_range;
  Functor   m_func;
  ValueType m_v;
};

template < typename MDRange, typename Functor >
struct MDFunctor< MDRange, Functor, void >
{
  using range_policy = MDRange;
  using functor_type = Functor;
  using work_tag     = typename range_policy::work_tag;
  using index_type   = typename range_policy::index_type;
  using iterate_type = typename Kokkos::Experimental::Impl::HostIterateTile< MDRange
                                                                           , Functor
                                                                           , std::integral_constant<int,MDRange::rank - 1>
                                                                           , work_tag
                                                                           , void  
                                                                           >;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDRange const& range, Functor const& f )
    : m_range( range )
    , m_func( f )
  {}


  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor const& ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor( MDFunctor && ) = default;

  KOKKOS_INLINE_FUNCTION
  MDFunctor& operator=( MDFunctor && ) = default;

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(index_type t) const
  {
    iterate_type(m_range, m_func, t).apply();
  }
  
  MDRange m_range;
  Functor m_func;
};

} // namespace Impl
#endif

//md_parallel_for
template <typename MDRange, typename Functor, typename Enable = void>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && !std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::MDFunctor<MDRange, Functor, void> g(range, f);

  using range_policy = typename MDRange::range_policy;

  Kokkos::parallel_for( range_policy(0, range.m_num_tiles).set_chunk_size(1), g, str );
}

// Cuda specialization
#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )
template <typename MDRange, typename Functor>
void md_parallel_for( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f); 
  closure.execute();
}

template <typename MDRange, typename Functor>
void md_parallel_for( MDRange const& range
                    , Functor const& f
                    , const std::string& str = ""
                    , typename std::enable_if<( true
                      #if defined( KOKKOS_HAVE_CUDA)
                      && std::is_same< typename MDRange::range_policy::execution_space, Kokkos::Cuda>::value
                      #endif
                      ) >::type* = 0
                    )
{
  Impl::DeviceIterateTile<MDRange, Functor, typename MDRange::work_tag> closure(range, f); 
  closure.execute();
}
#endif

//md_parallel_reduce
template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    , const std::string& str = ""
                    )
{
  Impl::MDFunctor<MDRange, Functor, ValueType> g(range, f, v);

  using range_policy = typename MDRange::range_policy;

  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

template <typename MDRange, typename Functor, typename ValueType>
void md_parallel_reduce( const std::string& str
                    , MDRange const& range
                    , Functor const& f
                    , ValueType & v
                    )
{
  Impl::MDFunctor<MDRange, Functor, ValueType> g(range, f, v);

  using range_policy = typename MDRange::range_policy;

  Kokkos::parallel_reduce( str, range_policy(0, range.m_num_tiles).set_chunk_size(1), g, v );
}

}} // namespace Kokkos::Experimental

#endif //KOKKOS_CORE_EXP_MD_RANGE_POLICY_HPP

