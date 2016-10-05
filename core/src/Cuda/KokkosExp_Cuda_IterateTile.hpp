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

#ifndef KOKKOS_CUDA_EXP_ITERATE_TILE_HPP
#define KOKKOS_CUDA_EXP_ITERATE_TILE_HPP

#include <iostream>
#include <algorithm>
#include <stdio.h>

#include <Kokkos_Macros.hpp>

/* only compile this file if CUDA is enabled for Kokkos */
#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )

#include <utility>

#if (KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>
#endif

namespace Kokkos { namespace Experimental { namespace Impl {


template< class DriverType >
__global__
static void cuda_parallel_launch( const DriverType driver )
{
  driver();
}

// ------------------------------------------------------------------ //
// Cuda IterateTile
#if defined( __CUDACC__ ) && defined( KOKKOS_HAVE_CUDA )

// No longer use this... why compiler errors?
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
      block( m_rp.m_tile[0] , m_rp.m_tile[1] , 1); //pad for mult of 16? check within max num threads bounds? 
      grid( std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , 65535 ) , std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1 ) / block.y , 65535 ) , 1);
    }
    else
    {
      block( m_rp.m_tile[1] , m_rp.m_tile[0] , 1); 
      grid( std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.x - 1 ) / block.x , 65535 ) , std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.y - 1 ) / block.y , 65535 ) , 1);
    }
  }

  dim3 &block;
  dim3 &grid;
  RP const& m_rp;
};

// ------------------------------------------------------------------ //

template < int N , typename RP , typename Functor , typename Tag >
struct apply_left;
template < int N , typename RP , typename Functor , typename Tag >
struct apply_right;

// Specializations for void tag type

// Rank2
template< typename RP , typename Functor >
struct apply_left<2,RP,Functor,void >
{
  using index_type = typename RP::index_type;

  __device__
  apply_left( RP const& _rp , Functor const& _f )
  : m_rp(_rp)
  , m_func(_f)
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
  Functor const& m_func;
};

template< typename RP , typename Functor >
struct apply_right<2,RP,Functor,void>
{
  using index_type = typename RP::index_type;

  __device__ 
  apply_right( RP const& _rp , Functor const& _f)
  : m_rp(_rp)
  , m_func(_f)
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
  Functor const& m_func;
};

template< typename RP , typename Functor , typename Tag >
struct apply_left<2,RP,Functor,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_left( RP const& _rp , Functor const& _f )
  : m_rp(_rp)
  , m_func(_f)
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
  Functor const& m_func;
};

template< typename RP , typename Functor , typename Tag >
struct apply_right<2,RP,Functor,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_right( RP const& _rp , Functor const& _f )
  : m_rp(_rp)
  , m_func(_f)
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
  Functor const& m_func;
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
      apply_left<RP::rank,RP,Functor,Tag>(m_rp,m_func).exec_range();
    } else {
      apply_right<RP::rank,RP,Functor,Tag>(m_rp,m_func).exec_range();
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
    //dim3 block, grid;
    const unsigned int maxblocks = 65535;
    if ( RP::rank == 2 )
    {
      if (RP::outer_direction == RP::Left) {
        const dim3 block( m_rp.m_tile[0] , m_rp.m_tile[1] , 1); //pad for mult of 16? check within max num threads bounds? 
        const dim3 grid( std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , maxblocks ) , std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1 ) / block.y , maxblocks ) , 1);
        cuda_parallel_launch<<<grid,block>>>(*this);
      }
      else
      {
        const dim3 block( m_rp.m_tile[1] , m_rp.m_tile[0] , 1); 
        const dim3 grid( std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.x - 1 ) / block.x , maxblocks ) , std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.y - 1 ) / block.y , maxblocks ) , 1);
        cuda_parallel_launch<<<grid,block>>>(*this);
      }
    }
    else
    {
      printf(" Exceeded rank bounds with Cuda\n");
    }
//    BlockGridSizes<RP::rank,RP>(m_rp,block,grid).apply();

//    cuda_parallel_launch<<<grid,block>>>(*this);
//    CudaParallelLaunch< DeviceIterateTile >( *this , grid , block , 0 );
  }


protected:
/*
  template < int N >
  struct apply_left;
  template < int N >
  struct apply_right;

// Specializations for void tag type

// Rank2
  template<>
  struct apply_left<2>
  {
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
      }
  };

  template<>
  struct apply_right<2>
  {
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
  };
*/

  RP        const& m_rp;
  Functor   const& m_func;
};

#if 0
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

private:
/*
  template < int N >
  void get_block_thread_sizes( dim3 &blocksize, dim3 &gridsize );

  inline
  void get_block_thread_sizes<2>( dim3 &blocksize, dim3 &gridsize )
  {
    // max blocks 65535 per blockdim or total?
    // when should tile dims be checked to be < 1024 and power of two (or add padding)?
    const dim3 block( rp.m_tile[0] , rp.m_tile[1] , 1); //pad for mult of 16? check within max num threads bounds? 
    const dim3 grid( std::min( ( rp.m_upper[0] - rp.m_lower[0] + block.x - 1 ) / block.x , cuda_internal_maximum_grid_count() ) , std::min( ( rp.m_upper[1] - rp.m_lower[1] + block.y - 1 ) / block.y , cuda_internal_maximum_grid_count() ) , 1);
  }
*/
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
/*
// Specializations for tag type
  template<int N>
  struct apply_left;
  template<int N>
  struct apply_right;

// Rank2
  template<>
  struct apply_left<2>
  {
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
  };

  template<>
  struct apply_right<2>
  {
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
  };
*/

  RP        const& m_rp;
  Functor   const& m_func;
};
#endif

#endif // KOKKOS_HAVE_CUDA

} } } //end namespace Kokkos::Experimental::Impl


#endif

#endif
