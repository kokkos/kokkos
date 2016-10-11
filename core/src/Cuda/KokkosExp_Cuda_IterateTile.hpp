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

// Cannot include the file below, leads to following type of errors:
// /home/ndellin/kokkos/core/src/Cuda/Kokkos_CudaExec.hpp(84): error: incomplete type is not allowed
//#include<Cuda/Kokkos_CudaExec.hpp>

#if (KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <typeinfo>
#endif

namespace Kokkos { namespace Experimental { namespace Impl {

// ------------------------------------------------------------------ //

template< class DriverType >
__global__
static void cuda_parallel_launch( const DriverType driver )
{
  driver();
}

template< class DriverType >
struct CudaLaunch
{
  inline
  CudaLaunch( const DriverType & driver
                    , const dim3       & grid
                    , const dim3       & block
            )
  {
    cuda_parallel_launch< DriverType ><<< grid , block >>>(driver);
  }

};

// ------------------------------------------------------------------ //
template< int N , typename RP , typename Functor , typename Tag >
struct apply_impl;

//Rank 2
// Specializations for void tag type
template< typename RP , typename Functor >
struct apply_impl<2,RP,Functor,void >
{
  using index_type = typename RP::index_type;

  __device__
  apply_impl( const RP & _rp , const Functor & _f )
  : m_rp(_rp)
  , m_func(_f)
  {}

  inline __device__
  void exec_range() const
  {
// LL
  if (RP::inner_direction == RP::Left) {
 /*
    index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
    index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

    for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
    for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
            m_func(i, j);
    } }
*/
    for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
      const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
      if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
        for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
          const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
          if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
            m_func(offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  } 
// LR
  else {
/*
    index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
    index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

    for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
    for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
            m_func(i, j);
    } }
*/
    for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
      const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
      if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
        for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
          const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
          if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
            m_func(offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }

  } //end exec_range

private:
  const RP & m_rp;
  const Functor & m_func;

};

// Tag specialization
template< typename RP , typename Functor , typename Tag >
struct apply_impl<2,RP,Functor,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_impl( const RP & _rp , const Functor & _f )
  : m_rp(_rp)
  , m_func(_f)
  {}

  inline __device__
  void exec_range() const
  {
  if (RP::inner_direction == RP::Left) {
    // Loop over size maxnumblocks until full range covered
/*
    index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
    index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

    for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
    for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
            m_func(Tag(), i, j);
    } }
*/
    for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
      const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
      if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
        for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
          const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
          if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
            m_func(Tag(), offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }
  else {
/*
    index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
    index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

    for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
    for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
            m_func(Tag(), i, j);
    } }
*/
    for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
      const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
      if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
        for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
          const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
          if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
            m_func(Tag(), offset_0 , offset_1);
          }
        } //end inner for
      } //end outer if
    } //end outer for
  }

  } //end exec_range

private:
  const RP & m_rp;
  const Functor & m_func;
};


//Rank 3
// Specializations for void tag type
template< typename RP , typename Functor >
struct apply_impl<3,RP,Functor,void >
{
  using index_type = typename RP::index_type;

  __device__
  apply_impl( const RP & _rp , const Functor & _f )
  : m_rp(_rp)
  , m_func(_f)
  {}

  inline __device__
  void exec_range() const
  {
// LL
    if (RP::inner_direction == RP::Left) {
      /*
         index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
         index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

         for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
         for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
         m_func(i, j);
         } }
         */
      for ( index_type tile_id2 = blockIdx.z; tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z ) { 
        const index_type offset_2 = tile_id2*m_rp.m_tile[2] + threadIdx.z;
        if ( offset_2 < m_rp.m_upper[2] && threadIdx.z < m_rp.m_tile[2] ) {
          for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
            const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
            if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
              for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
                const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
                if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
                  m_func(offset_0 , offset_1 , offset_2);
                }
              } //end inner for
            } //end outer if
          } //end outer for
        } //end outer if
      } //end outer for
    } 
// LR
  else {
    /*
       index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
       index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

       for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
       for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
       m_func(i, j);
       } }
       */
    for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
      const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
      if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
        for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
          const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
          if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {

            for ( index_type tile_id2 = blockIdx.z; tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z ) { 
              const index_type offset_2 = tile_id2*m_rp.m_tile[2] + threadIdx.z;
              if ( offset_2 < m_rp.m_upper[2] && threadIdx.z < m_rp.m_tile[2] ) {
                m_func(offset_0 , offset_1 , offset_2);
              }
            } //end inner for
          } //end outer if
        } //end inner for
      } //end outer if
    } //end outer for
  }

  } //end exec_range

private:
  const RP & m_rp;
  const Functor & m_func;

};

// Tag specialization
template< typename RP , typename Functor , typename Tag >
struct apply_impl<3,RP,Functor,Tag>
{
  using index_type = typename RP::index_type;

  inline __device__
  apply_impl( const RP & _rp , const Functor & _f )
  : m_rp(_rp)
  , m_func(_f)
  {}

  inline __device__
  void exec_range() const
  {
    if (RP::inner_direction == RP::Left) {
      // Loop over size maxnumblocks until full range covered
      /*
         index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
         index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

         for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
         for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
         m_func(Tag(), i, j);
         } }
         */
      for ( index_type tile_id2 = blockIdx.z; tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z ) { 
        const index_type offset_2 = tile_id2*m_rp.m_tile[2] + threadIdx.z;
        if ( offset_2 < m_rp.m_upper[2] && threadIdx.z < m_rp.m_tile[2] ) {
          for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
            const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
            if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
              for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
                const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
                if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
                  m_func(Tag(), offset_0 , offset_1 , offset_2);
                }
              } //end inner for
            } //end inner for
          } //end outer if
        } //end outer if
      } //end outer for
    }
    else {
      /*
         index_type offset_1 = blockIdx.y*m_rp.m_tile[1] + threadIdx.y;
         index_type offset_0 = blockIdx.x*m_rp.m_tile[0] + threadIdx.x;

         for ( index_type i = offset_0; i < m_rp.m_upper[0], threadIdx.x < m_rp.m_tile[0]; i += (gridDim.x*m_rp.m_tile[0]) ) {
         for ( index_type j = offset_1; j < m_rp.m_upper[1], threadIdx.y < m_rp.m_tile[1]; j += (gridDim.y*m_rp.m_tile[1]) ) {
         m_func(Tag(), i, j);
         } }
         */
      for ( index_type tile_id0 = blockIdx.x; tile_id0 < m_rp.m_tile_end[0]; tile_id0 += gridDim.x ) { 
        const index_type offset_0 = tile_id0*m_rp.m_tile[0] + threadIdx.x;
        if ( offset_0 < m_rp.m_upper[0] && threadIdx.x < m_rp.m_tile[0] ) {
          for ( index_type tile_id1 = blockIdx.y; tile_id1 < m_rp.m_tile_end[1]; tile_id1 += gridDim.y ) { 
            const index_type offset_1 = tile_id1*m_rp.m_tile[1] + threadIdx.y;
            if ( offset_1 < m_rp.m_upper[1] && threadIdx.y < m_rp.m_tile[1] ) {
              for ( index_type tile_id2 = blockIdx.z; tile_id2 < m_rp.m_tile_end[2]; tile_id2 += gridDim.z ) { 
                const index_type offset_2 = tile_id2*m_rp.m_tile[2] + threadIdx.z;
                if ( offset_2 < m_rp.m_upper[2] && threadIdx.z < m_rp.m_tile[2] ) {
                  m_func(Tag(), offset_0 , offset_1 , offset_2);
                }
              }
            }
          } //end inner for
        } //end outer if
      } //end outer for
    }

  } //end exec_range

private:
  const RP & m_rp;
  const Functor & m_func;
};


// ----------------------------------------------------------------------------------

template < typename RP
         , typename Functor
         , typename Tag
         >
struct DeviceIterateTile
{
  using index_type = typename RP::index_type;
  using point_type = typename RP::point_type;

  struct VoidDummy {};
  typedef typename std::conditional< std::is_same<Tag, void>::value, VoidDummy, Tag>::type usable_tag;

  DeviceIterateTile( const RP & rp, const Functor & func )
    : m_rp{rp}
    , m_func{func}
  {}

private:
  inline __device__  
  void apply() const
  {
    apply_impl<RP::rank,RP,Functor,Tag>(m_rp,m_func).exec_range();
  } //end apply

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
    const unsigned int maxblocks = 65535;
    if ( RP::rank == 2 )
    {
      const dim3 block( m_rp.m_tile[0] , m_rp.m_tile[1] , 1); //pad for mult of 16? check within max num threads bounds? 
      const dim3 grid( std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , maxblocks ) , std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1 ) / block.y , maxblocks ) , 1);
      CudaLaunch< DeviceIterateTile >( *this , grid , block );

    }
    else if ( RP::rank == 3 )
    {
      const dim3 block( m_rp.m_tile[0] , m_rp.m_tile[1] , m_rp.m_tile[2] ); //pad for mult of 16? check within max num threads bounds? 
      const dim3 grid( 
          std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , maxblocks ) 
        , std::min( ( m_rp.m_upper[1] - m_rp.m_lower[1] + block.y - 1 ) / block.y , maxblocks ) 
        , std::min( ( m_rp.m_upper[2] - m_rp.m_lower[2] + block.z - 1 ) / block.z , maxblocks ) 
        );
      CudaLaunch< DeviceIterateTile >( *this , grid , block );

    }
    else if ( RP::rank == 4 )
    {
      // id0,id1 encoded within threadIdx.x; id2 to threadIdx.y; id3 to threadIdx.z
      const dim3 block( m_rp.m_tile[0]*m_rp.m_tile[1] , m_rp.m_tile[2] , m_rp.m_tile[3] ); //pad for mult of 16? check within max num threads bounds? 
      const dim3 grid( 
          std::min( ( ( m_rp.m_upper[0] - m_rp.m_lower[0] + m_rp.m_tile[0] - 1 ) / m_rp.m_tile[0] 
                    *  ( m_rp.m_upper[1] - m_rp.m_lower[1] + m_rp.m_tile[1] - 1 ) / m_rp.m_tile[1] )
                  , maxblocks ) 
          //std::min( ( m_rp.m_upper[0] - m_rp.m_lower[0] + block.x - 1 ) / block.x , maxblocks ) 
        , std::min( ( m_rp.m_upper[2] - m_rp.m_lower[2] + block.y - 1 ) / block.y , maxblocks ) 
        , std::min( ( m_rp.m_upper[3] - m_rp.m_lower[3] + block.z - 1 ) / block.z , maxblocks ) 
        );
      CudaLaunch< DeviceIterateTile >( *this , grid , block );

    }
    else
    {
      printf(" Exceeded rank bounds with Cuda\n");
    }
//    CudaParallelLaunch< DeviceIterateTile >( *this , grid , block , 0 );

  } //end execute

protected:
  const RP         m_rp;
  const Functor    m_func;
};

} } } //end namespace Kokkos::Experimental::Impl

#endif
#endif
