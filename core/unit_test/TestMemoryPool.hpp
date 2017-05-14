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


#ifndef KOKKOS_UNITTEST_MEMPOOL_HPP
#define KOKKOS_UNITTEST_MEMPOOL_HPP

#include <cstdio>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <impl/Kokkos_Timer.hpp>

namespace TestMemoryPool {

template< class DeviceType >
struct TestMemoryPool_Functor {

  typedef Kokkos::View< uintptr_t * , DeviceType >         ptrs_type ;
  typedef Kokkos::MemoryPool< DeviceType > pool_type ;

  pool_type pool ;
  ptrs_type ptrs ;

  TestMemoryPool_Functor( const pool_type & arg_pool , size_t n )
    : pool( arg_pool )
    , ptrs( "ptrs" , n )
    {}

  // Specify reduction argument value_type to avoid
  // confusion with tag-dispatch.

  using value_type = long ;

  struct TagAlloc {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagAlloc , int i , long & update ) const noexcept
    {
      unsigned alloc_size = 32 * ( 1 + ( i % 5 ));
      ptrs(i) = (uintptr_t)  pool.allocate( alloc_size );
      if ( ptrs(i) ) { ++update ; }
    }

  struct TagDealloc {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagDealloc , int i , long & update ) const noexcept
    {
      if ( ptrs(i) && ( 0 == i % 3 ) ) {
        unsigned alloc_size = 32 * ( 1 + ( i % 5 ));
        pool.deallocate( (void*) ptrs(i) , alloc_size );
        ptrs(i) = 0 ;
        ++update ;
      }
    }

  struct TagRealloc {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagRealloc , int i , long & update ) const noexcept
    {
      if ( 0 == ptrs(i) ) {
        unsigned alloc_size = 32 * ( 1 + ( i % 5 ));
        ptrs(i) = (uintptr_t)  pool.allocate( alloc_size );
        if ( ptrs(i) ) { ++update ; }
      }
    }

  struct TagMixItUp {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagMixItUp , int i , long & update ) const noexcept
    {
      if ( ptrs(i) && ( 0 == i % 3 ) ) {

        unsigned alloc_size = 32 * ( 1 + ( i % 5 ));

        pool.deallocate( (void*) ptrs(i) , alloc_size );

        ptrs(i) = (uintptr_t)  pool.allocate( alloc_size );

        if ( ptrs(i) ) { ++update ; }
      }
    }
};

template< class PoolType >
void print_memory_pool_stats
  ( typename PoolType::usage_statistics const & stats )
{
  std::cout << "MemoryPool {" << std::endl
            << "  bytes capacity = " << stats.capacity_bytes << std::endl
            << "  bytes used     = " << stats.consumed_bytes << std::endl
            << "  bytes reserved = " << stats.reserved_bytes << std::endl
            << "  bytes free     = " << ( stats.capacity_bytes -
               ( stats.consumed_bytes + stats.reserved_bytes ) ) << std::endl
            << "  alloc used     = " << stats.consumed_blocks << std::endl
            << "  alloc reserved = " << stats.reserved_blocks << std::endl
            << "  super used     = " << stats.consumed_superblocks << std::endl
            << "  super reserved = " << ( stats.capacity_superblocks -
                                    stats.consumed_superblocks ) << std::endl
            << "}" << std::endl ;
}

template< class DeviceType >
void test_memory_pool_v2( const bool print_statistics
                        , const bool print_superblocks )
{
  typedef typename DeviceType::memory_space     memory_space ;
  typedef typename DeviceType::execution_space  execution_space ;
  typedef Kokkos::MemoryPool< DeviceType > pool_type ;
  typedef TestMemoryPool_Functor< DeviceType > functor_type ;

  typedef typename functor_type::TagAlloc   TagAlloc ;
  typedef typename functor_type::TagDealloc TagDealloc ;
  typedef typename functor_type::TagRealloc TagRealloc ;
  typedef typename functor_type::TagMixItUp TagMixItUp ;

  const size_t    total_alloc_size = 10000000 ;
  const unsigned  min_block_size   = 64 ;
  const unsigned  max_block_size   = 256 ;
  const long      nfill            = 70000 ;

  for ( uint32_t k = 0 , min_superblock_size = 10000 ;
        k < 3 ; ++k , min_superblock_size *= 10 ) {

    typename pool_type::usage_statistics stats ;

    pool_type pool( memory_space()
                  , total_alloc_size
                  , min_block_size
                  , max_block_size
                  , min_superblock_size );

    functor_type functor(pool,nfill);

    long result = 0 ;
    long ndel  = 0 ;

    Kokkos::parallel_reduce
      ( Kokkos::RangePolicy< execution_space , TagAlloc >(0,nfill)
      , functor
      , result
      );

    pool.get_usage_statistics( stats );

    const int fill_error = ( nfill != result ) ||
                           ( nfill != long(stats.consumed_blocks) );

    if ( fill_error || print_statistics ) print_memory_pool_stats< pool_type >( stats );
    if ( fill_error || print_superblocks ) pool.print_state( std::cout );

    ASSERT_EQ( nfill , result );
    ASSERT_EQ( nfill , long(stats.consumed_blocks) );

    Kokkos::parallel_reduce
      ( Kokkos::RangePolicy< execution_space , TagDealloc >(0,nfill)
      , functor
      , ndel
      );

    pool.get_usage_statistics( stats );

    const int del_error = ( nfill - ndel ) != long(stats.consumed_blocks);

    if ( del_error || print_statistics ) print_memory_pool_stats< pool_type >( stats );
    if ( del_error || print_superblocks ) pool.print_state( std::cout );

    ASSERT_EQ( ( nfill - ndel ) , long(stats.consumed_blocks) );

    Kokkos::parallel_reduce
      ( Kokkos::RangePolicy< execution_space , TagRealloc >(0,nfill)
      , functor
      , result
      );

    pool.get_usage_statistics( stats );

    const int refill_error = ( ndel != result ) ||
                             ( nfill != long(stats.consumed_blocks) );

    if ( refill_error || print_statistics ) print_memory_pool_stats< pool_type >( stats );
    if ( refill_error || print_superblocks ) pool.print_state( std::cout );

    ASSERT_EQ( ndel , result );
    ASSERT_EQ( nfill , long(stats.consumed_blocks) );

    Kokkos::parallel_reduce
      ( Kokkos::RangePolicy< execution_space , TagMixItUp >(0,nfill)
      , functor
      , result
      );

    pool.get_usage_statistics( stats );

    const int mix_error = ( ndel != result ) ||
                          ( nfill != long(stats.consumed_blocks) );

    if ( mix_error || print_statistics ) print_memory_pool_stats< pool_type >( stats );
    if ( mix_error || print_superblocks ) pool.print_state( std::cout );

    ASSERT_EQ( ndel , result );
    ASSERT_EQ( nfill , long(stats.consumed_blocks) );
  }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

} // namespace TestMemoryPool {

#undef TESTMEMORYPOOL_PRINT
#undef TESTMEMORYPOOL_PRINT_STATUS
#undef STRIDE
#undef STRIDE_ALLOC

namespace Test {

TEST_F( TEST_CATEGORY, memory_pool )
{
  TestMemoryPool::test_memory_pool_v2< TEST_EXECSPACE >(false,false);
}
}
#endif
