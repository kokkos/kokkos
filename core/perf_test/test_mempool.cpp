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

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#define USE_MEMORY_POOL_V2

using ExecSpace   = Kokkos::DefaultExecutionSpace ;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space ;

using MemoryPool =
#if defined( USE_MEMORY_POOL_V2 )
Kokkos::Experimental::MemoryPoolv2< ExecSpace > ;
#else
Kokkos::Experimental::MemoryPool< ExecSpace > ;
#endif

struct TestFunctor {

  typedef Kokkos::View< uintptr_t * , ExecSpace >  ptrs_type ;

  enum : unsigned { chunk = 64 };

  MemoryPool  pool ;
  ptrs_type   ptrs ;
  unsigned    stride_chunk ;
  unsigned    fill_stride ;
  unsigned    range_iter ;
  unsigned    repeat ;

  TestFunctor( size_t    total_alloc_size
             , unsigned  min_superblock_size
             , unsigned  number_alloc
             , unsigned  arg_stride_alloc
             , unsigned  arg_stride_chunk
             , unsigned  arg_repeat )
    : pool()
    , ptrs()
    , stride_chunk(0)
    , fill_stride(0)
    , repeat(0)
    {
      MemorySpace m ;

#if defined( USE_MEMORY_POOL_V2 )

      const unsigned min_block_size = chunk ;
      const unsigned max_block_size = chunk * arg_stride_chunk ;
      pool = MemoryPool( m , total_alloc_size
                           , min_block_size
                           , max_block_size
                           , min_superblock_size );
#else

      const unsigned superblock_size_lg2 =
        Kokkos::Impl::
         integral_power_of_two_that_contains( min_superblock_size );

      pool = MemoryPool( m , total_alloc_size , superblock_size_lg2 );

#endif

      ptrs = ptrs_type( Kokkos::view_alloc( m , "ptrs") , number_alloc );
      fill_stride = arg_stride_alloc ;
      stride_chunk = arg_stride_chunk ;
      range_iter   = fill_stride * number_alloc ;
      repeat       = arg_repeat ;
    }

  //----------------------------------------

  typedef long value_type ;

  //----------------------------------------

  struct TagFill {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagFill , int i , value_type & update ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

        ptrs(j) = (uintptr_t) pool.allocate(size_alloc);

        if ( ptrs(j) ) ++update ;
      }
    }

  bool test_fill()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagFill > policy ;

      long result = 0 ;

      Kokkos::parallel_reduce( policy(0,range_iter), *this , result );

      return result == ptrs.extent(0);
    }

  //----------------------------------------

  struct TagDel {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagDel , int i ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

        pool.deallocate( (void*) ptrs(j) , size_alloc );
      }
    }

  void test_del()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagDel > policy ;

      Kokkos::parallel_for( policy(0,range_iter), *this );
    }

  //----------------------------------------

  struct TagAllocDealloc {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagAllocDealloc , int i , long & update ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        if ( 0 == j % 3 ) {

          for ( int k = 0 ; k < repeat ; ++k ) {

            const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

            pool.deallocate( (void*) ptrs(j) , size_alloc );
        
            ptrs(j) = (uintptr_t) pool.allocate(size_alloc);

            if ( 0 == ptrs(j) ) update++ ;
          }
        }
      }
    }

  bool test_alloc_dealloc()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagAllocDealloc > policy ;

      long error_count = 0 ;

      Kokkos::parallel_reduce( policy(0,range_iter), *this , error_count );

      return 0 == error_count ;
    }
};



int main( int argc , char* argv[] )
{
  static const char help_flag[] = "--help" ;
  static const char alloc_size_flag[]   = "--alloc_size=" ;
  static const char super_size_flag[]   = "--super_size=" ;
  static const char chunk_span_flag[]   = "--chunk_span=" ;
  static const char fill_stride_flag[]  = "--fill_stride=" ;
  static const char fill_level_flag[]   = "--fill_level=" ;
  static const char repeat_outer_flag[] = "--repeat_outer=" ;
  static const char repeat_inner_flag[] = "--repeat_inner=" ;

  long total_alloc_size    = 1000000 ;
  int  min_superblock_size =   10000 ;
  int  chunk_span          =       5 ;
  int  fill_stride        =       1 ;
  int  fill_level         =      70 ;
  int  repeat_outer   =       1 ;
  int  repeat_inner   =       1 ;

  int  ask_help = 0 ;

  for(int i=1;i<argc;i++)
  {
     const char * const a = argv[i];

     if ( ! strncmp(a,help_flag,strlen(help_flag) ) ) ask_help = 1 ;

     if ( ! strncmp(a,alloc_size_flag,strlen(alloc_size_flag) ) )
       total_alloc_size = atol( a + strlen(alloc_size_flag) );

     if ( ! strncmp(a,super_size_flag,strlen(super_size_flag) ) )
       min_superblock_size = atoi( a + strlen(super_size_flag) );

     if ( ! strncmp(a,fill_stride_flag,strlen(fill_stride_flag) ) )
       fill_stride = atoi( a + strlen(fill_stride_flag) );

     if ( ! strncmp(a,fill_level_flag,strlen(fill_level_flag) ) )
       fill_level = atoi( a + strlen(fill_level_flag) );

     if ( ! strncmp(a,chunk_span_flag,strlen(chunk_span_flag) ) )
       chunk_span = atoi( a + strlen(chunk_span_flag) );

     if ( ! strncmp(a,repeat_outer_flag,strlen(repeat_outer_flag) ) )
       repeat_outer = atoi( a + strlen(repeat_outer_flag) );

     if ( ! strncmp(a,repeat_inner_flag,strlen(repeat_inner_flag) ) )
       repeat_inner = atoi( a + strlen(repeat_inner_flag) );
  }

  const int mean_chunk   = TestFunctor::chunk * ( 1 + ( chunk_span / 2 ) );
  const int number_alloc = double(total_alloc_size) * double(fill_level) /
                           ( double(mean_chunk) * double(100) );

  double time = 0 ;

  int error = 0 ;

  if ( ask_help ) {
    std::cout << "command line options:"
              << " " << help_flag
              << " " << alloc_size_flag << "##"
              << " " << super_size_flag << "##"
              << " " << fill_stride_flag << "##"
              << " " << fill_level_flag << "##"
              << " " << chunk_span_flag << "##"
              << " " << repeat_outer_flag << "##"
              << " " << repeat_inner_flag << "##"
              << std::endl ;
  }
  else {

    Kokkos::initialize(argc,argv);

    TestFunctor functor( total_alloc_size
                       , min_superblock_size
                       , number_alloc
                       , fill_stride
                       , chunk_span
                       , repeat_inner );

    if ( ! functor.test_fill() ) {
      Kokkos::abort("  fill failed");
    }

    Kokkos::Impl::Timer timer ;

    for ( int i = 0 ; i < repeat_outer ; ++i ) {
      error |= ! functor.test_alloc_dealloc();
    }

    time = timer.seconds();

    Kokkos::finalize();
  }

  printf( "\"mempool: alloc super stride level span inner outer number\" %ld %d %d %d %d %d %d %d\n"
        , total_alloc_size
        , min_superblock_size
        , fill_stride
        , fill_level
        , chunk_span
        , repeat_inner
        , repeat_outer
        , number_alloc );

  printf( "\"mempool: alloc/dealloc test time\" %.8f\n"
        , time );

  printf( "\"mempool: alloc/dealloc pairs per second\" %f\n"
        , (number_alloc * repeat_inner) / time );

  if ( error ) { printf("  TEST FAILED\n"); }

  return 0 ;
}

