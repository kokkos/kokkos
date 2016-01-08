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

#include <stdio.h>
#include <iostream>
#include <cmath>

#include <impl/Kokkos_Timer.hpp>

namespace TestMemoryPool {

struct pointer_obj {
  uint64_t * ptr;
};

template < typename PointerView, typename MemorySpace >
struct allocate_memory {
  typedef typename PointerView::execution_space  execution_space;
  typedef typename execution_space::size_type    size_type;

  PointerView m_pointers;
  size_t m_num_ptrs;
  size_t m_chunk_size;
  MemorySpace m_space;

  allocate_memory( PointerView & ptrs, size_t nptrs,
                   size_t cs, MemorySpace & sp )
    : m_pointers( ptrs ), m_num_ptrs( nptrs ),
      m_chunk_size( cs ), m_space( sp )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_for( m_num_ptrs, *this );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i ) const
  {
    m_pointers[i].ptr =
      static_cast< uint64_t * >( m_space.allocate( m_chunk_size ) );
  }
};

template < typename PointerView >
struct fill_memory {
  typedef typename PointerView::execution_space  execution_space;
  typedef typename execution_space::size_type    size_type;

  PointerView m_pointers;
  size_t m_num_ptrs;

  fill_memory( PointerView & ptrs, size_t nptrs )
    : m_pointers( ptrs ), m_num_ptrs( nptrs )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_for( m_num_ptrs, *this );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i ) const
  {
    *m_pointers[i].ptr = i;
  }
};

template < typename PointerView >
struct sum_memory {
  typedef typename PointerView::execution_space  execution_space;
  typedef typename execution_space::size_type    size_type;
  typedef uint64_t                               value_type;

  PointerView m_pointers;
  size_t m_num_ptrs;
  uint64_t & result;

  sum_memory( PointerView & ptrs, size_t nptrs, uint64_t & res )
    : m_pointers( ptrs ), m_num_ptrs( nptrs ), result( res )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_reduce( m_num_ptrs, *this, result );
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type & v ) const
  { v = 0; }

  KOKKOS_INLINE_FUNCTION
  void join( volatile value_type & dst, volatile value_type const & src ) const
  { dst += src; }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i, value_type & result ) const
  {
    result += *m_pointers[i].ptr;
  }
};

template < typename PointerView, typename MemorySpace >
struct deallocate_memory {
  typedef typename PointerView::execution_space  execution_space;
  typedef typename execution_space::size_type    size_type;

  PointerView m_pointers;
  size_t m_num_ptrs;
  size_t m_chunk_size;
  MemorySpace m_space;

  deallocate_memory( PointerView & ptrs, size_t nptrs,
                     size_t cs, MemorySpace & sp )
    : m_pointers( ptrs ), m_num_ptrs( nptrs ), m_chunk_size( cs ), m_space( sp )
  {
    // Initialize the view with the out degree of each vertex.
    Kokkos::parallel_for( m_num_ptrs, *this );
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( size_type i ) const
  {
    m_space.deallocate( m_pointers[i].ptr, m_chunk_size );
  }
};

#define PRECISION 6
#define SHIFTW 21
#define SHIFTW2 12

template < typename F >
void print_results( const std::string & text, F elapsed_time )
{
  std::cout << std::setw( SHIFTW ) << text << std::setw( SHIFTW2 )
            << std::fixed << std::setprecision( PRECISION ) << elapsed_time
            << std::endl;
}

template < typename F, typename T >
void print_results( const std::string & text, unsigned long long width,
                    F elapsed_time, T result )
{
  std::cout << std::setw( SHIFTW ) << text << std::setw( SHIFTW2 )
            << std::fixed << std::setprecision( PRECISION ) << elapsed_time
            << "     " << std::setw( width ) << result << std::endl;
}

template < typename F >
void print_results( const std::string & text, unsigned long long width,
                    F elapsed_time, const std::string & result )
{
  std::cout << std::setw( SHIFTW ) << text << std::setw( SHIFTW2 )
            << std::fixed << std::setprecision( PRECISION ) << elapsed_time
            << "     " << std::setw( width ) << result << std::endl;
}

template < class ExecSpace >
bool test_mempool( size_t chunk_size, size_t total_size )
{
  typedef Kokkos::View<pointer_obj*, ExecSpace>             pointer_view;
  typedef typename pointer_view::memory_space               memory_space;
  typedef Kokkos::Experimental::MemoryPool< memory_space >  pool_memory_space;

  uint64_t result;
  size_t num_chunks = total_size / chunk_size;
  double elapsed_time = 0;
  bool return_val = true;

  std::cout << std::setw( SHIFTW ) << "chunk_size: " << std::setw( 10 )
            << chunk_size << std::endl
            << std::setw( SHIFTW ) << "total_size: " << std::setw( 10 )
            << total_size << std::endl
            << std::setw( SHIFTW ) << "num_chunks: " << std::setw( 10 )
            << num_chunks << std::endl;

  pointer_view pointers( "pointers", num_chunks );

  memory_space mspace;
  pool_memory_space m_space( mspace, chunk_size, total_size );

  Kokkos::Impl::Timer timer;

  {
    allocate_memory< pointer_view, pool_memory_space >
      am( pointers, num_chunks, chunk_size, m_space );
  }

  ExecSpace::fence();
  elapsed_time = timer.seconds();
  print_results( "allocate chunks: ", elapsed_time );
  timer.reset();

  {
    fill_memory< pointer_view > fm( pointers, num_chunks );
  }

  ExecSpace::fence();
  elapsed_time = timer.seconds();
  print_results( "fill chunks: ", elapsed_time );
  timer.reset();

  {
    sum_memory< pointer_view > sm( pointers, num_chunks, result );
  }

  ExecSpace::fence();
  elapsed_time = timer.seconds();
  print_results( "sum chunks: ", 10, elapsed_time, result );

  if ( result != ( num_chunks * ( num_chunks - 1 ) ) / 2 ) {
    std::cerr << "Invalid sum value in memory." << std::endl;
    return_val = false;
  }

  timer.reset();

  {
    deallocate_memory< pointer_view, pool_memory_space >
      dm( pointers, num_chunks, chunk_size, m_space );
  }

  ExecSpace::fence();
  elapsed_time = timer.seconds();
  print_results( "deallocate chunks: ", elapsed_time );
  timer.reset();

  return return_val;
}

}

#endif
