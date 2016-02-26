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

/*--------------------------------------------------------------------------*/

#include <stdio.h>

namespace Test {

template< class Device , class WorkSpec = size_t >
struct TestScan {

  typedef  Device    execution_space ;
  typedef  long int  value_type ;

  KOKKOS_INLINE_FUNCTION
  void operator()( const int iwork , value_type & update , const bool final_pass ) const
  {
    const value_type n = iwork + 1 ;
    const value_type imbalance = ( (1000 <= n) && (0 == n % 1000) ) ? 1000 : 0 ;

    // Insert an artificial load imbalance

    for ( value_type i = 0 ; i < imbalance ; ++i ) { ++update ; }

    update += n - imbalance ;

    if ( final_pass ) {
      const value_type answer = n & 1 ? ( n * ( ( n + 1 ) / 2 ) ) : ( ( n / 2 ) * ( n + 1 ) );

      if ( answer != update ) {
        printf("TestScan(%d,%ld) != %ld\n",iwork,update,answer);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type & update ) const { update = 0 ; }

  KOKKOS_INLINE_FUNCTION
  void join( volatile       value_type & update ,
             volatile const value_type & input ) const
  { update += input ; }

  TestScan( const WorkSpec & N )
    { parallel_scan( N , *this ); }

  TestScan( const WorkSpec & Start , const WorkSpec & N )
    {
      typedef Kokkos::RangePolicy<execution_space> exec_policy ;
      parallel_scan( exec_policy( Start , N ) , *this );
    }

  static void test_range( const WorkSpec & begin , const WorkSpec & end )
    {
      for ( WorkSpec i = begin ; i < end ; ++i ) {
        (void) TestScan( i );
      }
    }
};

}

