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

#include <cstdio>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_ViewLayoutTiled.hpp>

#include <type_traits>
#include <typeinfo>

namespace Test {

namespace {

template <typename ExecSpace >
struct TestViewLayoutTiled {

  typedef double Scalar;

  static constexpr int T0 = 2;
  static constexpr int T1 = 4;
  static constexpr int T2 = 4;
  static constexpr int T3 = 2;
  static constexpr int T4 = 4;
  static constexpr int T5 = 4;
  static constexpr int T6 = 4;
  static constexpr int T7 = 4;

  // Rank 2
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Left, T0, T1>   LayoutLL_2D_2x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Left, T0, T1>  LayoutRL_2D_2x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Right, T0, T1>  LayoutLR_2D_2x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Right, T0, T1> LayoutRR_2D_2x4;

  // Rank 3
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Left, T0, T1, T2>   LayoutLL_3D_2x4x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Left, T0, T1, T2>  LayoutRL_3D_2x4x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Right, T0, T1, T2>  LayoutLR_3D_2x4x4;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Right, T0, T1, T2> LayoutRR_3D_2x4x4;

  // Rank 4
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Left, T0, T1, T2, T3>   LayoutLL_4D_2x4x4x2;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Left, T0, T1, T2, T3>  LayoutRL_4D_2x4x4x2;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Left, Kokkos::Pattern::Iterate::Right, T0, T1, T2, T3>  LayoutLR_4D_2x4x4x2;
  typedef Kokkos::LayoutTiled<Kokkos::Pattern::Iterate::Right, Kokkos::Pattern::Iterate::Right, T0, T1, T2, T3> LayoutRR_4D_2x4x4x2;


#define DEBUG_VERBOSE_OUTPUT 0

  // Rank 2 tests
  // Create N0 x N1 view with tile dimensions of T0 x T1
  static void test_view_layout_tiled_2d( const int N0, const int N1 )
  {

//    const int T0 = 2;
//    const int T1 = 4;
    const int FT = T0*T1;

    const int NT0 = int( std::ceil( N0 / T0 ) );
    const int NT1 = int( std::ceil( N1 / T1 ) );

    // 4x12 view - 48 items
    // tiledims 2x4
    // full tile 8
    // num tiles 2x3
    std::cout << "Begin output \n N0 = " << N0 << " N1 = " << N1 << "\n T0 = " << T0 << " T1 = " << T1 << "  FTD = " << FT << "  NT0 = " << NT0 << "  NT1 = " << NT1 << std::endl;
    long counter[4] = {0};
    // Create LL View
    {
      std::cout << "Init LL View" << std::endl;
      Kokkos::View< Scalar**, LayoutLL_2D_2x4, Kokkos::HostSpace > v("v", N0, N1);
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j) = ( ti + tj*NT0 )*FT + ( i + j*T0 );
        } }
      } }

      std::cout << "\nOutput L-L pattern 2x4 tiles:" << std::endl;
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj );
          if ( tile_subview(i,j) != v(ti*T0+i, tj*T1+j) ) { ++counter[0]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1 = " << ti*T0 + i << "," << tj*T1 + j << std::endl;
          std::cout << "ti,tj,i,j: " << ti << "," << tj << "," << i << "," << j << "  v = " << v(ti*T0 + i, tj*T1+j) << "  flat idx = " << ( ti + tj*NT0 )*FT + ( i + j*T0 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j) << std::endl;
#endif
        } }
      } }
    } // end scope

    // Create RL View
    {
      std::cout << "\nInit RL View" << std::endl;
      Kokkos::View< Scalar**, LayoutRL_2D_2x4, Kokkos::HostSpace > v("v", N0, N1);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j) = ( ti*NT1 + tj )*FT + ( i + j*T0 );
        } }
      } }

      std::cout << "\nOutput R-L pattern 2x4 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj );
          if ( tile_subview(i,j) != v(ti*T0+i, tj*T1+j) ) { ++counter[1]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1 = " << ti*T0 + i << "," << tj*T1 + j << std::endl;
          std::cout << "ti,tj,i,j: " << ti << "," << tj << "," << i << "," << j << "  v = " << v(ti*T0 + i, tj*T1+j) << "  flat idx = " << ( ti*NT1 + tj )*FT + ( i + j*T0 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j) << std::endl;
#endif
        } }
      } }
    } // end scope

    // Create LR View
    {
      std::cout << "\nInit LR View" << std::endl;
      Kokkos::View< Scalar**, LayoutLR_2D_2x4, Kokkos::HostSpace > v("v", N0, N1);
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
          v(ti*T0 + i, tj*T1+j) = ( ti + tj*NT0 )*FT + ( i*T1 + j );
        } }
      } }

      std::cout << "\nOutput L-R pattern 2x4 tiles:" << std::endl;
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj );
          if ( tile_subview(i,j) != v(ti*T0+i, tj*T1+j) ) { ++counter[2]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1 = " << ti*T0 + i << "," << tj*T1 + j << std::endl;
          std::cout << "ti,tj,i,j: " << ti << "," << tj << "," << i << "," << j << "  v = " << v(ti*T0 + i, tj*T1+j) << "  flat idx = " << ( ti + tj*NT0 )*FT + ( i*T1 + j ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j) << std::endl;
#endif
        } }
      } }
    } // end scope

    // Create RR View
    {
      std::cout << "\nInit RR View" << std::endl;
      Kokkos::View< Scalar**, LayoutRR_2D_2x4, Kokkos::HostSpace > v("v", N0, N1);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
          v(ti*T0 + i, tj*T1+j) = ( ti*NT1 + tj )*FT + ( i*T1 + j );
        } }
      } }

      std::cout << "\nOutput R-R pattern 2x4 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj );
          if ( tile_subview(i,j) != v(ti*T0+i, tj*T1+j) ) { ++counter[3]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1 = " << ti*T0 + i << "," << tj*T1 + j << std::endl;
          std::cout << "ti,tj,i,j: " << ti << "," << tj << "," << i << "," << j << "  v = " << v(ti*T0 + i, tj*T1+j) << "  flat idx = " << ( ti*NT1 + tj )*FT + ( i*T1 + j ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } }
      } }
    } // end scope

#if DEBUG_SUMMARY_OUTPUT
    std::cout << "subview_tile vs view errors:\n"
      << " LL: " << counter[0]
      << " RL: " << counter[1]
      << " LR: " << counter[2]
      << " RR: " << counter[3] 
      << std::endl;
#endif

    ASSERT_EQ(counter[0], long(0));
    ASSERT_EQ(counter[1], long(0));
    ASSERT_EQ(counter[2], long(0));
    ASSERT_EQ(counter[3], long(0));

    // Test create_mirror_view, deep_copy
    {
      std::cout << "\nCreate LL View - test mirror view and deep_copy and use in parallel_for" << std::endl;
      typedef typename Kokkos::View< Scalar**, LayoutLL_2D_2x4, ExecSpace > ViewType;
      ViewType v("v", N0, N1);

      typename ViewType::HostMirror hv = Kokkos::create_mirror_view(v);

      std::cout << "  ViewType: " << typeid(ViewType()).name() << "\n  HostMirror: " << typeid(typename ViewType::HostMirror() ).name() << std::endl;
      std::cout << "  ViewType Layout: " << typeid(typename ViewType::array_layout()).name() << "\n  HostMirror Layout: " << typeid(typename ViewType::HostMirror::array_layout() ).name() << std::endl;

      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          hv(ti*T0 + i, tj*T1+j) = ( ti + tj*NT0 )*FT + ( i + j*T0 );
        } }
      } }
#if 1
      Kokkos::deep_copy(v, hv);

      Kokkos::MDRangePolicy< Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>, ExecSpace > mdrangepolicy( {0,0}, {NT0, NT1}, {T0,T1} );

      Kokkos::parallel_for( "ViewTile rank 2 test 1", mdrangepolicy, 
        KOKKOS_LAMBDA (const int ti, const int tj) {
          for ( int j = 0; j < T1; ++j ) {
          for ( int i = 0; i < T0; ++i ) {
            if ( (ti*T0 + i < N0) && (tj*T1 + j < N1) ) { v(ti*T0 + i, tj*T1+j) += 1; }
          } }
        });

      Kokkos::deep_copy(hv, v);

      long counter_subview = 0;
      long counter_inc = 0;
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        auto tile_subview = Kokkos::tile_subview( hv, ti, tj );
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          if ( tile_subview(i,j) != hv(ti*T0+i, tj*T1+j) ) { ++counter_subview; }
          if ( tile_subview(i,j) != (( ti + tj*NT0 )*FT + ( i + j*T0 ) + 1 )) { ++counter_inc; }
        } }
      } }
      ASSERT_EQ(counter_subview, long(0));
      ASSERT_EQ(counter_inc, long(0));
#endif    
    }

  } // end test_view_layout_tiled_2d


  static void test_view_layout_tiled_3d( const int N0, const int N1, const int N2 )
  {

    const int FT = T0*T1*T2;

    const int NT0 = int( std::ceil( N0 / T0 ) );
    const int NT1 = int( std::ceil( N1 / T1 ) );
    const int NT2 = int( std::ceil( N2 / T2 ) );

    // 4x12x16 view
    // tiledims 2x4x4
    // full tile 32
    // num tiles 2x3x4
    std::cout << "\nBegin output \n N0 = " << N0 << " N1 = " << N1 << " N2 = " << N2 << "\n T0 = " << T0 << " T1 = " << T1 << " T2 = " << T2 << "  FTD = " << FT << "  NT0 = " << NT0 << "  NT1 = " << NT1 << " NT2 = " << NT2 << std::endl;
    long counter[4] = {0};
    // Create LL View
    {
      std::cout << "Init LL View" << std::endl;
      Kokkos::View< Scalar***, LayoutLL_3D_2x4x4, Kokkos::HostSpace > v("v", N0, N1, N2);
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k) = ( ti + tj*NT0 + tk*N0*N1 )*FT + ( i + j*T0 + k*T0*T1 );
        } } }
      } } }

      std::cout << "\nOutput L-L pattern 2x4x4 tiles:" << std::endl;
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk );
          if ( tile_subview(i,j,k) != v(ti*T0+i, tj*T1+j, tk*T2+k) ) { ++counter[0]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << std::endl;
          std::cout << "ti,tj,tk,i,j,k: " << ti << "," << tj << "," << tk << "," << i << "," << j << "," << k << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k) << "  flat idx = " << ( ti + tj*NT0 + tk*N0*N1 )*FT + ( i + j*T0 + k*T0*T1 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } }
      } } }
    } // end scope

    // Create RL View
    {
      std::cout << "\nInit RL View" << std::endl;
      Kokkos::View< Scalar***, LayoutRL_3D_2x4x4, Kokkos::HostSpace > v("v", N0, N1, N2);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k) = ( ti*NT1*NT2 + tj*NT2 + tk )*FT + ( i + j*T0 + k*T0*T1 );
        } } }
      } } }

      std::cout << "\nOutput R-L pattern 2x4x4 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk );
          if ( tile_subview(i,j,k) != v(ti*T0+i, tj*T1+j, tk*T2+k) ) { ++counter[1]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << std::endl;
          std::cout << "ti,tj,tk,i,j,k: " << ti << "," << tj << "," << tk << "," << i << "," << j << "," << k << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k) << "  flat idx = " << ( ti*NT1*NT2 + tj*NT2 + tk )*FT + ( i + j*T0 + k*T0*T1 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k) << std::endl;
#endif
        } } }
      } } }
    } // end scope

    // Create LR View
    {
      std::cout << "\nInit LR View" << std::endl;
      Kokkos::View< Scalar***, LayoutLR_3D_2x4x4, Kokkos::HostSpace > v("v", N0, N1, N2);
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k) = ( ti + tj*NT0 + tk*NT0*NT1 )*FT + ( i*T1*T2 + j*T2 + k );
        } } }
      } } }

      std::cout << "\nOutput L-R pattern 2x4x4 tiles:" << std::endl;
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk );
          if ( tile_subview(i,j,k) != v(ti*T0+i, tj*T1+j, tk*T2+k) ) { ++counter[2]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << std::endl;
          std::cout << "ti,tj,tk,i,j,k: " << ti << "," << tj << "," << tk << "," << i << "," << j << "," << k << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k) << "  flat idx = " << ( ti + tj*NT0 + tk*NT0*NT1 )*FT + ( i*T1*T2 + j*T2 + k ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } }
      } } }
    } // end scope

    // Create RR View
    {
      std::cout << "\nInit RR View" << std::endl;
      Kokkos::View< Scalar***, LayoutRR_3D_2x4x4, Kokkos::HostSpace > v("v", N0, N1, N2);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k) = ( ti*NT1*NT2 + tj*NT2 + tk )*FT + ( i*T1*T2 + j*T2 + k );
        } } }
      } } }

      std::cout << "\nOutput R-R pattern 2x4x4 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk );
          if ( tile_subview(i,j,k) != v(ti*T0+i, tj*T1+j, tk*T2+k) ) { ++counter[3]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << std::endl;
          std::cout << "ti,tj,tk,i,j,k: " << ti << "," << tj << "," << tk << "," << i << "," << j << "," << k << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k) << "  flat idx = " << ( ti*NT1*NT2 + tj*NT2 + tk )*FT + ( i*T1*T2 + j*T2 + k ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } }
      } } }
    } // end scope

#if DEBUG_SUMMARY_OUTPUT
    std::cout << "subview_tile vs view errors:\n"
      << " LL: " << counter[0]
      << " RL: " << counter[1]
      << " LR: " << counter[2]
      << " RR: " << counter[3] 
      << std::endl;
#endif

    ASSERT_EQ(counter[0], long(0));
    ASSERT_EQ(counter[1], long(0));
    ASSERT_EQ(counter[2], long(0));
    ASSERT_EQ(counter[3], long(0));

  } // end test_view_layout_tiled_3d


  static void test_view_layout_tiled_4d( const int N0, const int N1, const int N2, const int N3 )
  {

    const int FT = T0*T1*T2*T3;

    const int NT0 = int( std::ceil( N0 / T0 ) );
    const int NT1 = int( std::ceil( N1 / T1 ) );
    const int NT2 = int( std::ceil( N2 / T2 ) );
    const int NT3 = int( std::ceil( N3 / T3 ) );

    // 4x12x16x12 view
    // tiledims 2x4x4x2
    // full tile 64
    // num tiles 2x3x4x6
    std::cout << "\nBegin output \n N0 = " << N0 << " N1 = " << N1 << " N2 = " << N2  << " N3 = " << N3 << "\n T0 = " << T0 << " T1 = " << T1 << " T2 = " << T2 << " T3 = " << T3 << "  FTD = " << FT << "  NT0 = " << NT0 << "  NT1 = " << NT1 << " NT2 = " << NT2 << " NT3 = " << NT3 << std::endl;
    long counter[4] = {0};
    // Create LL View
    {
      std::cout << "Init LL View" << std::endl;
      Kokkos::View< Scalar****, LayoutLL_4D_2x4x4x2, Kokkos::HostSpace > v("v", N0, N1, N2, N3);
      for ( int tl = 0; tl < NT3; ++tl ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int l = 0; l < T3; ++l ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) = ( ti + tj*NT0 + tk*N0*N1 + tl*N0*N1*N2 )*FT + ( i + j*T0 + k*T0*T1 + l*T0*T1*T2 );
        } } } }
      } } } }

      std::cout << "\nOutput L-L pattern 2x4x4x2 tiles:" << std::endl;
      for ( int tl = 0; tl < NT3; ++tl ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int l = 0; l < T3; ++l ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk, tl );
          if ( tile_subview(i,j,k,l) != v(ti*T0+i, tj*T1+j, tk*T2+k, tl*T3 + l) ) { ++counter[0]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2,idx3 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << "," << tl*T3 + l<< std::endl;
          std::cout << "ti,tj,tk,tl: " << ti << "," << tj << "," << tk << "," << tl << ","
          << "  i,j,k,l: " <<  i << "," << j << "," << k << "," << l
          << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) 
          << "  flat idx = " << ( ti + tj*NT0 + tk*N0*N1 + tl*N0*N1*N2 )*FT + ( i + j*T0 + k*T0*T1 + l*T0*T1*T2 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k,l) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } } }
      } } } }
    } // end scope

    // Create RL View
    {
      std::cout << "\nInit RL View" << std::endl;
      Kokkos::View< Scalar****, LayoutRL_4D_2x4x4x2, Kokkos::HostSpace > v("v", N0, N1, N2, N3);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tl = 0; tl < NT3; ++tl ) {
        for ( int l = 0; l < T3; ++l ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) = ( ti*NT1*NT2*N3 + tj*NT2*N3 + tk*N3 + tl )*FT + ( i + j*T0 + k*T0*T1 + l*T0*T1*T2 );
        } } } }
      } } } }

      std::cout << "\nOutput R-L pattern 2x4x4x2 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tl = 0; tl < NT3; ++tl ) {
        for ( int l = 0; l < T3; ++l ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int i = 0; i < T0; ++i ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk, tl );
          if ( tile_subview(i,j,k,l) != v(ti*T0+i, tj*T1+j, tk*T2+k, tl*T3 + l) ) { ++counter[1]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2,idx3 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << "," << tl*T3 + l<< std::endl;
          std::cout << "ti,tj,tk,tl: " << ti << "," << tj << "," << tk << "," << tl << ","
          << "  i,j,k,l: " <<  i << "," << j << "," << k << "," << l
          << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) 
          << "  flat idx = " << ( ti*NT1*NT2*N3 + tj*NT2*N3 + tk*N3 + tl )*FT + ( i + j*T0 + k*T0*T1 + l*T0*T1*T2 ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k,l) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } } }
      } } } }
    } // end scope

    // Create LR View
    {
      std::cout << "\nInit LR View" << std::endl;
      Kokkos::View< Scalar****, LayoutLR_4D_2x4x4x2, Kokkos::HostSpace > v("v", N0, N1, N2, N3);
      for ( int tl = 0; tl < NT3; ++tl ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int l = 0; l < T3; ++l ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) = ( ti + tj*NT0 + tk*NT0*NT1 + tl*NT0*NT1*NT2 )*FT + ( i*T1*T2*T3 + j*T2*T3 + k*T3 + l );
        } } } }
      } } } }

      std::cout << "\nOutput L-R pattern 2x4x4x2 tiles:" << std::endl;
      for ( int tl = 0; tl < NT3; ++tl ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int ti = 0; ti < NT0; ++ti ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int l = 0; l < T3; ++l ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk, tl );
          if ( tile_subview(i,j,k,l) != v(ti*T0+i, tj*T1+j, tk*T2+k, tl*T3 + l) ) { ++counter[2]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2,idx3 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << "," << tl*T3 + l<< std::endl;
          std::cout << "ti,tj,tk,tl: " << ti << "," << tj << "," << tk << "," << tl << ","
          << "  i,j,k,l: " <<  i << "," << j << "," << k << "," << l
          << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) 
          << "  flat idx = " << ( ti + tj*NT0 + tk*NT0*NT1 + tl*NT0*NT1*NT2 )*FT + ( i*T1*T2*T3 + j*T2*T3 + k*T3 + l ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k,l) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } } }
      } } } }
    } // end scope

    // Create RR View
    {
      std::cout << "\nInit RR View" << std::endl;
      Kokkos::View< Scalar****, LayoutRR_4D_2x4x4x2, Kokkos::HostSpace > v("v", N0, N1, N2, N3);
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tl = 0; tl < NT3; ++tl ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int l = 0; l < T3; ++l ) {
          v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) = ( ti*NT1*NT2*NT3 + tj*NT2*NT3 + tk*NT3 + tl )*FT + ( i*T1*T2*T3 + j*T2*T3 + k*T3 + l );
        } } } }
      } } } }

      std::cout << "\nOutput R-R pattern 2x4x4x2 tiles:" << std::endl;
      for ( int ti = 0; ti < NT0; ++ti ) {
      for ( int tj = 0; tj < NT1; ++tj ) {
      for ( int tk = 0; tk < NT2; ++tk ) {
      for ( int tl = 0; tl < NT3; ++tl ) {
        for ( int i = 0; i < T0; ++i ) {
        for ( int j = 0; j < T1; ++j ) {
        for ( int k = 0; k < T2; ++k ) {
        for ( int l = 0; l < T3; ++l ) {
          auto tile_subview = Kokkos::tile_subview( v, ti, tj, tk, tl );
          if ( tile_subview(i,j,k,l) != v(ti*T0+i, tj*T1+j, tk*T2+k, tl*T3 + l) ) { ++counter[3]; }
#if DEBUG_VERBOSE_OUTPUT
          std::cout << "idx0,idx1,idx2,idx3 = " << ti*T0 + i << "," << tj*T1 + j << "," << tk*T2 + k << "," << tl*T3 + l<< std::endl;
          std::cout << "ti,tj,tk,tl: " << ti << "," << tj << "," << tk << "," << tl << ","
          << "  i,j,k,l: " <<  i << "," << j << "," << k << "," << l
          << "  v = " << v(ti*T0 + i, tj*T1+j, tk*T2 + k, tl*T3 + l) 
          << "  flat idx = " << ( ti*NT1*NT2*NT3 + tj*NT2*NT3 + tk*NT3 + tl )*FT + ( i*T1*T2*T3 + j*T2*T3 + k*T3 + l ) << std::endl;
          std::cout << "subview_tile output = " << tile_subview(i,j,k,l) << std::endl;
          std::cout << "subview tile rank = " << Kokkos::rank(tile_subview) << std::endl;
#endif
        } } } }
      } } } }
    } // end scope

#if DEBUG_SUMMARY_OUTPUT
    std::cout << "subview_tile vs view errors:\n"
      << " LL: " << counter[0]
      << " RL: " << counter[1]
      << " LR: " << counter[2]
      << " RR: " << counter[3] 
      << std::endl;
#endif

    ASSERT_EQ(counter[0], long(0));
    ASSERT_EQ(counter[1], long(0));
    ASSERT_EQ(counter[2], long(0));
    ASSERT_EQ(counter[3], long(0));

  } // end test_view_layout_tiled_4d



#ifdef DEBUG_VERBOSE_OUTPUT
#undef DEBUG_VERBOSE_OUTPUT
#endif
#ifdef DEBUG_SUMMARY_OUTPUT
#undef DEBUG_SUMMARY_OUTPUT
#endif

}; // end struct

} // namespace

TEST_F( TEST_CATEGORY , view_layouttiled) {
  // These two examples are iterating by tile, then within a tile - not by extents
  // If N# is not a power of two, but want to iterate by tile then within a tile, need to check that mapped index is within extent
  TestViewLayoutTiled< TEST_EXECSPACE >::test_view_layout_tiled_2d( 4, 12 );
  TestViewLayoutTiled< TEST_EXECSPACE >::test_view_layout_tiled_3d( 4, 12, 16 );
  TestViewLayoutTiled< TEST_EXECSPACE >::test_view_layout_tiled_4d( 4, 12, 16, 12 );
}

} // namespace Test
