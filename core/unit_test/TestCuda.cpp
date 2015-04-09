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

#include <gtest/gtest.h>

#include <iostream>

#include <Kokkos_Core.hpp>

//----------------------------------------------------------------------------

#include <impl/Kokkos_ViewTileLeft.hpp>

//----------------------------------------------------------------------------

#include <TestViewImpl.hpp>
#include <TestAtomic.hpp>

#include <TestViewAPI.hpp>
#include <TestViewSubview.hpp>
#include <TestTile.hpp>

#include <TestReduce.hpp>
#include <TestScan.hpp>
#include <TestRange.hpp>
#include <TestTeam.hpp>
#include <TestAggregate.hpp>
#include <TestAggregateReduction.hpp>
#include <TestCompilerMacros.hpp>
#include <TestMemorySpaceTracking.hpp>
#include <TestTeamVector.hpp>
#include <TestTemplateMetaFunctions.hpp>
#include <TestCXX11Deduction.hpp>

//----------------------------------------------------------------------------

class cuda : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
    Kokkos::Cuda::print_configuration( std::cout );
    Kokkos::HostSpace::execution_space::initialize();
    Kokkos::Cuda::initialize( Kokkos::Cuda::SelectDevice(0) );
  }
  static void TearDownTestCase()
  {
    Kokkos::Cuda::finalize();
    Kokkos::HostSpace::execution_space::finalize();
  }
};

//----------------------------------------------------------------------------

namespace Test {

__global__
void test_abort()
{
  Kokkos::Impl::VerifyExecutionCanAccessMemorySpace<
    Kokkos::CudaSpace ,
    Kokkos::HostSpace >::verify();
}

__global__
void test_cuda_spaces_int_value( int * ptr )
{
  if ( *ptr == 42 ) { *ptr = 2 * 42 ; }
}


TEST_F( cuda , compiler_macros )
{
  ASSERT_TRUE( ( TestCompilerMacros::Test< Kokkos::Cuda >() ) );
}

TEST_F( cuda , memory_space )
{
  TestMemorySpace< Kokkos::Cuda >();
}

TEST_F( cuda, spaces )
{
  if ( Kokkos::CudaUVMSpace::available() ) {

    Kokkos::Impl::AllocationTracker tracker = Kokkos::CudaUVMSpace::allocate_and_track("uvm_ptr",sizeof(int));

    int * uvm_ptr = (int*) tracker.alloc_ptr();

    *uvm_ptr = 42 ;

    Kokkos::Cuda::fence();
    test_cuda_spaces_int_value<<<1,1>>>(uvm_ptr);
    Kokkos::Cuda::fence();

    EXPECT_EQ( *uvm_ptr, int(2*42) );

  }
}


TEST_F( cuda, view_impl )
{
  // test_abort<<<32,32>>>(); // Aborts the kernel with CUDA version 4.1 or greater

  test_view_impl< Kokkos::Cuda >();
}

TEST_F( cuda, view_api )
{
  typedef Kokkos::View< const int * , Kokkos::Cuda , Kokkos::MemoryTraits< Kokkos::RandomAccess > > view_texture_managed ;
  typedef Kokkos::View< const int * , Kokkos::Cuda , Kokkos::MemoryTraits< Kokkos::RandomAccess | Kokkos::Unmanaged > > view_texture_unmanaged ;

  TestViewAPI< double , Kokkos::Cuda >();

#if 0
  Kokkos::View<double, Kokkos::Cuda > x("x");
  Kokkos::View<double[1], Kokkos::Cuda > y("y");
  // *x = 10 ;
  // x() = 10 ;
  // y[0] = 10 ;
  // y(0) = 10 ;
#endif
}


TEST_F( cuda, view_subview_left_0 ) {
  TestViewSubview::test_left_0< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_left_1 ) {
  TestViewSubview::test_left_1< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_left_2 ) {
  TestViewSubview::test_left_2< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_left_3 ) {
  TestViewSubview::test_left_3< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_right_0 ) {
  TestViewSubview::test_right_0< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_right_1 ) {
  TestViewSubview::test_right_1< Kokkos::CudaUVMSpace >();
}

TEST_F( cuda, view_subview_right_3 ) {
  TestViewSubview::test_right_3< Kokkos::CudaUVMSpace >();
}




TEST_F( cuda, range_tag )
{
  TestRange< Kokkos::Cuda >::test_for(1000);
  TestRange< Kokkos::Cuda >::test_reduce(1000);
  TestRange< Kokkos::Cuda >::test_scan(1000);
}

TEST_F( cuda, team_tag )
{
  TestTeamPolicy< Kokkos::Cuda >::test_for(1000);
  TestTeamPolicy< Kokkos::Cuda >::test_reduce(1000);
}

TEST_F( cuda, reduce )
{
  TestReduce< long ,   Kokkos::Cuda >( 10000000 );
  TestReduce< double , Kokkos::Cuda >( 1000000 );
}

TEST_F( cuda, reduce_team )
{
  TestReduceTeam< long ,   Kokkos::Cuda >( 10000000 );
  TestReduceTeam< double , Kokkos::Cuda >( 1000000 );
}

TEST_F( cuda, shared_team )
{
  TestSharedTeam< Kokkos::Cuda >();
}

TEST_F( cuda, reduce_dynamic )
{
  TestReduceDynamic< long ,   Kokkos::Cuda >( 10000000 );
  TestReduceDynamic< double , Kokkos::Cuda >( 1000000 );
}

TEST_F( cuda, reduce_dynamic_view )
{
  TestReduceDynamicView< long ,   Kokkos::Cuda >( 10000000 );
  TestReduceDynamicView< double , Kokkos::Cuda >( 1000000 );
}

TEST_F( cuda, atomic )
{
  const int loop_count = 1e3 ;

  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::Cuda>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::Cuda>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::Cuda>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::Cuda>(100,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::Cuda>(100,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::Cuda>(100,3) ) );
}

//----------------------------------------------------------------------------

TEST_F( cuda, tile_layout)
{
  TestTile::test< Kokkos::Cuda , 1 , 1 >( 1 , 1 );
  TestTile::test< Kokkos::Cuda , 1 , 1 >( 2 , 3 );
  TestTile::test< Kokkos::Cuda , 1 , 1 >( 9 , 10 );

  TestTile::test< Kokkos::Cuda , 2 , 2 >( 1 , 1 );
  TestTile::test< Kokkos::Cuda , 2 , 2 >( 2 , 3 );
  TestTile::test< Kokkos::Cuda , 2 , 2 >( 4 , 4 );
  TestTile::test< Kokkos::Cuda , 2 , 2 >( 9 , 9 );

  TestTile::test< Kokkos::Cuda , 2 , 4 >( 9 , 9 );
  TestTile::test< Kokkos::Cuda , 4 , 4 >( 9 , 9 );

  TestTile::test< Kokkos::Cuda , 4 , 4 >( 1 , 1 );
  TestTile::test< Kokkos::Cuda , 4 , 4 >( 4 , 4 );
  TestTile::test< Kokkos::Cuda , 4 , 4 >( 9 , 9 );
  TestTile::test< Kokkos::Cuda , 4 , 4 >( 9 , 11 );

  TestTile::test< Kokkos::Cuda , 8 , 8 >( 1 , 1 );
  TestTile::test< Kokkos::Cuda , 8 , 8 >( 4 , 4 );
  TestTile::test< Kokkos::Cuda , 8 , 8 >( 9 , 9 );
  TestTile::test< Kokkos::Cuda , 8 , 8 >( 9 , 11 );
}


TEST_F( cuda , view_aggregate )
{
  TestViewAggregate< Kokkos::Cuda >();
  TestViewAggregateReduction< Kokkos::Cuda >();
}


TEST_F( cuda , scan )
{
  TestScan< Kokkos::Cuda >::test_range( 1 , 1000 );
  TestScan< Kokkos::Cuda >( 1000000 );
  TestScan< Kokkos::Cuda >( 10000000 );
  Kokkos::Cuda::fence();
}

TEST_F( cuda , team_scan )
{
  TestScanTeam< Kokkos::Cuda >( 10 );
  TestScanTeam< Kokkos::Cuda >( 10000 );
}

}

//----------------------------------------------------------------------------

TEST_F( cuda , template_meta_functions )
{
  TestTemplateMetaFunctions<int, Kokkos::Cuda >();
}

//----------------------------------------------------------------------------

#ifdef KOKKOS_HAVE_CXX11

namespace Test {

TEST_F( cuda , reduction_deduction )
{
  TestCXX11::test_reduction_deduction< Kokkos::Cuda >();
}

TEST_F( cuda , team_vector )
{
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(0) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(1) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(2) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(3) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(4) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(5) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(6) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(7) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(8) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(9) ) );
  ASSERT_TRUE( ( TestTeamVector::Test< Kokkos::Cuda >(10) ) );
}

}
#endif

