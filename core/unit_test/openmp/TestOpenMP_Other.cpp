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

#include <openmp/TestOpenMP.hpp>

#include <mutex>

namespace Test {

TEST_F( openmp, init )
{
  ;
}

TEST_F( openmp, mdrange_for )
{
  Kokkos::Timer timer;
  TestMDRange_2D< Kokkos::OpenMP >::test_for2( 10000, 1000 );
  std::cout << " 2D: " << timer.seconds() << std::endl;

  timer.reset();
  TestMDRange_3D< Kokkos::OpenMP >::test_for3( 100, 100, 1000 );
  std::cout << " 3D: " << timer.seconds() << std::endl;

  timer.reset();
  TestMDRange_4D< Kokkos::OpenMP >::test_for4( 100, 10, 100, 100 );
  std::cout << " 4D: " << timer.seconds() << std::endl;

  timer.reset();
  TestMDRange_5D< Kokkos::OpenMP >::test_for5( 100, 10, 10, 100, 50 );
  std::cout << " 5D: " << timer.seconds() << std::endl;

  timer.reset();
  TestMDRange_6D< Kokkos::OpenMP >::test_for6( 10, 10, 10, 10, 50, 50 );
  std::cout << " 6D: " << timer.seconds() << std::endl;
}

TEST_F( openmp, mdrange_reduce )
{
  TestMDRange_2D< Kokkos::OpenMP >::test_reduce2( 100, 100 );
  TestMDRange_3D< Kokkos::OpenMP >::test_reduce3( 100, 10, 100 );
}

TEST_F( openmp, policy_construction )
{
  TestRangePolicyConstruction< Kokkos::OpenMP >();
  TestTeamPolicyConstruction< Kokkos::OpenMP >();
}

TEST_F( openmp, range_tag )
{
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_for( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_reduce( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_scan( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_for( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_reduce( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_scan( 0 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_dynamic_policy( 0 );

  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_for( 2 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_reduce( 2 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_scan( 2 );

  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_for( 3 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_reduce( 3 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_scan( 3 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_dynamic_policy( 3 );

  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_for( 1000 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_reduce( 1000 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Static> >::test_scan( 1000 );

  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_for( 1001 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_reduce( 1001 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_scan( 1001 );
  TestRange< Kokkos::OpenMP, Kokkos::Schedule<Kokkos::Dynamic> >::test_dynamic_policy( 1000 );
}

//----------------------------------------------------------------------------

TEST_F( openmp, compiler_macros )
{
  ASSERT_TRUE( ( TestCompilerMacros::Test< Kokkos::OpenMP >() ) );
}

//----------------------------------------------------------------------------

TEST_F( openmp , concurrent_bitset )
{
  test_concurrent_bitset< Kokkos::OpenMP >( 100000 );
}

//----------------------------------------------------------------------------

TEST_F( openmp, memory_pool )
{
  TestMemoryPool::test_memory_pool_v2< Kokkos::OpenMP >(false,false);
}

//----------------------------------------------------------------------------

#if defined( KOKKOS_ENABLE_TASKDAG )

TEST_F( openmp, task_spawn )
{
  TestTaskScheduler::TestTaskSpawn< Kokkos::OpenMP >::run();
}

TEST_F( openmp, task_fib )
{
  const int N = 24 ; // 25 triggers tbd bug on Cuda/Pascal
  for ( int i = 0; i < N; ++i ) {
    TestTaskScheduler::TestFib< Kokkos::OpenMP >::run( i, ( i + 1 ) * ( i + 1 ) * 10000 );
  }
}

TEST_F( openmp, task_depend )
{
  for ( int i = 0; i < 25; ++i ) {
    TestTaskScheduler::TestTaskDependence< Kokkos::OpenMP >::run( i );
  }
}

TEST_F( openmp, task_team )
{
  TestTaskScheduler::TestTaskTeam< Kokkos::OpenMP >::run( 1000 );
  //TestTaskScheduler::TestTaskTeamValue< Kokkos::OpenMP >::run( 1000 ); // Put back after testing.
}

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */

//----------------------------------------------------------------------------

#if defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP )
TEST_F( openmp, cxx11 )
{
  if ( std::is_same< Kokkos::DefaultExecutionSpace, Kokkos::OpenMP >::value ) {
    ASSERT_TRUE( ( TestCXX11::Test< Kokkos::OpenMP >( 1 ) ) );
    ASSERT_TRUE( ( TestCXX11::Test< Kokkos::OpenMP >( 2 ) ) );
    ASSERT_TRUE( ( TestCXX11::Test< Kokkos::OpenMP >( 3 ) ) );
    ASSERT_TRUE( ( TestCXX11::Test< Kokkos::OpenMP >( 4 ) ) );
  }
}
#endif

TEST_F( openmp, tile_layout )
{
  TestTile::test< Kokkos::OpenMP, 1, 1 >( 1, 1 );
  TestTile::test< Kokkos::OpenMP, 1, 1 >( 2, 3 );
  TestTile::test< Kokkos::OpenMP, 1, 1 >( 9, 10 );

  TestTile::test< Kokkos::OpenMP, 2, 2 >( 1, 1 );
  TestTile::test< Kokkos::OpenMP, 2, 2 >( 2, 3 );
  TestTile::test< Kokkos::OpenMP, 2, 2 >( 4, 4 );
  TestTile::test< Kokkos::OpenMP, 2, 2 >( 9, 9 );

  TestTile::test< Kokkos::OpenMP, 2, 4 >( 9, 9 );
  TestTile::test< Kokkos::OpenMP, 4, 2 >( 9, 9 );

  TestTile::test< Kokkos::OpenMP, 4, 4 >( 1, 1 );
  TestTile::test< Kokkos::OpenMP, 4, 4 >( 4, 4 );
  TestTile::test< Kokkos::OpenMP, 4, 4 >( 9, 9 );
  TestTile::test< Kokkos::OpenMP, 4, 4 >( 9, 11 );

  TestTile::test< Kokkos::OpenMP, 8, 8 >( 1, 1 );
  TestTile::test< Kokkos::OpenMP, 8, 8 >( 4, 4 );
  TestTile::test< Kokkos::OpenMP, 8, 8 >( 9, 9 );
  TestTile::test< Kokkos::OpenMP, 8, 8 >( 9, 11 );
}

TEST_F( openmp, dispatch )
{
  const int repeat = 100;
  for ( int i = 0; i < repeat; ++i ) {
    for ( int j = 0; j < repeat; ++j ) {
      Kokkos::parallel_for( Kokkos::RangePolicy< Kokkos::OpenMP >( 0, j )
                          , KOKKOS_LAMBDA( int ) {} );
    }
  }
}


TEST_F( openmp, partition_master )
{
  using Mutex = Kokkos::Experimental::MasterLock<Kokkos::OpenMP>;

  Mutex mtx;
  int errors = 0;

  auto master = [&errors, &mtx](int partition_id, int num_partitions) {

    const int pool_size = Kokkos::OpenMP::thread_pool_size();
    if ( partition_id == 0) {
      printf( "partitons: %2d   size: %2d\n", num_partitions, pool_size);
    }

    {
      std::unique_lock<Mutex> lock(mtx);
      if ( Kokkos::OpenMP::in_parallel() ) {
        ++errors;
      }
      if ( Kokkos::OpenMP::thread_pool_rank() != 0 ) {
        ++errors;
      }
    }

    {
      int local_errors = 0;
      Kokkos::parallel_reduce( 1000
                           , [pool_size]( const int , int & errs ) {
          if ( Kokkos::OpenMP::thread_pool_size() != pool_size ) {
            ++errs;
          }
        }
        , local_errors
      );
      Kokkos::atomic_add( &errors, local_errors );
    }

    Kokkos::Experimental::UniqueToken< Kokkos::OpenMP > token;

    Kokkos::View<int*> count( "",  token.size() );

    Kokkos::parallel_for( 1000, KOKKOS_LAMBDA ( const int ) {
      int i = token.acquire();
      ++count[i];
      token.release(i);
    });

    Kokkos::View<int> sum ("");
    Kokkos::parallel_for( token.size(), KOKKOS_LAMBDA ( const int i ) {
      Kokkos::atomic_add( sum.data(), count[i] );
    });

    if (sum() != 1000) {
      Kokkos::atomic_add( &errors, 1 );
    }
  };

  master(0,1);

  ASSERT_EQ( errors, 0 );

  Kokkos::OpenMP::partition_master( master );
  ASSERT_EQ( errors, 0 );

  Kokkos::OpenMP::partition_master( master, 4 );
  ASSERT_EQ( errors, 0 );

  Kokkos::OpenMP::partition_master( master, 0, 4 );
  ASSERT_EQ( errors, 0 );

  Kokkos::OpenMP::partition_master( master, 2, 2 );
  ASSERT_EQ( errors, 0 );
}

} // namespace Test
