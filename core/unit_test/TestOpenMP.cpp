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

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_LAMBDA
#undef KOKKOS_LAMBDA
#endif
#define KOKKOS_LAMBDA [=]

#include <Kokkos_Core.hpp>

//----------------------------------------------------------------------------

#include <TestAtomic.hpp>
#include <TestAtomicOperations.hpp>

#include <TestViewAPI.hpp>
#include <TestViewSubview.hpp>
#include <TestViewOfClass.hpp>

#include <TestSharedAlloc.hpp>
#include <TestViewMapping.hpp>

#include <TestRange.hpp>
#include <TestTeam.hpp>
#include <TestReduce.hpp>
#include <TestScan.hpp>
#include <TestAggregate.hpp>
#include <TestCompilerMacros.hpp>
#include <TestMemoryPool.hpp>


#include <TestCXX11.hpp>
#include <TestCXX11Deduction.hpp>
#include <TestTeamVector.hpp>
#include <TestTemplateMetaFunctions.hpp>

#include <TestPolicyConstruction.hpp>

#include <TestMDRange.hpp>

namespace Test {

class openmp : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
    const unsigned numa_count       = Kokkos::hwloc::get_available_numa_count();
    const unsigned cores_per_numa   = Kokkos::hwloc::get_available_cores_per_numa();
    const unsigned threads_per_core = Kokkos::hwloc::get_available_threads_per_core();

    const unsigned threads_count = std::max( 1u , numa_count ) *
                                   std::max( 2u , ( cores_per_numa * threads_per_core ) / 2 );

    Kokkos::OpenMP::initialize( threads_count );
    Kokkos::OpenMP::print_configuration( std::cout , true );
    srand(10231);
  }

  static void TearDownTestCase()
  {
    Kokkos::OpenMP::finalize();

    omp_set_num_threads(1);

    ASSERT_EQ( 1 , omp_get_max_threads() );
  }
};


TEST_F( openmp , md_range ) {
  TestMDRange_2D< Kokkos::OpenMP >::test_for2(100,100);

  TestMDRange_3D< Kokkos::OpenMP >::test_for3(100,100,100);
}

TEST_F( openmp , impl_shared_alloc ) {
  test_shared_alloc< Kokkos::HostSpace , Kokkos::OpenMP >();
}

TEST_F( openmp, policy_construction) {
  TestRangePolicyConstruction< Kokkos::OpenMP >();
  TestTeamPolicyConstruction< Kokkos::OpenMP >();
}

TEST_F( openmp , impl_view_mapping ) {
  test_view_mapping< Kokkos::OpenMP >();
  test_view_mapping_subview< Kokkos::OpenMP >();
  test_view_mapping_operator< Kokkos::OpenMP >();
  TestViewMappingAtomic< Kokkos::OpenMP >::run();
}

TEST_F( openmp, view_api) {
  TestViewAPI< double , Kokkos::OpenMP >();
}

TEST_F( openmp , view_nested_view )
{
  ::Test::view_nested_view< Kokkos::OpenMP >();
}

TEST_F( openmp , atomics )
{
  const int loop_count = 1e4 ;

  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<int,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned int,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long int,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<unsigned long int,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<long long int,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::OpenMP>(loop_count,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::OpenMP>(loop_count,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<double,Kokkos::OpenMP>(loop_count,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::OpenMP>(100,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::OpenMP>(100,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<float,Kokkos::OpenMP>(100,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<Kokkos::complex<double> ,Kokkos::OpenMP>(100,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<Kokkos::complex<double> ,Kokkos::OpenMP>(100,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<Kokkos::complex<double> ,Kokkos::OpenMP>(100,3) ) );

  ASSERT_TRUE( ( TestAtomic::Loop<TestAtomic::SuperScalar<4> ,Kokkos::OpenMP>(100,1) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<TestAtomic::SuperScalar<4> ,Kokkos::OpenMP>(100,2) ) );
  ASSERT_TRUE( ( TestAtomic::Loop<TestAtomic::SuperScalar<4> ,Kokkos::OpenMP>(100,3) ) );
}

TEST_F( openmp , atomic_operations )
{
  const int start = 1; //Avoid zero for division
  const int end = 11;

  for (int i = start; i < end; ++i)
  {
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 4 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 5 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 6 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 7 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 8 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 9 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 11 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<int,Kokkos::OpenMP>(start, end-i, 12 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 4 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 5 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 6 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 7 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 8 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 9 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 11 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned int,Kokkos::OpenMP>(start, end-i, 12 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 4 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 5 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 6 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 7 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 8 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 9 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 11 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long int,Kokkos::OpenMP>(start, end-i, 12 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 4 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 5 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 6 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 7 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 8 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 9 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 11 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<unsigned long int,Kokkos::OpenMP>(start, end-i, 12 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 4 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 5 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 6 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 7 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 8 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 9 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 11 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestIntegralType<long long int,Kokkos::OpenMP>(start, end-i, 12 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<double,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<double,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<double,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<double,Kokkos::OpenMP>(start, end-i, 4 ) ) );

    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<float,Kokkos::OpenMP>(start, end-i, 1 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<float,Kokkos::OpenMP>(start, end-i, 2 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<float,Kokkos::OpenMP>(start, end-i, 3 ) ) );
    ASSERT_TRUE( ( TestAtomicOperations::AtomicOperationsTestNonIntegralType<float,Kokkos::OpenMP>(start, end-i, 4 ) ) );
  }

}

} // namespace test

