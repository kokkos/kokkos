/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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

#include <qthreads/TestQthreads.hpp>

namespace Test {

TEST_F(qthreads, long_reduce) {
#if 0
  TestReduce< long, Kokkos::Qthreads >( 0 );
  TestReduce< long, Kokkos::Qthreads >( 1000000 );
#endif
}

TEST_F(qthreads, double_reduce) {
#if 0
  TestReduce< double, Kokkos::Qthreads >( 0 );
  TestReduce< double, Kokkos::Qthreads >( 1000000 );
#endif
}

TEST_F(qthreads, reducers) {
#if 0
  TestReducers< int, Kokkos::Qthreads >::execute_integer();
  TestReducers< size_t, Kokkos::Qthreads >::execute_integer();
  TestReducers< double, Kokkos::Qthreads >::execute_float();
  TestReducers< Kokkos::complex<double >, Kokkos::Qthreads>::execute_basic();
#endif
}

TEST_F(qthreads, long_reduce_dynamic) {
#if 0
  TestReduceDynamic< long, Kokkos::Qthreads >( 0 );
  TestReduceDynamic< long, Kokkos::Qthreads >( 1000000 );
#endif
}

TEST_F(qthreads, double_reduce_dynamic) {
#if 0
  TestReduceDynamic< double, Kokkos::Qthreads >( 0 );
  TestReduceDynamic< double, Kokkos::Qthreads >( 1000000 );
#endif
}

TEST_F(qthreads, long_reduce_dynamic_view) {
#if 0
  TestReduceDynamicView< long, Kokkos::Qthreads >( 0 );
  TestReduceDynamicView< long, Kokkos::Qthreads >( 1000000 );
#endif
}

TEST_F(qthreads, scan) {
#if 0
  TestScan< Kokkos::Qthreads >::test_range( 1, 1000 );
  TestScan< Kokkos::Qthreads >( 0 );
  TestScan< Kokkos::Qthreads >( 100000 );
  TestScan< Kokkos::Qthreads >( 10000000 );
  Kokkos::Qthreads().fence();
#endif
}

TEST_F(qthreads, scan_small) {
#if 0
  typedef TestScan< Kokkos::Qthreads, Kokkos::Impl::QthreadsExecUseScanSmall > TestScanFunctor;

  for ( int i = 0; i < 1000; ++i ) {
    TestScanFunctor( 10 );
    TestScanFunctor( 10000 );
  }
  TestScanFunctor( 1000000 );
  TestScanFunctor( 10000000 );

  Kokkos::Qthreads().fence();
#endif
}

TEST_F(qthreads, team_scan) {
#if 0
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 0 );
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 0 );
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 10 );
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 10 );
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 10000 );
  TestScanTeam< Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 10000 );
#endif
}

TEST_F(qthreads, team_long_reduce) {
#if 0
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 0 );
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 0 );
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 3 );
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 3 );
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 100000 );
  TestReduceTeam< long, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 100000 );
#endif
}

TEST_F(qthreads, team_double_reduce) {
#if 0
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 0 );
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 0 );
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 3 );
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 3 );
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Static> >( 100000 );
  TestReduceTeam< double, Kokkos::Qthreads, Kokkos::Schedule<Kokkos::Dynamic> >( 100000 );
#endif
}

TEST_F(qthreads, reduction_deduction) {
#if 0
  TestCXX11::test_reduction_deduction< Kokkos::Qthreads >();
#endif
}

}  // namespace Test
