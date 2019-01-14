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

namespace Test {

namespace {

#ifdef KOKKOS_ENABLE_HDF5

template < typename ExecSpace >
struct TestCheckPointView {


   static void test_view_chkpt(int dim0, int dim1) {

      typedef Kokkos::View<double**,Kokkos::HostSpace> Rank2ViewType;
      Rank2ViewType view_2;
      view_2 = Rank2ViewType("memory_view_2", dim0, dim1);

      typedef Kokkos::Experimental::HDF5Space hdf5_space_type;
      Kokkos::View<double**,hdf5_space_type> cp_view("/home/jsmiles/Development/cpview2", dim0, dim1);

      for (int i = 0; i < dim0; i++) {
         for (int j = 0; j < dim1; j++) {
            view_2(i,j) = i + j;
         }
      }

      // host_space to ExecSpace
      Kokkos::deep_copy( cp_view, view_2 );
      Kokkos::fence();

      for (int i = 0; i < dim0; i++) {
         for (int j = 0; j < dim1; j++) {
             view_2(i,j) = 0;
         }
      }

      // ExecSpace to host_space 
      Kokkos::deep_copy( view_2, cp_view );
      Kokkos::fence();

      for (int i = 0; i < dim0; i++) {
         for (int j = 0; j < dim1; j++) {
            ASSERT_EQ(view_2(i,j), i + j);
         }
      }

   }

};

#endif


} // namespace

#ifdef KOKKOS_ENABLE_HDF5

TEST_F( TEST_CATEGORY , view_checkpoint_tests ) {
  TestCheckPointView< TEST_EXECSPACE >::test_view_chkpt(10,10);
  //remove("/home/jsmiles/Development/cpview2");
  TestCheckPointView< TEST_EXECSPACE >::test_view_chkpt(100,100);
  //remove("/home/jsmiles/Development/cpview2");
}

#endif
} // namespace Test
