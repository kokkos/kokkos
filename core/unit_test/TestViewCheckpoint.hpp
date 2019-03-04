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

template < typename ExecSpace, typename CpFileSpace >
struct TestCheckPointView {
   static bool consistency_check() {
      return true;
   }

   static void test_view_chkpt( int dim0, int dim1 ) {
       int N = 1;
       typedef Kokkos::LayoutLeft       Layout;
       typedef Kokkos::HostSpace        defaultMemSpace;  // default device
       typedef CpFileSpace              fileSystemSpace;  // file system

       fileSystemSpace::set_default_path("/home/jsmiles/Development/Data");
       fileSystemSpace fs;
       typedef Kokkos::View<double**, Layout, defaultMemSpace> local_view_type;

       local_view_type A("view_A", dim0, dim1);
       local_view_type B("view_B", dim0, dim1);
       local_view_type::HostMirror h_A = Kokkos::create_mirror_view(A);
       local_view_type::HostMirror h_B = Kokkos::create_mirror_view(B);

       auto F_A = Kokkos::create_chkpt_mirror(fs, h_A);
       auto F_B = Kokkos::create_chkpt_mirror(fs, h_B);

       fileSystemSpace::restore_all_views();  // restart from existing…
       for ( int i = 0; i < N; i++ ) {
          Kokkos::deep_copy(A, h_A);  Kokkos::deep_copy(B, h_B);

          Kokkos::parallel_for (dim0, KOKKOS_LAMBDA(const int i) {
              for (int j=0; j< dim1; j++) {
                 A(i,j) = i*j;  B(i,j) = i*j*2;
              }
          });
          Kokkos::deep_copy(h_A, A);  Kokkos::deep_copy(h_B, B);  

          if (!consistency_check()) {
             fileSystemSpace::restore_view("view_A"); // restore data 
             fileSystemSpace::restore_view("view_B"); // restore data 
          } else {
             fileSystemSpace::checkpoint_views();  // save result
          }
       }
    }

};

template < typename ExecSpace, typename CpFileSpace >
struct TestFSDeepCopy {


   static void test_view_chkpt(std::string file_name, int dim0, int dim1) {

      typedef Kokkos::View<double**,Kokkos::HostSpace> Rank2ViewType;
      Rank2ViewType view_2;
      view_2 = Rank2ViewType("memory_view_2", dim0, dim1);

      typedef CpFileSpace cp_file_space_type;
      Kokkos::View<double**,cp_file_space_type> cp_view(file_name, dim0, dim1);

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

TEST_F( TEST_CATEGORY , view_checkpoint_hdf5 ) {
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::HDF5Space >::test_view_chkpt("/home/jsmiles/Development/cp_view.hdf",10,10);
  remove("/home/jsmiles/Development/cp_view.hdf");
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::HDF5Space >::test_view_chkpt("/home/jsmiles/Development/cp_view.hdf",100,100);
  remove("/home/jsmiles/Development/cp_view.hdf");
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::HDF5Space >::test_view_chkpt("/home/jsmiles/Development/cp_view.hdf",10000,10000);
  remove("/home/jsmiles/Development/cp_view.hdf");

  TestCheckPointView< TEST_EXECSPACE, Kokkos::Experimental::HDF5Space >::test_view_chkpt(10,10);
}

TEST_F( TEST_CATEGORY , view_checkpoint_sio ) {
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::StdFileSpace >::test_view_chkpt("/home/jsmiles/Development/cp_view.bin",10,10);
  remove("/home/jsmiles/Development/cp_view.bin");
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::StdFileSpace >::test_view_chkpt("/home/jsmiles/Development/cp_view.bin",100,100);
  remove("/home/jsmiles/Development/cp_view.bin");
  TestFSDeepCopy< TEST_EXECSPACE, Kokkos::Experimental::StdFileSpace >::test_view_chkpt("/home/jsmiles/Development/cp_view.bin",10000,10000);
  remove("/home/jsmiles/Development/cp_view.bin");
  TestCheckPointView< TEST_EXECSPACE, Kokkos::Experimental::StdFileSpace >::test_view_chkpt(10,10);
}

#endif
} // namespace Test
