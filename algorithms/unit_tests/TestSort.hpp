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

#ifndef KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP
#define KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP

#include <gtest/gtest.h>
#include<Kokkos_Core.hpp>
#include<Kokkos_DynamicView.hpp>
#include<Kokkos_Random.hpp>
#include<Kokkos_Sort.hpp>

namespace Test {

namespace Impl{

template<class ExecutionSpace, class Scalar>
struct is_sorted_struct {
  typedef unsigned int value_type;
  typedef ExecutionSpace execution_space;

  Kokkos::View<Scalar*,ExecutionSpace> keys;

  is_sorted_struct(Kokkos::View<Scalar*,ExecutionSpace> keys_):keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator() (int i, unsigned int& count) const {
    if(keys(i)>keys(i+1)) count++;
  }
};

template<class ExecutionSpace, class Scalar>
struct sum {
  typedef double value_type;
  typedef ExecutionSpace execution_space;

  Kokkos::View<Scalar*,ExecutionSpace> keys;

  sum(Kokkos::View<Scalar*,ExecutionSpace> keys_):keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator() (int i, double& count) const {
    count+=keys(i);
  }
};

template<class ExecutionSpace, class Scalar>
struct bin3d_is_sorted_struct {
  typedef unsigned int value_type;
  typedef ExecutionSpace execution_space;

  Kokkos::View<Scalar*[3],ExecutionSpace> keys;

  int max_bins;
  Scalar min;
  Scalar max;

  bin3d_is_sorted_struct(Kokkos::View<Scalar*[3],ExecutionSpace> keys_,int max_bins_,Scalar min_,Scalar max_):
    keys(keys_),max_bins(max_bins_),min(min_),max(max_) {
  }
  KOKKOS_INLINE_FUNCTION
  void operator() (int i, unsigned int& count) const {
    int ix1 = int ((keys(i,0)-min)/max * max_bins);
    int iy1 = int ((keys(i,1)-min)/max * max_bins);
    int iz1 = int ((keys(i,2)-min)/max * max_bins);
    int ix2 = int ((keys(i+1,0)-min)/max * max_bins);
    int iy2 = int ((keys(i+1,1)-min)/max * max_bins);
    int iz2 = int ((keys(i+1,2)-min)/max * max_bins);

    if (ix1>ix2)  count++;
    else if(ix1==ix2) {
      if (iy1>iy2)  count++;
      else if ((iy1==iy2) && (iz1>iz2))  count++;
    }
  }
};

template<class ExecutionSpace, class Scalar>
struct sum3D {
  typedef double value_type;
  typedef ExecutionSpace execution_space;

  Kokkos::View<Scalar*[3],ExecutionSpace> keys;

  sum3D(Kokkos::View<Scalar*[3],ExecutionSpace> keys_):keys(keys_) {}
  KOKKOS_INLINE_FUNCTION
  void operator() (int i, double& count) const {
    count+=keys(i,0);
    count+=keys(i,1);
    count+=keys(i,2);
  }
};

template<class ExecutionSpace, typename KeyType>
void test_1D_sort(unsigned int n,bool force_kokkos) {
  typedef Kokkos::View<KeyType*,ExecutionSpace> KeyViewType;
  KeyViewType keys("Keys",n);

  // Test sorting array with all numbers equal
  Kokkos::deep_copy(keys,KeyType(1));
  Kokkos::sort(keys,force_kokkos);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys,g,Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type::MAX_URAND);

  double sum_before = 0.0;
  double sum_after = 0.0;
  unsigned int sort_fails = 0;

  Kokkos::parallel_reduce(n,sum<ExecutionSpace, KeyType>(keys),sum_before);

  Kokkos::sort(keys,force_kokkos);

  Kokkos::parallel_reduce(n,sum<ExecutionSpace, KeyType>(keys),sum_after);
  Kokkos::parallel_reduce(n-1,is_sorted_struct<ExecutionSpace, KeyType>(keys),sort_fails);

  double ratio = sum_before/sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum = (ratio > (1.0-epsilon)) && (ratio < (1.0+epsilon)) ? 1 : 0;

  ASSERT_EQ(sort_fails,0);
  ASSERT_EQ(equal_sum,1);
}

template<class ExecutionSpace, typename KeyType>
void test_3D_sort(unsigned int n) {
  typedef Kokkos::View<KeyType*[3],ExecutionSpace > KeyViewType;

  KeyViewType keys("Keys",n*n*n);

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys,g,100.0);

  double sum_before = 0.0;
  double sum_after = 0.0;
  unsigned int sort_fails = 0;

  Kokkos::parallel_reduce(keys.dimension_0(),sum3D<ExecutionSpace, KeyType>(keys),sum_before);

  int bin_1d = 1;
  while( bin_1d*bin_1d*bin_1d*4< (int) keys.dimension_0() ) bin_1d*=2;
  int bin_max[3] = {bin_1d,bin_1d,bin_1d};
  typename KeyViewType::value_type min[3] = {0,0,0};
  typename KeyViewType::value_type max[3] = {100,100,100};

  typedef Kokkos::BinOp3D< KeyViewType > BinOp;
  BinOp bin_op(bin_max,min,max);
  Kokkos::BinSort< KeyViewType , BinOp >
    Sorter(keys,bin_op,false);
  Sorter.create_permute_vector();
  Sorter.template sort< KeyViewType >(keys);

  Kokkos::parallel_reduce(keys.dimension_0(),sum3D<ExecutionSpace, KeyType>(keys),sum_after);
  Kokkos::parallel_reduce(keys.dimension_0()-1,bin3d_is_sorted_struct<ExecutionSpace, KeyType>(keys,bin_1d,min[0],max[0]),sort_fails);

  double ratio = sum_before/sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum = (ratio > (1.0-epsilon)) && (ratio < (1.0+epsilon)) ? 1 : 0;

  if ( sort_fails )
    printf("3D Sort Sum: %f %f Fails: %u\n",sum_before,sum_after,sort_fails);

  ASSERT_EQ(sort_fails,0);
  ASSERT_EQ(equal_sum,1);
}

//----------------------------------------------------------------------------

template<class ExecutionSpace, typename KeyType>
void test_dynamic_view_sort(unsigned int n )
{
  typedef typename ExecutionSpace::memory_space memory_space ;
  typedef Kokkos::Experimental::DynamicView<KeyType*,ExecutionSpace> KeyDynamicViewType;
  typedef Kokkos::View<KeyType*,ExecutionSpace> KeyViewType;

  const size_t upper_bound = 2 * n ;

  const size_t total_alloc_size = n * sizeof(KeyType) * 1.2 ;
  const size_t superblock_size  = std::min(total_alloc_size, size_t(1000000));

  typename KeyDynamicViewType::memory_pool
    pool( memory_space()
        , n * sizeof(KeyType) * 1.2
        ,     500 /* min block size in bytes */
        ,   30000 /* max block size in bytes */
        , superblock_size
        );

  KeyDynamicViewType keys("Keys",pool,upper_bound);

  keys.resize_serial(n);

  KeyViewType keys_view("KeysTmp", n );

  // Test sorting array with all numbers equal
  Kokkos::deep_copy(keys_view,KeyType(1));
  Kokkos::deep_copy(keys,keys_view);
  Kokkos::sort(keys, 0 /* begin */ , n /* end */ );

  Kokkos::Random_XorShift64_Pool<ExecutionSpace> g(1931);
  Kokkos::fill_random(keys_view,g,Kokkos::Random_XorShift64_Pool<ExecutionSpace>::generator_type::MAX_URAND);

  Kokkos::deep_copy(keys,keys_view);

  double sum_before = 0.0;
  double sum_after = 0.0;
  unsigned int sort_fails = 0;

  Kokkos::parallel_reduce(n,sum<ExecutionSpace, KeyType>(keys_view),sum_before);

  Kokkos::sort(keys, 0 /* begin */ , n /* end */ );

  Kokkos::deep_copy( keys_view , keys );

  Kokkos::parallel_reduce(n,sum<ExecutionSpace, KeyType>(keys_view),sum_after);
  Kokkos::parallel_reduce(n-1,is_sorted_struct<ExecutionSpace, KeyType>(keys_view),sort_fails);

  double ratio = sum_before/sum_after;
  double epsilon = 1e-10;
  unsigned int equal_sum = (ratio > (1.0-epsilon)) && (ratio < (1.0+epsilon)) ? 1 : 0;

  if ( sort_fails != 0 || equal_sum != 1 ) {
    std::cout << " N = " << n
              << " ; sum_before = " << sum_before
              << " ; sum_after = " << sum_after
              << " ; ratio = " << ratio
              << std::endl ;
  }

  ASSERT_EQ(sort_fails,0);
  ASSERT_EQ(equal_sum,1);
}

//----------------------------------------------------------------------------

template<class ExecutionSpace>
void test_issue_1160()
{
  Kokkos::View<int*, ExecutionSpace> element_("element", 10);
  Kokkos::View<double*, ExecutionSpace> x_("x", 10);
  Kokkos::View<double*, ExecutionSpace> v_("y", 10);

  auto h_element = Kokkos::create_mirror_view(element_);
  auto h_x = Kokkos::create_mirror_view(x_);
  auto h_v = Kokkos::create_mirror_view(v_);

  h_element(0) = 142;
  h_element(1) = 141;
  h_element(2) = 140;
  h_element(3) = 5;
  h_element(4) = 4;
  h_element(5) = 3;
  h_element(6) = 2;
  h_element(7) = 1;
  h_element(8) = 0;
  h_element(9) = -1;
  h_element(10) = -2;

  for (int i = 0; i < 10; ++i) {
    h_v(i, 0) = h_x(i, 0) = double(h_element(i));
  }
  Kokkos::deep_copy(element_, h_element);
  Kokkos::deep_copy(x_, h_x);
  Kokkos::deep_copy(v_, h_v);

  typedef Kokkos::View<LocalOrdinal*, ExecutionSpace> KeyViewType;
  typedef Kokkos::BinOp1D< KeyViewType > BinOp;

  int begin = 3;
  int end = 8;
  BinOp1D binner(end - begin, begin, end - 1);

  Kokkos::BinSort<decltype(element_) , BinOp > Sorter(element_,begin,end,binner,false);
  Sorter.create_permute_vector();
  Sorter.sort(element_,begin,end);

  Sorter.sort(x_,begin,end);
  Sorter.sort(v_,begin,end);

  Kokkos::deep_copy(h_element, element_);
  Kokkos::deep_copy(h_x, x_);
  Kokkos::deep_copy(h_v, v_);

  ASSERT_EQ(h_element(0), 142);
  ASSERT_EQ(h_element(1), 141);
  ASSERT_EQ(h_element(2), 140);
  ASSERT_EQ(h_element(3), 1);
  ASSERT_EQ(h_element(4), 2);
  ASSERT_EQ(h_element(5), 3);
  ASSERT_EQ(h_element(6), 4);
  ASSERT_EQ(h_element(7), 5);
  ASSERT_EQ(h_element(8), 0);
  ASSERT_EQ(h_element(9), -1);
  ASSERT_EQ(h_element(10), -2);

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(h_element(i), int(h_x(i, 0)));
    ASSERT_EQ(h_element(i), int(h_v(i, 0)));
  }
}

//----------------------------------------------------------------------------

template<class ExecutionSpace, typename KeyType>
void test_sort(unsigned int N)
{
  test_1D_sort<ExecutionSpace,KeyType>(N*N*N, true);
  test_1D_sort<ExecutionSpace,KeyType>(N*N*N, false);
#if !defined(KOKKOS_ENABLE_ROCM)
  test_3D_sort<ExecutionSpace,KeyType>(N);
  test_dynamic_view_sort<ExecutionSpace,KeyType>(N*N);
#endif
  test_issue_1160<ExecutionSpace>();
}

}
}
#endif /* KOKKOS_ALGORITHMS_UNITTESTS_TESTSORT_HPP */
