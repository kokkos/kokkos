//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include <mdspan/mdarray.hpp>
#include <vector>

#include <gtest/gtest.h>
#include "offload_utils.hpp"

#ifdef __cpp_lib_memory_resource
#include <memory_resource>


//For testing, prints allocs and deallocs to cout
struct ChatterResource : std::pmr::memory_resource{
  ChatterResource() = default;
  ChatterResource(std::pmr::memory_resource* upstream): upstream(upstream){}
  ChatterResource(const ChatterResource&) = delete;
  ChatterResource(ChatterResource&&) = delete;
  ChatterResource& operator=(const ChatterResource&) = delete;
  ChatterResource& operator=(ChatterResource&&) = delete;

  private:

    void* do_allocate( std::size_t bytes, std::size_t alignment ) override{
        std::cout << "Allocation - size: " << bytes << ", alignment: " << alignment << std::endl;
        return upstream->allocate(bytes, alignment);
    }

    void do_deallocate( void* p, std::size_t bytes, std::size_t alignment ) override{
        std::cout << "Deallocation - size: " << bytes << ", alignment: " << alignment << std::endl;
        upstream->deallocate(p, bytes, alignment);
    }

    bool do_is_equal( const std::pmr::memory_resource& other ) const noexcept override{
        return this == &other;
    }

    std::pmr::memory_resource* upstream = std::pmr::get_default_resource();
};
#endif

namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dyn = Kokkos::dynamic_extent;

template<int Rank>
struct mdarray_values;

template<>
struct mdarray_values<0> {
  template<class MDA>
  static void check(const MDA& m) {
    ASSERT_EQ(MDSPAN_IMPL_OP(m), 42);
  }
  template<class pointer, class extents_type>
  static void fill(const pointer& ptr, const extents_type&, bool) {
    ptr[0] = 42;
  }
};

template<>
struct mdarray_values<1> {
  template<class MDA>
  static void check(const MDA& m) {
    using index_type = typename MDA::index_type;
    for(index_type i=0; i<m.extent(0); i++)
      ASSERT_EQ(MDSPAN_IMPL_OP(m,i), 42 + i);
  }
  template<class pointer, class extents_type>
  static void fill(const pointer& ptr, const extents_type& ext, bool) {
    using index_type = typename extents_type::index_type;
    for(index_type i=0; i<ext.extent(0); i++)
      ptr[i] = 42 + i;
  }
};

template<>
struct mdarray_values<2> {
  template<class MDA>
  static void check(const MDA& m) {
    using index_type = typename MDA::index_type;
    for(index_type i=0; i<m.extent(0); i++)
      for(index_type j=0; j<m.extent(1); j++) {
        auto tmp = MDSPAN_IMPL_OP(m,i,j);
        ASSERT_EQ(tmp, 42 + i*1000 + j);
      }
  }
  template<class pointer, class extents_type>
  static void fill(const pointer& ptr, const extents_type& ext, bool is_layout_right) {
    using index_type = typename extents_type::index_type;
    using value_type = std::remove_pointer_t<pointer>;
    for(index_type i=0; i<ext.extent(0); i++)
      for(index_type j=0; j<ext.extent(1); j++)
        if(is_layout_right)
          ptr[i*ext.extent(1)+j] = static_cast<value_type>(42 + i*1000 + j);
        else
          ptr[i+j*ext.extent(0)] = static_cast<value_type>(42 + i*1000 + j);
  }
};

template<>
struct mdarray_values<3> {
  template<class MDA>
  static void check(const MDA& m) {
    for(int i=0; i<m.extent(0); i++)
      for(int j=0; j<m.extent(1); j++)
        for(int k=0; k<m.extent(2); k++) {
          auto tmp = MDSPAN_IMPL_OP(m,i,j,k);
          ASSERT_EQ(tmp, 42 + i*1000000 + j*1000 + k);
        }
  }
  template<class pointer, class extents_type>
  static void fill(const pointer& ptr, const extents_type& ext, bool is_layout_right) {
    for(int i=0; i<ext.extent(0); i++)
      for(int j=0; j<ext.extent(1); j++)
        for(int k=0; k<ext.extent(2); k++)
          if(is_layout_right)
            ptr[(i*ext.extent(1)+j)*ext.extent(2)+k] = 42 + i*1000000 + j*1000+k;
          else
            ptr[i+(j+k*ext.extent(1))*ext.extent(0)] = 42 + i*1000000 + j*1000+k;
  }
};

template<class MDA>
void check_correctness(MDA& m, size_t rank, size_t rank_dynamic,
                       size_t extent_0, size_t extent_1, size_t extent_2,
                       size_t stride_0, size_t stride_1, size_t stride_2,
                       typename MDA::pointer ptr, bool ptr_matches,
                       bool exhaustive) {
  ASSERT_EQ(m.rank(), rank);
  ASSERT_EQ(m.rank_dynamic(), rank_dynamic);
#if MDSPAN_HAS_CXX_20
  if constexpr (MDA::extents_type::rank()>0) {
#endif
    if(rank>0) {
      ASSERT_EQ(m.extent(0), extent_0);
      ASSERT_EQ(m.stride(0), stride_0);
    }
    if(rank>1) {
      ASSERT_EQ(m.extent(1), extent_1);
      ASSERT_EQ(m.stride(1), stride_1);
    }
    if(rank>2) {
      ASSERT_EQ(m.extent(2), extent_2);
      ASSERT_EQ(m.stride(2), stride_2);
    }
#if MDSPAN_HAS_CXX_20
  }
#endif
  if(ptr_matches)
    ASSERT_EQ(m.data(),ptr);
  else
    ASSERT_NE(m.data(),ptr);
  ASSERT_EQ(m.is_exhaustive(),exhaustive);
  mdarray_values<MDA::rank()>::check(m);
}

void test_mdarray_ctor_data_carray() {
  size_t* errors = allocate_array<size_t>(1);
  errors[0] = 0;

  dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
    KokkosEx::mdarray<int, Kokkos::extents<size_t,1>, Kokkos::layout_right, std::array<int, 1>> m(Kokkos::extents<int,1>{});
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.rank(), 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.rank_dynamic(), 0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.extent(0), 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.static_extent(0), 1);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.stride(0), 1);
    m.data()[0] = {42};
    auto val = MDSPAN_IMPL_OP(m,0);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(val, 42);
    MDSPAN_IMPL_DEVICE_ASSERT_EQ(m.is_exhaustive(), true);
  });
  ASSERT_EQ(errors[0], 0);
  free_array(errors);
}

TEST(TestMdarrayCtorDataCArray, test_mdarray_ctor_data_carray) {
  MDSPAN_IMPL_TESTS_RUN_TEST(test_mdarray_ctor_data_carray())
}

// Construct from extents only
TEST(TestMdarrayCtorFromExtents, 0d_static) {
  KokkosEx::mdarray<int, Kokkos::extents<int>, Kokkos::layout_right, std::array<int,1>> m(Kokkos::extents<int>{});
  // ptr to fill, extents, is_layout_right
  mdarray_values<0>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, false, true);
}

// Construct from sizes only
TEST(TestMdarrayCtorFromSizes, 1d_static) {
  KokkosEx::mdarray<int, Kokkos::extents<int,1>, Kokkos::layout_right, std::array<int,1>> m(1);
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 0, 1, 0, 0, 1, 0, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizes, 2d_static) {
  KokkosEx::mdarray<int, Kokkos::extents<int,2,3>, Kokkos::layout_right, std::array<int,6>> m(2,3);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 0, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizes, 1d_dynamic) {
  KokkosEx::mdarray<int, Kokkos::dextents<int,1>, Kokkos::layout_right, std::array<int,1>> m(1);
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizes, 2d_dynamic) {
  KokkosEx::mdarray<int, Kokkos::dextents<size_t,2>> m(2,3);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizes, 2d_mixed) {
  KokkosEx::mdarray<int, Kokkos::extents<unsigned,2,Kokkos::dynamic_extent>> m(3);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

// Construct from container + sizes
TEST(TestMdarrayCtorFromContainerSizes, 1d_static) {
  std::array<int, 1> d{42};
  using mda_t = KokkosEx::mdarray<int, Kokkos::extents<unsigned,1>, Kokkos::layout_right, std::array<int,1>>;
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(d.data(),Kokkos::extents<unsigned,1>(),true);
  mda_t m(Kokkos::extents<unsigned, 1>{}, d);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 0, 1, 0, 0, 1, 0, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_static) {
  std::array<int, 6> d{42,43,44,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,3>, Kokkos::layout_right, std::array<int,6>> m(Kokkos::extents<int, 2,3>{},d);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 0, 2, 3, 0, 3, 1, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizes, 1d_dynamic) {
  std::vector<int> d{42};
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(d.data(),Kokkos::extents<int, 1>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 1>> m(Kokkos::extents<int, 1>{},d);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_dynamic) {
  std::vector<int> d{42,1,2,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(Kokkos::extents<int, 2,3>{},d);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizes, 2d_mixed) {
  std::vector<int> d{42,1,2,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,Kokkos::dynamic_extent>> m(Kokkos::extents<int, 2,3>{},d);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, d.data(), false, true);
}

// Construct from move container + sizes
TEST(TestMdarrayCtorFromMoveContainerSizes, 1d_static) {
  std::array<int, 1> d{42};
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(d.data(),Kokkos::extents<int, 1>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 1>, Kokkos::layout_right, std::array<int,1>> m(Kokkos::extents<int, 1>{},std::move(d));
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 0, 1, 0, 0, 1, 0, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_static) {
  std::array<int, 6> d{42,1,2,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,3>, Kokkos::layout_right, std::array<int,6>> m(Kokkos::extents<int, 2,3>{},std::move(d));
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 0, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 1d_dynamic) {
  std::vector<int> d{42};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(ptr,Kokkos::extents<int, 1>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 1>> m(Kokkos::extents<int, 1>{},std::move(d));
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, ptr, true, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_dynamic) {
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(ptr,Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(Kokkos::extents<int, 2,3>{},std::move(d));
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, ptr, true, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizes, 2d_mixed) {
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(ptr,Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,Kokkos::dynamic_extent>> m(Kokkos::extents<int, 2,3>{},std::move(d));
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, ptr, true, true);
}

// Construct from extents only
TEST(TestMdarrayCtorFromExtentsAlloc, 0d_static) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::extents<unsigned>> m(Kokkos::extents<unsigned>{},alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<0>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, false, true);
}

// Construct from sizes only
TEST(TestMdarrayCtorFromSizesAlloc, 1d_static) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::extents<int, 1>> m(Kokkos::extents<int, 1>(), alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 0, 1, 0, 0, 1, 0, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizesAlloc, 2d_static) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,3>> m(Kokkos::extents<int, 2,3>(), alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 0, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizesAlloc, 1d_dynamic) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::dextents<int, 1>> m(Kokkos::extents<int, 1>(), alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizesAlloc, 2d_dynamic) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(Kokkos::extents<int, 2,3>(), alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromSizesAlloc, 2d_mixed) {
  std::allocator<int> alloc;
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,Kokkos::dynamic_extent>> m(Kokkos::extents<int, 2,Kokkos::dynamic_extent>{3}, alloc);
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(m.data(),m.extents(),true);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, nullptr, false, true);
}

TEST(TestMdarrayCtorFromContainerSizesAlloc, 1d_dynamic) {
  std::allocator<int> alloc;
  std::vector<int> d{42};
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(d.data(),Kokkos::extents<int, 1>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 1>> m(Kokkos::dextents<int, 1>{1}, d, alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizesAlloc, 2d_dynamic) {
  std::allocator<int> alloc;
  std::vector<int> d{42,1,2,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(Kokkos::dextents<int, 2>{2,3}, d, alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromContainerSizesAlloc, 2d_mixed) {
  std::allocator<int> alloc;
  std::vector<int> d{42,1,2,3,4,41};
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(d.data(),Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,Kokkos::dynamic_extent>> m(Kokkos::extents<int, 2,Kokkos::dynamic_extent>{3}, d, alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, d.data(), false, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizesAlloc, 1d_dynamic) {
  std::allocator<int> alloc;
  std::vector<int> d{42};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<1>::fill(ptr,Kokkos::extents<int, 1>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 1>> m(Kokkos::extents<int, 1>(), std::move(d), alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 1, 1, 1, 0, 0, 1, 0, 0, ptr, true, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizesAlloc, 2d_dynamic) {
  std::allocator<int> alloc;
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(ptr,Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(Kokkos::extents<int, 2,3>(), std::move(d), alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 2, 2, 3, 0, 3, 1, 0, ptr, true, true);
}

TEST(TestMdarrayCtorFromMoveContainerSizesAlloc, 2d_mixed) {
  std::allocator<int> alloc;
  std::vector<int> d{42,1,2,3,4,41};
  auto ptr = d.data();
  // ptr to fill, extents, is_layout_right
  mdarray_values<2>::fill(ptr,Kokkos::extents<int, 2,3>(),true);
  KokkosEx::mdarray<int, Kokkos::extents<int, 2,Kokkos::dynamic_extent>> m(Kokkos::extents<int, 2,Kokkos::dynamic_extent>(3), std::move(d), alloc);
  // mdarray, rank, rank_dynamic, ext0, ext1, ext2, stride0, stride1, stride2, ptr, ptr_matches, exhaustive
  check_correctness(m, 2, 1, 2, 3, 0, 3, 1, 0, ptr, true, true);
}
// PMR


#ifdef __cpp_lib_memory_resource
TEST(TestMdarrayCtorWithPMR, 2d_mixed) {
    using array_2d_pmr_dynamic = KokkosEx::mdarray<int, Kokkos::dextents<int, 2>, Kokkos::layout_right, std::vector<int, std::pmr::polymorphic_allocator<int>>>;

    ChatterResource allocation_logger;
    constexpr bool test = std::uses_allocator_v<array_2d_pmr_dynamic, std::pmr::polymorphic_allocator<int>>;
    (void) test;

    array_2d_pmr_dynamic a{Kokkos::dextents<int, 2>{3,3}, &allocation_logger};
    array_2d_pmr_dynamic b{3,3};

    std::pmr::vector<array_2d_pmr_dynamic> top_container{&allocation_logger};
    top_container.reserve(4);

    top_container.emplace_back(3,3);
    top_container.emplace_back(a.mapping());
    top_container.emplace_back(a.mapping(), a.container());
    top_container.push_back({a});
}
#endif

// Construct from container only
TEST(TestMdarrayCtorDataStdArray, test_mdarray_ctor_data_carray) {
  std::array<int, 1> d = {42};
  KokkosEx::mdarray<int, Kokkos::extents<int, 1>, Kokkos::layout_right, std::array<int, 1>> m(Kokkos::extents<int, 1>{}, d);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(MDSPAN_IMPL_OP(m, 0), 42);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCtorDataVector, test_mdarray_ctor_data_carray) {
  std::vector<int> d = {42};
  KokkosEx::mdarray<int, Kokkos::extents<int, 1>, Kokkos::layout_right, std::vector<int>> m(Kokkos::extents<int, 1>{}, d);
  ASSERT_EQ(m.rank(), 1);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_EQ(m.extent(0), 1);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(MDSPAN_IMPL_OP(m, 0), 42);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCtorExtentsStdArrayConvertibleToSizeT, test_mdarray_ctor_extents_std_array_convertible_to_size_t) {
  std::vector<int> d{42, 17, 71, 24};
  std::array<int, 2> e{2, 2};
  KokkosEx::mdarray<int, Kokkos::dextents<int, 2>> m(e, d);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 2);
  ASSERT_EQ(m.extent(1), 2);
  ASSERT_EQ(m.stride(0), 2);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
}


TEST(TestMdarrayListInitializationLayoutLeft, test_mdarray_list_initialization_layout_left) {
  std::vector<int> d(16*32);
  auto ptr = d.data();
  KokkosEx::mdarray<int, Kokkos::extents<int, dyn, dyn>, Kokkos::layout_left> m{Kokkos::dextents<int, 2>{16, 32}, std::move(d)};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 16);
  ASSERT_TRUE(m.is_exhaustive());
}


TEST(TestMdarrayListInitializationLayoutRight, test_mdarray_list_initialization_layout_right) {
  std::vector<int> d(16*32);
  auto ptr = d.data();
  KokkosEx::mdarray<int, Kokkos::extents<int, dyn, dyn>, Kokkos::layout_right> m{Kokkos::dextents<int, 2>{16, 32}, std::move(d)};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 32);
  ASSERT_EQ(m.stride(1), 1);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayListInitializationLayoutStride, test_mdarray_list_initialization_layout_stride) {
  std::vector<int> d(32*128);
  auto ptr = d.data();
  KokkosEx::mdarray<int, Kokkos::extents<int, dyn, dyn>, Kokkos::layout_stride> m{{Kokkos::dextents<int, 2>{16, 32}, std::array<std::size_t, 2>{1, 128}}, std::move(d)};
  ASSERT_EQ(m.data(), ptr);
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 16);
  ASSERT_EQ(m.extent(1), 32);
  ASSERT_EQ(m.stride(0), 1);
  ASSERT_EQ(m.stride(1), 128);
  ASSERT_FALSE(m.is_exhaustive());
}

#if 0

#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestMdarrayCTAD, extents_pack) {
  std::array<int, 1> d{42};
  KokkosEx::mdarray m(d.data(), 64, 128);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCTAD, ctad_pointer) {
  std::array<int,5> d = {1,2,3,4,5};
  KokkosEx::mdarray m(d.data());
  static_assert(std::is_same<decltype(m)::element_type,int>::value);
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCTAD, ctad_carray) {
  int data[5] = {1,2,3,4,5};
  KokkosEx::mdarray m(data);
  static_assert(std::is_same<decltype(m)::element_type,int>::value);
  ASSERT_EQ(m.data(), &data[0]);
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_TRUE(m.is_exhaustive());


  KokkosEx::mdarray m2(data, 3);
  static_assert(std::is_same<decltype(m2)::element_type,int>::value);
  ASSERT_EQ(m2.data(), &data[0]);
  ASSERT_EQ(m2.rank(), 1);
  ASSERT_EQ(m2.rank_dynamic(), 1);
  ASSERT_EQ(m2.extent(0), 3);
  ASSERT_TRUE(m2.is_exhaustive());
  ASSERT_EQ(MDSPAN_IMPL_OP(m2, 2), 3);
}

TEST(TestMdarrayCTAD, ctad_const_carray) {
  const int data[5] = {1,2,3,4,5};
  KokkosEx::mdarray m(data);
  static_assert(std::is_same<decltype(m)::element_type,const int>::value);
  ASSERT_EQ(m.data(), &data[0]);
  ASSERT_EQ(m.rank(), 0);
  ASSERT_EQ(m.rank_dynamic(), 0);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCTAD, extents_object) {
  std::array<int, 1> d{42};
  KokkosEx::mdarray m{d.data(), Kokkos::extents{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCTAD, extents_std_array) {
  std::array<int, 1> d{42};
  KokkosEx::mdarray m{d.data(), std::array{64, 128}};
  ASSERT_EQ(m.data(), d.data());
  ASSERT_EQ(m.rank(), 2);
  ASSERT_EQ(m.rank_dynamic(), 2);
  ASSERT_EQ(m.extent(0), 64);
  ASSERT_EQ(m.extent(1), 128);
  ASSERT_TRUE(m.is_exhaustive());
}

TEST(TestMdarrayCTAD, layout_left) {
  std::array<int, 1> d{42};

  KokkosEx::mdarray m0{d.data(), Kokkos::layout_left::mapping{Kokkos::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 16);
  ASSERT_TRUE(m0.is_exhaustive());

// TODO: Perhaps one day I'll get this to work.
/*
  KokkosEx::mdarray m1{d.data(), Kokkos::layout_left::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 16);
  ASSERT_TRUE(m1.is_exhaustive());
*/
}

TEST(TestMdarrayCTAD, layout_right) {
  std::array<int, 1> d{42};

  KokkosEx::mdarray m0{d.data(), Kokkos::layout_right::mapping{Kokkos::extents{16, 32}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 32);
  ASSERT_EQ(m0.stride(1), 1);
  ASSERT_TRUE(m0.is_exhaustive());

// TODO: Perhaps one day I'll get this to work.
/*
  KokkosEx::mdarray m1{d.data(), Kokkos::layout_right::mapping{{16, 32}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 32);
  ASSERT_EQ(m1.stride(1), 1);
  ASSERT_TRUE(m1.is_exhaustive());
*/
}

TEST(TestMdarrayCTAD, layout_stride) {
  std::array<int, 1> d{42};

  KokkosEx::mdarray m0{d.data(), Kokkos::layout_stride::mapping{Kokkos::extents{16, 32}, std::array{1, 128}}};
  ASSERT_EQ(m0.data(), d.data());
  ASSERT_EQ(m0.rank(), 2);
  ASSERT_EQ(m0.rank_dynamic(), 2);
  ASSERT_EQ(m0.extent(0), 16);
  ASSERT_EQ(m0.extent(1), 32);
  ASSERT_EQ(m0.stride(0), 1);
  ASSERT_EQ(m0.stride(1), 128);
  ASSERT_FALSE(m0.is_exhaustive());

  /*
  KokkosEx::mdarray m1{d.data(), Kokkos::layout_stride::mapping{Kokkos::extents{16, 32}, Kokkos::extents{1, 128}}};
  ASSERT_EQ(m1.data(), d.data());
  ASSERT_EQ(m1.rank(), 2);
  ASSERT_EQ(m1.rank_dynamic(), 2);
  ASSERT_EQ(m1.extent(0), 16);
  ASSERT_EQ(m1.extent(1), 32);
  ASSERT_EQ(m1.stride(0), 1);
  ASSERT_EQ(m1.stride(1), 128);
  ASSERT_FALSE(m1.is_exhaustive());
  */

// TODO: Perhaps one day I'll get this to work.
/*
  KokkosEx::mdarray m2{d.data(), Kokkos::layout_stride::mapping{{16, 32}, {1, 128}}};
  ASSERT_EQ(m2.data(), d.data());
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 16);
  ASSERT_EQ(m2.extent(1), 32);
  ASSERT_EQ(m2.stride(0), 1);
  ASSERT_EQ(m2.stride(1), 128);
  ASSERT_FALSE(m2.is_exhaustive());
*/
}
#endif

#endif
