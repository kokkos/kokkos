
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

namespace stdex = std::experimental;
TEST(defaultdevicetype, development_test) {
  /*
  template <std::size_t Idx, std::ptrdiff_t Extent, class Strides, class Exts,
            class Idxs, bool LayoutLeftBased>
  struct stride_storage_common;
  template <std::size_t Idx, std::ptrdiff_t Extent,
            std::ptrdiff_t... StaticStrides, std::ptrdiff_t... Exts,
            std::size_t... Idxs, bool LayoutLeftBased>
  struct stride_storage_common<
      Idx, Extent, std::integer_sequence<std::ptrdiff_t, StaticStrides...>,
      std::experimental::extents<Exts...>,
      std::integer_sequence<std::size_t, Idxs...>, LayoutLeftBased> {
  */
  /*
 template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
           std::ptrdiff_t Idx, class StaticStrides, class Extents, class
 IdxPack, bool LayoutLeftDerived, class Enable = void> struct
 stride_storage_impl;*/
  /*
  // Integer compile-time constant stride case
  template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
            std::ptrdiff_t Idx, class Strides, class Extents, class Idxs,
            bool LayoutLeftBased>
  struct stride_storage_impl<
      StaticStride, Extent, Idx, Strides, Extents, Idxs, LayoutLeftBased,
      std::enable_if_t<(StaticStride > 0) &&
                       StaticStride != std::experimental::dynamic_extent>>
      : stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                              LayoutLeftBased> {
                              */
  /*
  template <std::ptrdiff_t StaticStride, std::ptrdiff_t Extent,
            std::ptrdiff_t Idx, class Strides, class Extents, class Idxs,
            bool LayoutLeftBased>
  struct stride_storage_impl<
      StaticStride, Extent, Idx, Strides, Extents, Idxs, LayoutLeftBased,
      std::enable_if_t<is_static_dimension_stride<StaticStride>>>
      : stride_storage_common<Idx, Extent, Strides, Extents, Idxs,
                              LayoutLeftBased> {
  */
  // using dyn = std::experimental::dynamic_extent;
  // Kokkos::Impl::MDSpanLayoutForLayoutRightImpl<std::experimental::extents<-1,-1,-1>>
  // foo; Kokkos::Impl::layout_stride_general_impl<false,
  // std::integer_sequence<long, 1, -1, -9223372036854775807>,
  // std::experimental::extents<-1, -1, -1>, std::integer_sequence<unsigned long,
  // 0, 1, 2>> bar; Kokkos::Impl::layout_stride_general_impl<false,
  // std::integer_sequence<long, -9223372036854775807, -1, 1>,
  // std::experimental::extents<-1, -1, -1>, std::integer_sequence<unsigned long,
  // 0, 1, 2>> bar; Kokkos::Impl::layout_stride_general_impl<false,
  // std::integer_sequence<long, 1, -1, 1>, std::experimental::extents<-1, -1,
  // -1>, std::integer_sequence<unsigned long, 0, 1, 2>> bar;

  std::array<ptrdiff_t, 1> array{1};
  std::experimental::extents<stdex::dynamic_extent> ext(array);
  {
    Kokkos::BasicView<int ***, Kokkos::LayoutRight,
                      Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>
        foo("A", 7, 3, 5);

    printf("%p %p %p %p %p\n", foo.data(), &foo(0, 0, 1), &foo(0, 1, 0),
           &foo(1, 0, 0), &foo(3, 2, 3));
    printf("%p %li %li %li %li\n", foo.data(),
           ptrdiff_t(&foo(0, 2, 3) - foo.data()),
           ptrdiff_t(&foo(0, 1, 0) - foo.data()),
           ptrdiff_t(&foo(1, 0, 0) - foo.data()),
           ptrdiff_t(&foo(3, 2, 3) - foo.data()));
    printf("UseCount: %i\n", foo.use_count());
  }
  {
    // Kokkos::BasicView<int ***, Kokkos::LayoutRight,
    // Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>> foo_v("A",7,3,5);
    Kokkos::BasicView<int ***, std::experimental::layout_right,
                      Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>
        foo("A", 7, 3, 5);
    // auto foo_sub = subspan(foo.get_mdspan(),1,stdex::full_extent,
    // stdex::full_extent);
    // std::experimental::mdspan<int,std::experimental::extents<-1,-1>,std::experimental::layout_stride<-1,-1>,std::experimental::accessor_basic<int>>
    // foo_sub2 = foo_sub;

    // printf("%s\n",typeid(decltype(foo_sub)).name());
    Kokkos::BasicView<int **, std::experimental::layout_stride,
                      Kokkos::HostSpace>
        foo_sub1(foo, stdex::full_extent, stdex::full_extent, 3);
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 3; j++)
        if (&foo_sub1(i, j) != &foo(i, j, 3)) printf("Error A %i %i\n", i, j);
    // auto foo_sub = subspan(foo.get_mdspan(),3,stdex::full_extent,
    // stdex::full_extent);
    Kokkos::BasicView<int **, std::experimental::layout_right,
                      Kokkos::HostSpace>
        foo_sub2(foo, 3, stdex::full_extent, stdex::full_extent);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 5; j++)
        if (&foo_sub2(i, j) != &foo(3, i, j)) printf("Error B %i %i\n", i, j);

    auto foo_sub3 =
        Kokkos::subview(foo, stdex::full_extent, stdex::full_extent, 3);
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 3; j++)
        if (&foo_sub3(i, j) != &foo(i, j, 3)) printf("Error A %i %i\n", i, j);

    auto foo_sub4 =
        Kokkos::subview(foo, 3, stdex::full_extent, stdex::full_extent);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 5; j++)
        if (&foo_sub4(i, j) != &foo(3, i, j)) printf("Error B %i %i\n", i, j);

    // std::experimental::mdspan<int,std::experimental::extents<-1,-1,-1>,std::experimental::layout_right,std::experimental::accessor_basic<int>>
    //  foo(foo_v.data(),7,3,5);
    printf("%p %p %p %p %p\n", foo.data(), &foo(0, 0, 1), &foo(0, 1, 0),
           &foo(1, 0, 0), &foo(3, 2, 3));
    printf("%p %li %li %li %li\n", foo.data(),
           ptrdiff_t(&foo(0, 2, 3) - foo.data()),
           ptrdiff_t(&foo(0, 1, 0) - foo.data()),
           ptrdiff_t(&foo(1, 0, 0) - foo.data()),
           ptrdiff_t(&foo(3, 2, 3) - foo.data()));
    printf("UseCount: %i\n", foo.use_count());
  }
  {
    Kokkos::BasicView<int ***, Kokkos::LayoutLeft,
                      Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>
        foo("A", 7, 3, 5);
    // subspan(Kokkos::LayoutLeft(), foo.get_mdspan(),5,stdex::full_extent,
    // stdex::full_extent);
    auto foo_sub1 = submdspan(Kokkos::LayoutLeft(), foo.get_mdspan(),
                            stdex::full_extent, stdex::full_extent, 3);
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 3; j++)
        if (&foo_sub1(i, j) != &foo(i, j, 3)) printf("Error A %i %i\n", i, j);
    auto foo_sub2 = submdspan(Kokkos::LayoutLeft(), foo.get_mdspan(),
                            3, stdex::full_extent, stdex::full_extent);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 5; j++)
        if (&foo_sub2(i, j) != &foo(3, i, j)) printf("Error B %i %i\n",i,j);

    auto foo_sub3 = subview(foo,
                            stdex::full_extent, stdex::full_extent, 3);
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 3; j++)
        if (&foo_sub3(i, j) != &foo(i, j, 3)) printf("Error C %i %i\n", i, j);
    auto foo_sub4 = subview(foo,
                            3, stdex::full_extent, stdex::full_extent);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 5; j++)
        if (&foo_sub4(i, j) != &foo(3, i, j)) printf("Error D %i %i\n",i,j);

    printf("Hello Here\n");
    auto foo_sub5 = submdspan(Kokkos::LayoutLeft(), foo.get_mdspan(),
                            stdex::full_extent, 2, stdex::full_extent);
    printf("Hello Here Done\n");
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 5; j++)
        if (&foo_sub5(i, j) != &foo(i, 2, j)) printf("Error E %i %i\n", i, j);
    printf("%p %p %p %p %p\n", foo.data(), &foo(1, 0, 0), &foo(0, 2, 0), &foo(0, 0, 1), &foo(3, 2, 3));
    printf("%p %p %p %p\n", foo_sub5.data(), &foo_sub5(1, 0), &foo_sub5(0, 1), &foo_sub5(3, 3));
    printf("%p %li %li %li %li\n", foo.data(),
           ptrdiff_t(&foo(1, 0, 0) - foo.data()),
           ptrdiff_t(&foo(0, 1, 0) - foo.data()),
           ptrdiff_t(&foo(0, 0, 1) - foo.data()),
           ptrdiff_t(&foo(3, 2, 3) - foo.data()));
    printf("%p %li %li %li %li\n", foo_sub5.data(),
           ptrdiff_t(&foo_sub5(1, 0) - foo_sub5.data()),
           ptrdiff_t(&foo_sub5(0, 1) - foo_sub5.data()),
           ptrdiff_t(&foo_sub5(3, 3) - foo_sub5.data()),
           ptrdiff_t(&foo(3, 2, 3) - &foo_sub5(3,3)));
  }

  Kokkos::View<int **, Kokkos::OpenMP> a("A", 5, 5);
  {
    Kokkos::View<int **, Kokkos::LayoutRight> b = a;
    printf("%i\n", int(a.use_count()));
    auto c = a;
    printf("%i\n", int(a.use_count()));
    Kokkos::View<int **, Kokkos::OpenMP> d("D", 7, 7);
    d = a;
    printf("%i\n", int(a.use_count()));
  }
  //        printf("%i\n",int(a.use_count()));
  //        Kokkos::View<int*,Kokkos::LayoutRight> c(a,2,Kokkos::ALL);
  //        using View_3D      = typename Kokkos::View<int ***, Kokkos::OpenMP>;
  //        View_3D foo;
  //            using Host_View_3D = typename View_3D::HostMirror;
  //              Host_View_3D hostDataView_3D;
  //  printf("%i\n",Kokkos::Impl::is_static_dimension_stride<-9223372036854775807>?1:0);
  // Kokkos::Impl::stride_storage_impl<-9223372036854775807, -1, 0,
  // std::integer_sequence<long, -9223372036854775807, -1, 1>,
  // std::experimental::extents<-1, -1, -1>, std::integer_sequence<unsigned long,
  // 0, 1, 2>, false> foo;
  // Kokkos::Impl::stride_storage_impl<-9223372036854775807, -1, 0,
  // std::integer_sequence<long, -9223372036854775807, -1, 1>,
  // std::experimental::extents<-1, -1, -1>, std::integer_sequence<unsigned long,
  // 0, 1, 2>, true> foo;
}

}  // namespace Test
