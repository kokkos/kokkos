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

#include <gtest/gtest.h>

#include <sstream>
#include <iostream>

#include <Kokkos_Core.hpp>

namespace Test {

template <class Space>
struct TestViewMappingSubview {
  using ExecSpace = typename Space::execution_space;
  using MemSpace  = typename Space::memory_space;

  using range = Kokkos::pair<int, int>;

  enum { AN = 10 };
  using AT  = Kokkos::View<int*, ExecSpace>;
  using ACT = Kokkos::View<const int*, ExecSpace>;
  using AS  = Kokkos::Subview<AT, range>;

  enum { BN0 = 10, BN1 = 11, BN2 = 12 };
  using BT = Kokkos::View<int***, ExecSpace>;
  using BS = Kokkos::Subview<BT, range, range, range>;

  enum { CN0 = 10, CN1 = 11, CN2 = 12 };
  using CT = Kokkos::View<int** * [13][14], ExecSpace>;
  // changing CS to CTS here because when compiling with nvshmem, there is a
  // define for CS that makes this fail...
  using CTS = Kokkos::Subview<CT, range, range, range, int, int>;

  enum { DN0 = 10, DN1 = 11, DN2 = 12, DN3 = 13, DN4 = 14 };
  using DT = Kokkos::View<int** * [DN3][DN4], ExecSpace>;
  using DS = Kokkos::Subview<DT, int, range, range, range, int>;

  using DLT  = Kokkos::View<int** * [13][14], Kokkos::LayoutLeft, ExecSpace>;
  using DLS1 = Kokkos::Subview<DLT, range, int, int, int, int>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(
      DLS1::rank == 1 &&
          std::is_same<typename DLS1::array_layout, Kokkos::LayoutLeft>::value,
      "Subview layout error for rank 1 subview of left-most range of "
      "LayoutLeft");
#endif

  using DRT  = Kokkos::View<int** * [13][14], Kokkos::LayoutRight, ExecSpace>;
  using DRS1 = Kokkos::Subview<DRT, int, int, int, int, range>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  static_assert(
      DRS1::rank == 1 &&
          std::is_same<typename DRS1::array_layout, Kokkos::LayoutRight>::value,
      "Subview layout error for rank 1 subview of right-most range of "
      "LayoutRight");
#endif

  AT Aa;
  AS Ab;
  ACT Ac;
  BT Ba;
  BS Bb;
  CT Ca;
  CTS Cb;
  DT Da;
  DS Db;

  TestViewMappingSubview()
      : Aa("Aa", AN),
        Ab(Kokkos::subview(Aa, std::pair<int, int>(1, AN - 1))),
        Ac(Aa, std::pair<int, int>(1, AN - 1)),
        Ba("Ba", BN0, BN1, BN2),
        Bb(Kokkos::subview(Ba, std::pair<int, int>(1, BN0 - 1),
                           std::pair<int, int>(1, BN1 - 1),
                           std::pair<int, int>(1, BN2 - 1))),
        Ca("Ca", CN0, CN1, CN2),
        Cb(Kokkos::subview(Ca, std::pair<int, int>(1, CN0 - 1),
                           std::pair<int, int>(1, CN1 - 1),
                           std::pair<int, int>(1, CN2 - 1), 1, 2)),
        Da("Da", DN0, DN1, DN2),
        Db(Kokkos::subview(Da, 1, std::pair<int, int>(1, DN1 - 1),
                           std::pair<int, int>(1, DN2 - 1),
                           std::pair<int, int>(1, DN3 - 1), 2)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int, long& error_count) const {
    auto Ad = Kokkos::subview(Aa, Kokkos::pair<int, int>(1, AN - 1));

    for (int i = 1; i < AN - 1; ++i)
      if (&Aa[i] != &Ab[i - 1]) ++error_count;
    for (int i = 1; i < AN - 1; ++i)
      if (&Aa[i] != &Ac[i - 1]) ++error_count;
    for (int i = 1; i < AN - 1; ++i)
      if (&Aa[i] != &Ad[i - 1]) ++error_count;

    for (int i2 = 1; i2 < BN2 - 1; ++i2)
      for (int i1 = 1; i1 < BN1 - 1; ++i1)
        for (int i0 = 1; i0 < BN0 - 1; ++i0) {
          if (&Ba(i0, i1, i2) != &Bb(i0 - 1, i1 - 1, i2 - 1)) ++error_count;
        }

    for (int i2 = 1; i2 < CN2 - 1; ++i2)
      for (int i1 = 1; i1 < CN1 - 1; ++i1)
        for (int i0 = 1; i0 < CN0 - 1; ++i0) {
          if (&Ca(i0, i1, i2, 1, 2) != &Cb(i0 - 1, i1 - 1, i2 - 1))
            ++error_count;
        }

    for (int i2 = 1; i2 < DN3 - 1; ++i2)
      for (int i1 = 1; i1 < DN2 - 1; ++i1)
        for (int i0 = 1; i0 < DN1 - 1; ++i0) {
          if (&Da(1, i0, i1, i2, 2) != &Db(i0 - 1, i1 - 1, i2 - 1))
            ++error_count;
        }
  }

  void run() {
    TestViewMappingSubview<ExecSpace> self;

    ASSERT_EQ(Aa.extent(0), AN);
    ASSERT_EQ(Ab.extent(0), (size_t)AN - 2);
    ASSERT_EQ(Ac.extent(0), (size_t)AN - 2);
    ASSERT_EQ(Ba.extent(0), BN0);
    ASSERT_EQ(Ba.extent(1), BN1);
    ASSERT_EQ(Ba.extent(2), BN2);
    ASSERT_EQ(Bb.extent(0), (size_t)BN0 - 2);
    ASSERT_EQ(Bb.extent(1), (size_t)BN1 - 2);
    ASSERT_EQ(Bb.extent(2), (size_t)BN2 - 2);

    ASSERT_EQ(Ca.extent(0), CN0);
    ASSERT_EQ(Ca.extent(1), CN1);
    ASSERT_EQ(Ca.extent(2), CN2);
    ASSERT_EQ(Ca.extent(3), (size_t)13);
    ASSERT_EQ(Ca.extent(4), (size_t)14);
    ASSERT_EQ(Cb.extent(0), (size_t)CN0 - 2);
    ASSERT_EQ(Cb.extent(1), (size_t)CN1 - 2);
    ASSERT_EQ(Cb.extent(2), (size_t)CN2 - 2);

    ASSERT_EQ(Da.extent(0), DN0);
    ASSERT_EQ(Da.extent(1), DN1);
    ASSERT_EQ(Da.extent(2), DN2);
    ASSERT_EQ(Da.extent(3), DN3);
    ASSERT_EQ(Da.extent(4), DN4);

    ASSERT_EQ(Db.extent(0), (size_t)DN1 - 2);
    ASSERT_EQ(Db.extent(1), (size_t)DN2 - 2);
    ASSERT_EQ(Db.extent(2), (size_t)DN3 - 2);

    ASSERT_EQ(Da.stride_1(), Db.stride_0());
    ASSERT_EQ(Da.stride_2(), Db.stride_1());
    ASSERT_EQ(Da.stride_3(), Db.stride_2());

    long error_count = -1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace>(0, 1), *this,
                            error_count);

    ASSERT_EQ(error_count, 0);
  }
};

TEST(TEST_CATEGORY, view_mapping_subview) {
  TestViewMappingSubview<TEST_EXECSPACE> f;
  f.run();
}

}  // namespace Test
