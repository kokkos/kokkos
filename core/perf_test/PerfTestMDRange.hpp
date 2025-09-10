// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <iostream>

namespace Test {
template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct MultiDimRangePerf3D {
  using execution_space = DeviceType;
  using size_type       = typename execution_space::size_type;

  using iterate_type = Kokkos::Iterate;

  using view_type      = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;
  using host_view_type = typename view_type::host_mirror_type;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  MultiDimRangePerf3D(const view_type &A_, const view_type &B_,
                      const long &irange_, const long &jrange_,
                      const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long i, const long j, const long k) const {
    A(i, j, k) =
        0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                            B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                            B(i, j, k));
  }

  static double test_multi_index(const unsigned int icount,
                                 const unsigned int jcount,
                                 const unsigned int kcount,
                                 const unsigned int Ti = 1,
                                 const unsigned int Tj = 1,
                                 const unsigned int Tk = 1,
                                 const long iter       = 1) {
    // This test performs multidim range over all dims
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount + 2, jcount + 2, kcount + 2);
    using FunctorType =
        MultiDimRangePerf3D<execution_space, ScalarType, TestLayout>;

    double dt_min = 0;

    Kokkos::deep_copy(Atest, 1.0);
    execution_space().fence();
    Kokkos::deep_copy(Btest, 1.0);
    execution_space().fence();

    // LayoutRight
    if (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      using MDRangeType = typename Kokkos::MDRangePolicy<
          Kokkos::Rank<3, iterate_type::Right, iterate_type::Right>,
          execution_space>;
      using tile_type  = typename MDRangeType::tile_type;
      using point_type = typename MDRangeType::point_type;

      Kokkos::MDRangePolicy<
          Kokkos::Rank<3, iterate_type::Right, iterate_type::Right>,
          execution_space>
          policy(point_type{{0, 0, 0}}, point_type{{icount, jcount, kcount}},
                 tile_type{{Ti, Tj, Tk}});

      for (int i = 0; i < iter; ++i) {
        Kokkos::Timer timer;
        Kokkos::parallel_for(policy,
                             FunctorType(Atest, Btest, icount, jcount, kcount));
        execution_space().fence();
        const double dt = timer.seconds();
        if (0 == i)
          dt_min = dt;
        else
          dt_min = dt < dt_min ? dt : dt_min;

        // Correctness check - only the first run
        if (0 == i) {
          long numErrors = 0;
          host_view_type Ahost("Ahost", icount, jcount, kcount);
          Kokkos::deep_copy(Ahost, Atest);
          host_view_type Bhost("Bhost", icount + 2, jcount + 2, kcount + 2);
          Kokkos::deep_copy(Bhost, Btest);

          // On KNL, this may vectorize - add print statement to prevent
          // Also, compare against epsilon, as vectorization can change bitwise
          // answer
          for (long l = 0; l < static_cast<long>(icount); ++l) {
            for (long j = 0; j < static_cast<long>(jcount); ++j) {
              for (long k = 0; k < static_cast<long>(kcount); ++k) {
                ScalarType check =
                    0.25 *
                    (ScalarType)(Bhost(l + 2, j, k) + Bhost(l + 1, j, k) +
                                 Bhost(l, j + 2, k) + Bhost(l, j + 1, k) +
                                 Bhost(l, j, k + 2) + Bhost(l, j, k + 1) +
                                 Bhost(l, j, k));
                if (Ahost(l, j, k) - check != 0) {
                  ++numErrors;
                  std::cout << "  Correctness error at index: " << l << "," << j
                            << "," << k << "\n"
                            << "  multi Ahost = " << Ahost(l, j, k)
                            << "  expected = " << check
                            << "  multi Bhost(ijk) = " << Bhost(l, j, k)
                            << "  multi Bhost(l+1jk) = " << Bhost(l + 1, j, k)
                            << "  multi Bhost(l+2jk) = " << Bhost(l + 2, j, k)
                            << "  multi Bhost(ij+1k) = " << Bhost(l, j + 1, k)
                            << "  multi Bhost(ij+2k) = " << Bhost(l, j + 2, k)
                            << "  multi Bhost(ijk+1) = " << Bhost(l, j, k + 1)
                            << "  multi Bhost(ijk+2) = " << Bhost(l, j, k + 2)
                            << std::endl;
                  // exit(-1);
                }
              }
            }
          }
          if (numErrors != 0) {
            std::cout << "LR multi: errors " << numErrors << "  range product "
                      << icount * jcount * kcount << "  LL " << jcount * kcount
                      << "  LR " << icount * jcount << std::endl;
          }
          // else { std::cout << " multi: No errors!" <<  std::endl; }
        }
      }  // end for

    }
    // LayoutLeft
    else {
      Kokkos::MDRangePolicy<
          Kokkos::Rank<3, iterate_type::Left, iterate_type::Left>,
          execution_space>
          policy({{0, 0, 0}}, {{icount, jcount, kcount}}, {{Ti, Tj, Tk}});

      for (int i = 0; i < iter; ++i) {
        Kokkos::Timer timer;
        Kokkos::parallel_for(policy,
                             FunctorType(Atest, Btest, icount, jcount, kcount));
        execution_space().fence();
        const double dt = timer.seconds();
        if (0 == i)
          dt_min = dt;
        else
          dt_min = dt < dt_min ? dt : dt_min;

        // Correctness check - only the first run
        if (0 == i) {
          long numErrors = 0;
          host_view_type Ahost("Ahost", icount, jcount, kcount);
          Kokkos::deep_copy(Ahost, Atest);
          host_view_type Bhost("Bhost", icount + 2, jcount + 2, kcount + 2);
          Kokkos::deep_copy(Bhost, Btest);

          // On KNL, this may vectorize - add print statement to prevent
          // Also, compare against epsilon, as vectorization can change bitwise
          // answer
          for (long l = 0; l < static_cast<long>(icount); ++l) {
            for (long j = 0; j < static_cast<long>(jcount); ++j) {
              for (long k = 0; k < static_cast<long>(kcount); ++k) {
                ScalarType check =
                    0.25 *
                    (ScalarType)(Bhost(l + 2, j, k) + Bhost(l + 1, j, k) +
                                 Bhost(l, j + 2, k) + Bhost(l, j + 1, k) +
                                 Bhost(l, j, k + 2) + Bhost(l, j, k + 1) +
                                 Bhost(l, j, k));
                if (Ahost(l, j, k) - check != 0) {
                  ++numErrors;
                  std::cout << "  Correctness error at index: " << l << "," << j
                            << "," << k << "\n"
                            << "  multi Ahost = " << Ahost(l, j, k)
                            << "  expected = " << check
                            << "  multi Bhost(ijk) = " << Bhost(l, j, k)
                            << "  multi Bhost(l+1jk) = " << Bhost(l + 1, j, k)
                            << "  multi Bhost(l+2jk) = " << Bhost(l + 2, j, k)
                            << "  multi Bhost(ij+1k) = " << Bhost(l, j + 1, k)
                            << "  multi Bhost(ij+2k) = " << Bhost(l, j + 2, k)
                            << "  multi Bhost(ijk+1) = " << Bhost(l, j, k + 1)
                            << "  multi Bhost(ijk+2) = " << Bhost(l, j, k + 2)
                            << std::endl;
                  // exit(-1);
                }
              }
            }
          }
          if (numErrors != 0) {
            std::cout << " LL multi run: errors " << numErrors
                      << "  range product " << icount * jcount * kcount
                      << "  LL " << jcount * kcount << "  LR "
                      << icount * jcount << std::endl;
          }
          // else { std::cout << " multi: No errors!" <<  std::endl; }
        }
      }  // end for
    }

    return dt_min;
  }
};

template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct RangePolicyCollapseTwo {
  // RangePolicy for 3D range, but will collapse only 2 dims => like Rank<2> for
  // multi-dim; unroll 2 dims in one-dim

  using execution_space = DeviceType;
  using size_type       = typename execution_space::size_type;
  using layout          = TestLayout;

  using iterate_type = Kokkos::Iterate;

  using view_type      = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;
  using host_view_type = typename view_type::host_mirror_type;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  RangePolicyCollapseTwo(view_type &A_, const view_type &B_,
                         const long &irange_, const long &jrange_,
                         const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long r) const {
    if (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      // id(i,j,k) = k + j*Nk + i*Nk*Nj = k + Nk*(j + i*Nj) = k + Nk*r
      // r = j + i*Nj
      long i = int(r / jrange);
      long j = int(r - i * jrange);
      for (int k = 0; k < krange; ++k) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    } else if (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      // id(i,j,k) = i + j*Ni + k*Ni*Nj = i + Ni*(j + k*Nj) = i + Ni*r
      // r = j + k*Nj
      long k = int(r / jrange);
      long j = int(r - k * jrange);
      for (int i = 0; i < irange; ++i) {
        A(i, j, k) =
            0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) +
                                B(i, j + 2, k) + B(i, j + 1, k) +
                                B(i, j, k + 2) + B(i, j, k + 1) + B(i, j, k));
      }
    }
  }

  static double test_index_collapse_two(const unsigned int icount,
                                        const unsigned int jcount,
                                        const unsigned int kcount,
                                        const long iter = 1) {
    // This test refers to collapsing two dims while using the RangePolicy
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount + 2, jcount + 2, kcount + 2);
    using FunctorType =
        RangePolicyCollapseTwo<execution_space, ScalarType, TestLayout>;

    long collapse_index_rangeA = 0;
    long collapse_index_rangeB = 0;
    if (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      collapse_index_rangeA = static_cast<long>(icount) * jcount;
      collapse_index_rangeB = static_cast<long>(icount + 2) * (jcount + 2);
      //      std::cout << "   LayoutRight " << std::endl;
    } else if (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      collapse_index_rangeA = static_cast<long>(kcount) * jcount;
      collapse_index_rangeB = static_cast<long>(kcount + 2) * (jcount + 2);
      //      std::cout << "   LayoutLeft " << std::endl;
    } else {
      std::cout << "  LayoutRight or LayoutLeft required - will pass 0 as "
                   "range instead "
                << std::endl;
      exit(-1);
    }

    Kokkos::RangePolicy<execution_space> policy(0, (collapse_index_rangeA));

    double dt_min = 0;

    Kokkos::deep_copy(Atest, 1.0);
    execution_space().fence();
    Kokkos::deep_copy(Btest, 1.0);
    execution_space().fence();

    for (int i = 0; i < iter; ++i) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(policy,
                           FunctorType(Atest, Btest, icount, jcount, kcount));
      execution_space().fence();
      const double dt = timer.seconds();
      if (0 == i)
        dt_min = dt;
      else
        dt_min = dt < dt_min ? dt : dt_min;

      // Correctness check - first iteration only
      if (0 == i) {
        long numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount + 2, jcount + 2, kcount + 2);
        Kokkos::deep_copy(Bhost, Btest);

        // On KNL, this may vectorize - add print statement to prevent
        // Also, compare against epsilon, as vectorization can change bitwise
        // answer
        for (long l = 0; l < static_cast<long>(icount); ++l) {
          for (long j = 0; j < static_cast<long>(jcount); ++j) {
            for (long k = 0; k < static_cast<long>(kcount); ++k) {
              ScalarType check =
                  0.25 * (ScalarType)(Bhost(l + 2, j, k) + Bhost(l + 1, j, k) +
                                      Bhost(l, j + 2, k) + Bhost(l, j + 1, k) +
                                      Bhost(l, j, k + 2) + Bhost(l, j, k + 1) +
                                      Bhost(l, j, k));
              if (Ahost(l, j, k) - check != 0) {
                ++numErrors;
                std::cout << "  Correctness error at index: " << l << "," << j
                          << "," << k << "\n"
                          << "  flat Ahost = " << Ahost(l, j, k)
                          << "  expected = " << check << std::endl;
                // exit(-1);
              }
            }
          }
        }
        if (numErrors != 0) {
          std::cout << " RP collapse2: errors " << numErrors
                    << "  range product " << icount * jcount * kcount << "  LL "
                    << jcount * kcount << "  LR " << icount * jcount
                    << std::endl;
        }
        // else { std::cout << " RP collapse2: Pass! " << std::endl; }
      }
    }

    return dt_min;
  }
};

template <class DeviceType, typename ScalarType = double,
          typename TestLayout = Kokkos::LayoutRight>
struct RangePolicyCollapseAll {
  // RangePolicy for 3D range, but will collapse all dims

  using execution_space = DeviceType;
  using size_type       = typename execution_space::size_type;
  using layout          = TestLayout;

  using view_type      = Kokkos::View<ScalarType ***, TestLayout, DeviceType>;
  using host_view_type = typename view_type::host_mirror_type;

  view_type A;
  view_type B;
  const long irange;
  const long jrange;
  const long krange;

  RangePolicyCollapseAll(view_type &A_, const view_type &B_,
                         const long &irange_, const long &jrange_,
                         const long &krange_)
      : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const long r) const {
    if (std::is_same_v<TestLayout, Kokkos::LayoutRight>) {
      long i = int(r / (jrange * krange));
      long j = int((r - i * jrange * krange) / krange);
      long k = int(r - i * jrange * krange - j * krange);
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    } else if (std::is_same_v<TestLayout, Kokkos::LayoutLeft>) {
      long k = int(r / (irange * jrange));
      long j = int((r - k * irange * jrange) / irange);
      long i = int(r - k * irange * jrange - j * irange);
      A(i, j, k) =
          0.25 * (ScalarType)(B(i + 2, j, k) + B(i + 1, j, k) + B(i, j + 2, k) +
                              B(i, j + 1, k) + B(i, j, k + 2) + B(i, j, k + 1) +
                              B(i, j, k));
    }
  }

  static double test_collapse_all(const unsigned int icount,
                                  const unsigned int jcount,
                                  const unsigned int kcount,
                                  const long iter = 1) {
    // This test refers to collapsing all dims using the RangePolicy
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount + 2, jcount + 2, kcount + 2);
    using FunctorType =
        RangePolicyCollapseAll<execution_space, ScalarType, TestLayout>;

    const long flat_index_range = icount * static_cast<long>(jcount) * kcount;
    Kokkos::RangePolicy<execution_space> policy(0, flat_index_range);

    double dt_min = 0;

    Kokkos::deep_copy(Atest, 1.0);
    execution_space().fence();
    Kokkos::deep_copy(Btest, 1.0);
    execution_space().fence();

    for (int i = 0; i < iter; ++i) {
      Kokkos::Timer timer;
      Kokkos::parallel_for(policy,
                           FunctorType(Atest, Btest, icount, jcount, kcount));
      execution_space().fence();
      const double dt = timer.seconds();
      if (0 == i)
        dt_min = dt;
      else
        dt_min = dt < dt_min ? dt : dt_min;

      // Correctness check - first iteration only
      if (0 == i) {
        long numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount + 2, jcount + 2, kcount + 2);
        Kokkos::deep_copy(Bhost, Btest);

        // On KNL, this may vectorize - add print statement to prevent
        // Also, compare against epsilon, as vectorization can change bitwise
        // answer
        for (long l = 0; l < static_cast<long>(icount); ++l) {
          for (long j = 0; j < static_cast<long>(jcount); ++j) {
            for (long k = 0; k < static_cast<long>(kcount); ++k) {
              ScalarType check =
                  0.25 * (ScalarType)(Bhost(l + 2, j, k) + Bhost(l + 1, j, k) +
                                      Bhost(l, j + 2, k) + Bhost(l, j + 1, k) +
                                      Bhost(l, j, k + 2) + Bhost(l, j, k + 1) +
                                      Bhost(l, j, k));
              if (Ahost(l, j, k) - check != 0) {
                ++numErrors;
                std::cout << "  Callapse ALL Correctness error at index: " << l
                          << "," << j << "," << k << "\n"
                          << "  flat Ahost = " << Ahost(l, j, k)
                          << "  expected = " << check << std::endl;
                // exit(-1);
              }
            }
          }
        }
        if (numErrors != 0) {
          std::cout << " RP collapse all: errors " << numErrors
                    << "  range product " << icount * jcount * kcount << "  LL "
                    << jcount * kcount << "  LR " << icount * jcount
                    << std::endl;
        }
        // else { std::cout << " RP collapse all: Pass! " << std::endl; }
      }
    }

    return dt_min;
  }
};

}  // end namespace Test
