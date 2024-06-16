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

#include <fstream>
#include <gtest/gtest.h>
#include "Kokkos_Core.hpp"

namespace Test {

struct TestLargeArgTag {};
struct TestRealErfcxTag {};

template <class ExecSpace>
struct TestExponentialIntergral1Function {
  using ViewType     = Kokkos::View<double*, ExecSpace>;
  using HostViewType = Kokkos::View<double*, Kokkos::HostSpace>;

  ViewType d_x, d_expint;
  typename ViewType::HostMirror h_x, h_expint;
  HostViewType h_ref;

  void testit() {
    using Kokkos::fabs;
    using Kokkos::Experimental::infinity;

    d_x      = ViewType("d_x", 15);
    d_expint = ViewType("d_expint", 15);
    h_x      = Kokkos::create_mirror_view(d_x);
    h_expint = Kokkos::create_mirror_view(d_expint);
    h_ref    = HostViewType("h_ref", 15);

    // Generate test inputs
    h_x(0)  = -0.2;
    h_x(1)  = 0.0;
    h_x(2)  = 0.2;
    h_x(3)  = 0.8;
    h_x(4)  = 1.6;
    h_x(5)  = 5.1;
    h_x(6)  = 0.01;
    h_x(7)  = 0.001;
    h_x(8)  = 1.0;
    h_x(9)  = 1.001;
    h_x(10) = 1.01;
    h_x(11) = 1.1;
    h_x(12) = 7.2;
    h_x(13) = 10.3;
    h_x(14) = 15.4;
    Kokkos::deep_copy(d_x, h_x);

    // Call exponential integral function
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 15), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_expint, d_expint);

    // Reference values computed with Octave
    h_ref(0)  = -infinity<double>::value;  // x(0)=-0.2
    h_ref(1)  = infinity<double>::value;   // x(1)= 0.0
    h_ref(2)  = 1.222650544183893e+00;     // x(2) =0.2
    h_ref(3)  = 3.105965785455429e-01;     // x(3) =0.8
    h_ref(4)  = 8.630833369753976e-02;     // x(4) =1.6
    h_ref(5)  = 1.021300107861738e-03;     // x(5) =5.1
    h_ref(6)  = 4.037929576538113e+00;     // x(6) =0.01
    h_ref(7)  = 6.331539364136149e+00;     // x(7) =0.001
    h_ref(8)  = 2.193839343955205e-01;     // x(8) =1.0
    h_ref(9)  = 2.190164225274689e-01;     // x(9) =1.001
    h_ref(10) = 2.157416237944899e-01;     // x(10)=1.01
    h_ref(11) = 1.859909045360401e-01;     // x(11)=1.1
    h_ref(12) = 9.218811688716196e-05;     // x(12)=7.2
    h_ref(13) = 2.996734771597901e-06;     // x(13)=10.3
    h_ref(14) = 1.254522935050609e-08;     // x(14)=15.4

    EXPECT_EQ(h_ref(0), h_expint(0));
    EXPECT_EQ(h_ref(1), h_expint(1));
    for (int i = 2; i < 15; i++) {
      EXPECT_LE(std::abs(h_expint(i) - h_ref(i)), std::abs(h_ref(i)) * 1e-15);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_expint(i) = Kokkos::Experimental::expint1(d_x(i));
  }
};

template <class ExecSpace>
struct TestComplexErrorFunction {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;
  using DblViewType     = Kokkos::View<double*, ExecSpace>;
  using DblHostViewType = Kokkos::View<double*, Kokkos::HostSpace>;

  ViewType d_z, d_erf, d_erfcx;
  typename ViewType::HostMirror h_z, h_erf, h_erfcx;
  HostViewType h_ref_erf, h_ref_erfcx;

  DblViewType d_x, d_erfcx_dbl;
  typename DblViewType::HostMirror h_x, h_erfcx_dbl;
  DblHostViewType h_ref_erfcx_dbl;

  void testit() {
    using Kokkos::Experimental::infinity;

    d_z         = ViewType("d_z", 52);
    d_erf       = ViewType("d_erf", 52);
    d_erfcx     = ViewType("d_erfcx", 52);
    h_z         = Kokkos::create_mirror_view(d_z);
    h_erf       = Kokkos::create_mirror_view(d_erf);
    h_erfcx     = Kokkos::create_mirror_view(d_erfcx);
    h_ref_erf   = HostViewType("h_ref_erf", 52);
    h_ref_erfcx = HostViewType("h_ref_erfcx", 52);

    d_x             = DblViewType("d_x", 6);
    d_erfcx_dbl     = DblViewType("d_erfcx_dbl", 6);
    h_x             = Kokkos::create_mirror_view(d_x);
    h_erfcx_dbl     = Kokkos::create_mirror_view(d_erfcx_dbl);
    h_ref_erfcx_dbl = DblHostViewType("h_ref_erfcx_dbl", 6);

    // Generate test inputs
    // abs(z)<=2
    h_z(0)  = Kokkos::complex<double>(0.0011, 0);
    h_z(1)  = Kokkos::complex<double>(-0.0011, 0);
    h_z(2)  = Kokkos::complex<double>(1.4567, 0);
    h_z(3)  = Kokkos::complex<double>(-1.4567, 0);
    h_z(4)  = Kokkos::complex<double>(0, 0.0011);
    h_z(5)  = Kokkos::complex<double>(0, -0.0011);
    h_z(6)  = Kokkos::complex<double>(0, 1.4567);
    h_z(7)  = Kokkos::complex<double>(0, -1.4567);
    h_z(8)  = Kokkos::complex<double>(1.4567, 0.0011);
    h_z(9)  = Kokkos::complex<double>(1.4567, -0.0011);
    h_z(10) = Kokkos::complex<double>(-1.4567, 0.0011);
    h_z(11) = Kokkos::complex<double>(-1.4567, -0.0011);
    h_z(12) = Kokkos::complex<double>(1.4567, 0.5942);
    h_z(13) = Kokkos::complex<double>(1.4567, -0.5942);
    h_z(14) = Kokkos::complex<double>(-1.4567, 0.5942);
    h_z(15) = Kokkos::complex<double>(-1.4567, -0.5942);
    h_z(16) = Kokkos::complex<double>(0.0011, 0.5942);
    h_z(17) = Kokkos::complex<double>(0.0011, -0.5942);
    h_z(18) = Kokkos::complex<double>(-0.0011, 0.5942);
    h_z(19) = Kokkos::complex<double>(-0.0011, -0.5942);
    h_z(20) = Kokkos::complex<double>(0.0011, 0.0051);
    h_z(21) = Kokkos::complex<double>(0.0011, -0.0051);
    h_z(22) = Kokkos::complex<double>(-0.0011, 0.0051);
    h_z(23) = Kokkos::complex<double>(-0.0011, -0.0051);
    // abs(z)>2.0 and x>1
    h_z(24) = Kokkos::complex<double>(3.5, 0.0011);
    h_z(25) = Kokkos::complex<double>(3.5, -0.0011);
    h_z(26) = Kokkos::complex<double>(-3.5, 0.0011);
    h_z(27) = Kokkos::complex<double>(-3.5, -0.0011);
    h_z(28) = Kokkos::complex<double>(3.5, 9.7);
    h_z(29) = Kokkos::complex<double>(3.5, -9.7);
    h_z(30) = Kokkos::complex<double>(-3.5, 9.7);
    h_z(31) = Kokkos::complex<double>(-3.5, -9.7);
    h_z(32) = Kokkos::complex<double>(18.9, 9.7);
    h_z(33) = Kokkos::complex<double>(18.9, -9.7);
    h_z(34) = Kokkos::complex<double>(-18.9, 9.7);
    h_z(35) = Kokkos::complex<double>(-18.9, -9.7);
    // abs(z)>2.0 and 0<=x<=1 and abs(y)<6
    h_z(36) = Kokkos::complex<double>(0.85, 3.5);
    h_z(37) = Kokkos::complex<double>(0.85, -3.5);
    h_z(38) = Kokkos::complex<double>(-0.85, 3.5);
    h_z(39) = Kokkos::complex<double>(-0.85, -3.5);
    h_z(40) = Kokkos::complex<double>(0.0011, 3.5);
    h_z(41) = Kokkos::complex<double>(0.0011, -3.5);
    h_z(42) = Kokkos::complex<double>(-0.0011, 3.5);
    h_z(43) = Kokkos::complex<double>(-0.0011, -3.5);
    // abs(z)>2.0 and 0<=x<=1 and abs(y)>=6
    h_z(44) = Kokkos::complex<double>(0.85, 7.5);
    h_z(45) = Kokkos::complex<double>(0.85, -7.5);
    h_z(46) = Kokkos::complex<double>(-0.85, 7.5);
    h_z(47) = Kokkos::complex<double>(-0.85, -7.5);
    h_z(48) = Kokkos::complex<double>(0.85, 19.7);
    h_z(49) = Kokkos::complex<double>(0.85, -19.7);
    h_z(50) = Kokkos::complex<double>(-0.85, 19.7);
    h_z(51) = Kokkos::complex<double>(-0.85, -19.7);

    h_x(0) = -infinity<double>::value;
    h_x(1) = -1.2;
    h_x(2) = 0.0;
    h_x(3) = 1.2;
    h_x(4) = 10.5;
    h_x(5) = infinity<double>::value;

    Kokkos::deep_copy(d_z, h_z);
    Kokkos::deep_copy(d_x, h_x);

    // Call erf and erfcx functions
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 52), *this);
    Kokkos::fence();

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, TestRealErfcxTag>(0, 1),
                         *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_erf, d_erf);
    Kokkos::deep_copy(h_erfcx, d_erfcx);
    Kokkos::deep_copy(h_erfcx_dbl, d_erfcx_dbl);

    // Reference values computed with Octave
    h_ref_erf(0) = Kokkos::complex<double>(0.001241216583181022, 0);
    h_ref_erf(1) = Kokkos::complex<double>(-0.001241216583181022, 0);
    h_ref_erf(2) = Kokkos::complex<double>(0.9606095744865353, 0);
    h_ref_erf(3) = Kokkos::complex<double>(-0.9606095744865353, 0);
    h_ref_erf(4) = Kokkos::complex<double>(0, 0.001241217584429469);
    h_ref_erf(5) = Kokkos::complex<double>(0, -0.001241217584429469);
    h_ref_erf(6) = Kokkos::complex<double>(0, 4.149756424218223);
    h_ref_erf(7) = Kokkos::complex<double>(0, -4.149756424218223);
    h_ref_erf(8) =
        Kokkos::complex<double>(0.960609812745064, 0.0001486911741082233);
    h_ref_erf(9) =
        Kokkos::complex<double>(0.960609812745064, -0.0001486911741082233);
    h_ref_erf(10) =
        Kokkos::complex<double>(-0.960609812745064, 0.0001486911741082233);
    h_ref_erf(11) =
        Kokkos::complex<double>(-0.960609812745064, -0.0001486911741082233);
    h_ref_erf(12) =
        Kokkos::complex<double>(1.02408827958197, 0.04828570635603527);
    h_ref_erf(13) =
        Kokkos::complex<double>(1.02408827958197, -0.04828570635603527);
    h_ref_erf(14) =
        Kokkos::complex<double>(-1.02408827958197, 0.04828570635603527);
    h_ref_erf(15) =
        Kokkos::complex<double>(-1.02408827958197, -0.04828570635603527);
    h_ref_erf(16) =
        Kokkos::complex<double>(0.001766791817179109, 0.7585038120712589);
    h_ref_erf(17) =
        Kokkos::complex<double>(0.001766791817179109, -0.7585038120712589);
    h_ref_erf(18) =
        Kokkos::complex<double>(-0.001766791817179109, 0.7585038120712589);
    h_ref_erf(19) =
        Kokkos::complex<double>(-0.001766791817179109, -0.7585038120712589);
    h_ref_erf(20) =
        Kokkos::complex<double>(0.001241248867618165, 0.005754776682713324);
    h_ref_erf(21) =
        Kokkos::complex<double>(0.001241248867618165, -0.005754776682713324);
    h_ref_erf(22) =
        Kokkos::complex<double>(-0.001241248867618165, 0.005754776682713324);
    h_ref_erf(23) =
        Kokkos::complex<double>(-0.001241248867618165, -0.005754776682713324);
    h_ref_erf(24) =
        Kokkos::complex<double>(0.9999992569244941, 5.939313159932013e-09);
    h_ref_erf(25) =
        Kokkos::complex<double>(0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(26) =
        Kokkos::complex<double>(-0.9999992569244941, 5.939313159932013e-09);
    h_ref_erf(27) =
        Kokkos::complex<double>(-0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(28) =
        Kokkos::complex<double>(-1.915595842013002e+34, 1.228821279117683e+32);
    h_ref_erf(29) =
        Kokkos::complex<double>(-1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(30) =
        Kokkos::complex<double>(1.915595842013002e+34, 1.228821279117683e+32);
    h_ref_erf(31) =
        Kokkos::complex<double>(1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(32) = Kokkos::complex<double>(1, 5.959897539826596e-117);
    h_ref_erf(33) = Kokkos::complex<double>(1, -5.959897539826596e-117);
    h_ref_erf(34) = Kokkos::complex<double>(-1, 5.959897539826596e-117);
    h_ref_erf(35) = Kokkos::complex<double>(-1, -5.959897539826596e-117);
    h_ref_erf(36) =
        Kokkos::complex<double>(-9211.077162784413, 13667.93825589455);
    h_ref_erf(37) =
        Kokkos::complex<double>(-9211.077162784413, -13667.93825589455);
    h_ref_erf(38) =
        Kokkos::complex<double>(9211.077162784413, 13667.93825589455);
    h_ref_erf(39) =
        Kokkos::complex<double>(9211.077162784413, -13667.93825589455);
    h_ref_erf(40) = Kokkos::complex<double>(259.38847811225, 35281.28906479814);
    h_ref_erf(41) =
        Kokkos::complex<double>(259.38847811225, -35281.28906479814);
    h_ref_erf(42) =
        Kokkos::complex<double>(-259.38847811225, 35281.28906479814);
    h_ref_erf(43) =
        Kokkos::complex<double>(-259.38847811225, -35281.28906479814);
    h_ref_erf(44) =
        Kokkos::complex<double>(6.752085728270252e+21, 9.809477366939276e+22);
    h_ref_erf(45) =
        Kokkos::complex<double>(6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(46) =
        Kokkos::complex<double>(-6.752085728270252e+21, 9.809477366939276e+22);
    h_ref_erf(47) =
        Kokkos::complex<double>(-6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(48) =
        Kokkos::complex<double>(4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(49) =
        Kokkos::complex<double>(4.37526734926942e+166, 2.16796709605852e+166);
    h_ref_erf(50) =
        Kokkos::complex<double>(-4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(51) =
        Kokkos::complex<double>(-4.37526734926942e+166, 2.16796709605852e+166);

    h_ref_erfcx(0) = Kokkos::complex<double>(0.9987599919156778, 0);
    h_ref_erfcx(1) = Kokkos::complex<double>(1.001242428085786, 0);
    h_ref_erfcx(2) = Kokkos::complex<double>(0.3288157848563544, 0);
    h_ref_erfcx(3) = Kokkos::complex<double>(16.36639786516915, 0);
    h_ref_erfcx(4) =
        Kokkos::complex<double>(0.999998790000732, -0.001241216082557101);
    h_ref_erfcx(5) =
        Kokkos::complex<double>(0.999998790000732, 0.001241216082557101);
    h_ref_erfcx(6) =
        Kokkos::complex<double>(0.1197948131677216, -0.4971192955307743);
    h_ref_erfcx(7) =
        Kokkos::complex<double>(0.1197948131677216, 0.4971192955307743);
    h_ref_erfcx(8) =
        Kokkos::complex<double>(0.3288156873503045, -0.0001874479383970247);
    h_ref_erfcx(9) =
        Kokkos::complex<double>(0.3288156873503045, 0.0001874479383970247);
    h_ref_erfcx(10) =
        Kokkos::complex<double>(16.36629202874158, -0.05369111060785572);
    h_ref_erfcx(11) =
        Kokkos::complex<double>(16.36629202874158, 0.05369111060785572);
    h_ref_erfcx(12) =
        Kokkos::complex<double>(0.3020886508118801, -0.09424097887578842);
    h_ref_erfcx(13) =
        Kokkos::complex<double>(0.3020886508118801, 0.09424097887578842);
    h_ref_erfcx(14) =
        Kokkos::complex<double>(-2.174707722732267, -11.67259764091796);
    h_ref_erfcx(15) =
        Kokkos::complex<double>(-2.174707722732267, 11.67259764091796);
    h_ref_erfcx(16) =
        Kokkos::complex<double>(0.7019810779371267, -0.5319516793968513);
    h_ref_erfcx(17) =
        Kokkos::complex<double>(0.7019810779371267, 0.5319516793968513);
    h_ref_erfcx(18) =
        Kokkos::complex<double>(0.7030703366403597, -0.5337884198542978);
    h_ref_erfcx(19) =
        Kokkos::complex<double>(0.7030703366403597, 0.5337884198542978);
    h_ref_erfcx(20) =
        Kokkos::complex<double>(0.9987340467266177, -0.005743428170378673);
    h_ref_erfcx(21) =
        Kokkos::complex<double>(0.9987340467266177, 0.005743428170378673);
    h_ref_erfcx(22) =
        Kokkos::complex<double>(1.001216353762532, -0.005765867613873103);
    h_ref_erfcx(23) =
        Kokkos::complex<double>(1.001216353762532, 0.005765867613873103);
    h_ref_erfcx(24) =
        Kokkos::complex<double>(0.1552936427089241, -4.545593205871305e-05);
    h_ref_erfcx(25) =
        Kokkos::complex<double>(0.1552936427089241, 4.545593205871305e-05);
    h_ref_erfcx(26) =
        Kokkos::complex<double>(417949.5262869648, -3218.276197742372);
    h_ref_erfcx(27) =
        Kokkos::complex<double>(417949.5262869648, 3218.276197742372);
    h_ref_erfcx(28) =
        Kokkos::complex<double>(0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(29) =
        Kokkos::complex<double>(0.01879467905925653, 0.0515934271478583);
    h_ref_erfcx(30) =
        Kokkos::complex<double>(-0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(31) =
        Kokkos::complex<double>(-0.01879467905925653, 0.0515934271478583);
    h_ref_erfcx(32) =
        Kokkos::complex<double>(0.02362328821805, -0.01209735551897239);
    h_ref_erfcx(33) =
        Kokkos::complex<double>(0.02362328821805, 0.01209735551897239);
    h_ref_erfcx(34) = Kokkos::complex<double>(-2.304726099084567e+114,
                                              -2.942443198107089e+114);
    h_ref_erfcx(35) = Kokkos::complex<double>(-2.304726099084567e+114,
                                              2.942443198107089e+114);
    h_ref_erfcx(36) =
        Kokkos::complex<double>(0.04174017523145063, -0.1569865319886248);
    h_ref_erfcx(37) =
        Kokkos::complex<double>(0.04174017523145063, 0.1569865319886248);
    h_ref_erfcx(38) =
        Kokkos::complex<double>(-0.04172154858670504, -0.156980085534407);
    h_ref_erfcx(39) =
        Kokkos::complex<double>(-0.04172154858670504, 0.156980085534407);
    h_ref_erfcx(40) =
        Kokkos::complex<double>(6.355803055239174e-05, -0.1688298297427782);
    h_ref_erfcx(41) =
        Kokkos::complex<double>(6.355803055239174e-05, 0.1688298297427782);
    h_ref_erfcx(42) =
        Kokkos::complex<double>(-5.398806789669434e-05, -0.168829903432947);
    h_ref_erfcx(43) =
        Kokkos::complex<double>(-5.398806789669434e-05, 0.168829903432947);
    h_ref_erfcx(44) =
        Kokkos::complex<double>(0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(45) =
        Kokkos::complex<double>(0.008645103282302355, 0.07490521021566741);
    h_ref_erfcx(46) =
        Kokkos::complex<double>(-0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(47) =
        Kokkos::complex<double>(-0.008645103282302355, 0.07490521021566741);
    h_ref_erfcx(48) =
        Kokkos::complex<double>(0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(49) =
        Kokkos::complex<double>(0.001238176693606428, 0.02862247416909219);
    h_ref_erfcx(50) =
        Kokkos::complex<double>(-0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(51) =
        Kokkos::complex<double>(-0.001238176693606428, 0.02862247416909219);

    h_ref_erfcx_dbl(0) = infinity<double>::value;
    h_ref_erfcx_dbl(1) = 8.062854217063865e+00;
    h_ref_erfcx_dbl(2) = 1.0;
    h_ref_erfcx_dbl(3) = 3.785374169292397e-01;
    h_ref_erfcx_dbl(4) = 5.349189974656411e-02;
    h_ref_erfcx_dbl(5) = 0.0;

    for (int i = 0; i < 52; i++) {
      EXPECT_LE(Kokkos::abs(h_erf(i) - h_ref_erf(i)),
                Kokkos::abs(h_ref_erf(i)) * 1e-13);
    }

    for (int i = 0; i < 52; i++) {
      EXPECT_LE(Kokkos::abs(h_erfcx(i) - h_ref_erfcx(i)),
                Kokkos::abs(h_ref_erfcx(i)) * 1e-13);
    }

    EXPECT_EQ(h_erfcx_dbl(0), h_ref_erfcx_dbl(0));
    EXPECT_EQ(h_erfcx_dbl(5), h_ref_erfcx_dbl(5));
    for (int i = 1; i < 5; i++) {
      EXPECT_LE(std::abs(h_erfcx_dbl(i) - h_ref_erfcx_dbl(i)),
                std::abs(h_ref_erfcx_dbl(i)) * 1e-13);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_erf(i)   = Kokkos::Experimental::erf(d_z(i));
    d_erfcx(i) = Kokkos::Experimental::erfcx(d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestRealErfcxTag&, const int& /*i*/) const {
    d_erfcx_dbl(0) = Kokkos::Experimental::erfcx(d_x(0));
    d_erfcx_dbl(1) = Kokkos::Experimental::erfcx(d_x(1));
    d_erfcx_dbl(2) = Kokkos::Experimental::erfcx(d_x(2));
    d_erfcx_dbl(3) = Kokkos::Experimental::erfcx(d_x(3));
    d_erfcx_dbl(4) = Kokkos::Experimental::erfcx(d_x(4));
    d_erfcx_dbl(5) = Kokkos::Experimental::erfcx(d_x(5));
  }
};

template <class ExecSpace>
struct TestComplexBesselJ0Y0Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbj0, d_cby0;
  typename ViewType::HostMirror h_z, h_cbj0, h_cby0;
  HostViewType h_ref_cbj0, h_ref_cby0;

  ViewType d_z_large, d_cbj0_large, d_cby0_large;
  typename ViewType::HostMirror h_z_large, h_cbj0_large, h_cby0_large;
  HostViewType h_ref_cbj0_large, h_ref_cby0_large;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_cbj0     = ViewType("d_cbj0", N);
    d_cby0     = ViewType("d_cby0", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbj0     = Kokkos::create_mirror_view(d_cbj0);
    h_cby0     = Kokkos::create_mirror_view(d_cby0);
    h_ref_cbj0 = HostViewType("h_ref_cbj0", N);
    h_ref_cby0 = HostViewType("h_ref_cby0", N);

    // Generate test inputs
    h_z(0) = Kokkos::complex<double>(0.0, 0.0);
    // abs(z)<=25
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    // abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj0, d_cbj0);
    Kokkos::deep_copy(h_cby0, d_cby0);

    // Reference values computed with Octave
    h_ref_cbj0(0) = Kokkos::complex<double>(1.000000000000000e+00, 0);
    h_ref_cbj0(1) =
        Kokkos::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
    h_ref_cbj0(2) =
        Kokkos::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
    h_ref_cbj0(3) =
        Kokkos::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
    h_ref_cbj0(4) =
        Kokkos::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
    h_ref_cbj0(5) =
        Kokkos::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
    h_ref_cbj0(6) =
        Kokkos::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
    h_ref_cbj0(7) =
        Kokkos::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
    h_ref_cbj0(8) =
        Kokkos::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
    h_ref_cbj0(9) = Kokkos::complex<double>(-2.600519549019335e-01, 0);
    h_ref_cbj0(10) =
        Kokkos::complex<double>(-2.600519549019335e-01, +9.951051106466461e-18);
    h_ref_cbj0(11) = Kokkos::complex<double>(-1.624127813134866e-01, 0);
    h_ref_cbj0(12) =
        Kokkos::complex<double>(-1.624127813134866e-01, -1.387778780781446e-17);
    h_ref_cbj0(13) =
        Kokkos::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
    h_ref_cbj0(14) =
        Kokkos::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
    h_ref_cbj0(15) =
        Kokkos::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
    h_ref_cbj0(16) =
        Kokkos::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
    h_ref_cbj0(17) =
        Kokkos::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
    h_ref_cbj0(18) =
        Kokkos::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
    h_ref_cbj0(19) =
        Kokkos::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
    h_ref_cbj0(20) =
        Kokkos::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
    h_ref_cbj0(21) = Kokkos::complex<double>(-7.315701054899962e-02, 0);
    h_ref_cbj0(22) =
        Kokkos::complex<double>(-7.315701054899962e-02, -6.938893903907228e-18);
    h_ref_cbj0(23) = Kokkos::complex<double>(-9.147180408906189e-02, 0);
    h_ref_cbj0(24) =
        Kokkos::complex<double>(-9.147180408906189e-02, +1.387778780781446e-17);

    h_ref_cby0(0) = Kokkos::complex<double>(-infinity<double>::value, 0);
    h_ref_cby0(1) =
        Kokkos::complex<double>(1.000803196554890e+00, -1.231441609303427e+00);
    h_ref_cby0(2) =
        Kokkos::complex<double>(1.000803196554890e+00, +1.231441609303427e+00);
    h_ref_cby0(3) =
        Kokkos::complex<double>(-8.951643875605797e-01, -1.267028149911417e+00);
    h_ref_cby0(4) =
        Kokkos::complex<double>(-8.951643875605797e-01, +1.267028149911417e+00);
    h_ref_cby0(5) =
        Kokkos::complex<double>(-7.230667452992603e+02, -1.602439974000479e+03);
    h_ref_cby0(6) =
        Kokkos::complex<double>(-7.230667452992603e+02, +1.602439974000479e+03);
    h_ref_cby0(7) =
        Kokkos::complex<double>(7.230667450987011e+02, -1.602439988435912e+03);
    h_ref_cby0(8) =
        Kokkos::complex<double>(7.230667450987011e+02, +1.602439988435912e+03);
    h_ref_cby0(9) = Kokkos::complex<double>(3.768500100127903e-01, 0);
    h_ref_cby0(10) =
        Kokkos::complex<double>(3.768500100127903e-01, -5.201039098038670e-01);
    h_ref_cby0(11) = Kokkos::complex<double>(-3.598179027370283e-02, 0);
    h_ref_cby0(12) =
        Kokkos::complex<double>(-3.598179027370282e-02, -3.248255626269732e-01);
    h_ref_cby0(13) =
        Kokkos::complex<double>(1.256239642409530e+03, -1.012912186329053e+03);
    h_ref_cby0(14) =
        Kokkos::complex<double>(1.256239642409530e+03, +1.012912186329053e+03);
    h_ref_cby0(15) =
        Kokkos::complex<double>(-1.256239629882755e+03, -1.012912190698863e+03);
    h_ref_cby0(16) =
        Kokkos::complex<double>(-1.256239629882755e+03, +1.012912190698863e+03);
    h_ref_cby0(17) =
        Kokkos::complex<double>(4.338202411482646e+02, -1.040215130736213e+03);
    h_ref_cby0(18) =
        Kokkos::complex<double>(4.338202411482646e+02, +1.040215130736213e+03);
    h_ref_cby0(19) =
        Kokkos::complex<double>(-4.338202362137545e+02, -1.040215138602435e+03);
    h_ref_cby0(20) =
        Kokkos::complex<double>(-4.338202362137545e+02, +1.040215138602435e+03);
    h_ref_cby0(21) = Kokkos::complex<double>(1.318364704235323e-01, 0);
    h_ref_cby0(22) =
        Kokkos::complex<double>(1.318364704235323e-01, -1.463140210979992e-01);
    h_ref_cby0(23) = Kokkos::complex<double>(4.735895220944939e-02, 0);
    h_ref_cby0(24) =
        Kokkos::complex<double>(4.735895220944938e-02, -1.829436081781237e-01);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbj0(i) - h_ref_cbj0(i)),
                Kokkos::abs(h_ref_cbj0(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cby0(0), h_cby0(0));
    for (int i = 1; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cby0(i) - h_ref_cby0(i)),
                Kokkos::abs(h_ref_cby0(i)) * 1e-13);
    }

    ////Test large arguments
    d_z_large        = ViewType("d_z_large", 6);
    d_cbj0_large     = ViewType("d_cbj0_large", 6);
    d_cby0_large     = ViewType("d_cby0_large", 6);
    h_z_large        = Kokkos::create_mirror_view(d_z_large);
    h_cbj0_large     = Kokkos::create_mirror_view(d_cbj0_large);
    h_cby0_large     = Kokkos::create_mirror_view(d_cby0_large);
    h_ref_cbj0_large = HostViewType("h_ref_cbj0_large", 2);
    h_ref_cby0_large = HostViewType("h_ref_cby0_large", 2);

    h_z_large(0) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(1) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(2) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(3) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(4) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(5) = Kokkos::complex<double>(-10000.0, 100.0);

    Kokkos::deep_copy(d_z_large, h_z_large);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj0_large, d_cbj0_large);
    Kokkos::deep_copy(h_cby0_large, d_cby0_large);

    h_ref_cbj0_large(0) =
        Kokkos::complex<double>(-9.561811498244175e+40, -4.854995782103029e+40);
    h_ref_cbj0_large(1) =
        Kokkos::complex<double>(-9.561811498244175e+40, +4.854995782103029e+40);

    h_ref_cby0_large(0) =
        Kokkos::complex<double>(4.854995782103029e+40, -9.561811498244175e+40);
    h_ref_cby0_large(1) =
        Kokkos::complex<double>(-4.854995782103029e+40, -9.561811498244175e+40);

    EXPECT_TRUE((Kokkos::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) <
                 Kokkos::abs(h_ref_cbj0_large(0)) * 1e-12) &&
                (Kokkos::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) >
                 Kokkos::abs(h_ref_cbj0_large(0)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cbj0_large(1) - h_ref_cbj0_large(0)) >
                Kokkos::abs(h_ref_cbj0_large(0)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cbj0_large(2) - h_ref_cbj0_large(0)) <
                Kokkos::abs(h_ref_cbj0_large(0)) * 1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) <
                 Kokkos::abs(h_ref_cbj0_large(1)) * 1e-12) &&
                (Kokkos::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) >
                 Kokkos::abs(h_ref_cbj0_large(1)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cbj0_large(4) - h_ref_cbj0_large(1)) >
                Kokkos::abs(h_ref_cbj0_large(1)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cbj0_large(5) - h_ref_cbj0_large(1)) <
                Kokkos::abs(h_ref_cbj0_large(1)) * 1e-13);

    EXPECT_TRUE((Kokkos::abs(h_cby0_large(0) - h_ref_cby0_large(0)) <
                 Kokkos::abs(h_ref_cby0_large(0)) * 1e-12) &&
                (Kokkos::abs(h_cby0_large(0) - h_ref_cby0_large(0)) >
                 Kokkos::abs(h_ref_cby0_large(0)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cby0_large(1) - h_ref_cby0_large(0)) >
                Kokkos::abs(h_ref_cby0_large(0)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cby0_large(2) - h_ref_cby0_large(0)) <
                Kokkos::abs(h_ref_cby0_large(0)) * 1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cby0_large(3) - h_ref_cby0_large(1)) <
                 Kokkos::abs(h_ref_cby0_large(1)) * 1e-12) &&
                (Kokkos::abs(h_cby0_large(3) - h_ref_cby0_large(1)) >
                 Kokkos::abs(h_ref_cby0_large(1)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cby0_large(4) - h_ref_cby0_large(1)) >
                Kokkos::abs(h_ref_cby0_large(1)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cby0_large(5) - h_ref_cby0_large(1)) <
                Kokkos::abs(h_ref_cby0_large(1)) * 1e-13);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbj0(i) = Kokkos::Experimental::cyl_bessel_j(0, d_z(i));
    d_cby0(i) = Kokkos::Experimental::cyl_bessel_y(0, d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestLargeArgTag&, const int& /*i*/) const {
    d_cbj0_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cbj0_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               11000, 3000);
    d_cbj0_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               11000, 7500);
    d_cbj0_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cbj0_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               11000, 3000);
    d_cbj0_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_j0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               11000, 7500);

    d_cby0_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cby0_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               11000, 3000);
    d_cby0_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               11000, 7500);
    d_cby0_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cby0_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               11000, 3000);
    d_cby0_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_y0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               11000, 7500);
  }
};

template <class ExecSpace>
struct TestComplexBesselJ1Y1Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbj1, d_cby1;
  typename ViewType::HostMirror h_z, h_cbj1, h_cby1;
  HostViewType h_ref_cbj1, h_ref_cby1;

  ViewType d_z_large, d_cbj1_large, d_cby1_large;
  typename ViewType::HostMirror h_z_large, h_cbj1_large, h_cby1_large;
  HostViewType h_ref_cbj1_large, h_ref_cby1_large;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_cbj1     = ViewType("d_cbj1", N);
    d_cby1     = ViewType("d_cby1", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbj1     = Kokkos::create_mirror_view(d_cbj1);
    h_cby1     = Kokkos::create_mirror_view(d_cby1);
    h_ref_cbj1 = HostViewType("h_ref_cbj1", N);
    h_ref_cby1 = HostViewType("h_ref_cby1", N);

    // Generate test inputs
    h_z(0) = Kokkos::complex<double>(0.0, 0.0);
    // abs(z)<=25
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    // abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj1, d_cbj1);
    Kokkos::deep_copy(h_cby1, d_cby1);

    // Reference values computed with Octave
    h_ref_cbj1(0) = Kokkos::complex<double>(0, 0);
    h_ref_cbj1(1) =
        Kokkos::complex<double>(7.801488485792540e-01, -1.260982060238848e+00);
    h_ref_cbj1(2) =
        Kokkos::complex<double>(7.801488485792540e-01, +1.260982060238848e+00);
    h_ref_cbj1(3) =
        Kokkos::complex<double>(-7.801488485792543e-01, -1.260982060238848e+00);
    h_ref_cbj1(4) =
        Kokkos::complex<double>(-7.801488485792543e-01, +1.260982060238848e+00);
    h_ref_cbj1(5) =
        Kokkos::complex<double>(-7.469476253429664e+02, -1.576608505254311e+03);
    h_ref_cbj1(6) =
        Kokkos::complex<double>(-7.469476253429664e+02, +1.576608505254311e+03);
    h_ref_cbj1(7) =
        Kokkos::complex<double>(7.469476253429661e+02, -1.576608505254311e+03);
    h_ref_cbj1(8) =
        Kokkos::complex<double>(7.469476253429661e+02, +1.576608505254311e+03);
    h_ref_cbj1(9) = Kokkos::complex<double>(3.390589585259365e-01, 0);
    h_ref_cbj1(10) =
        Kokkos::complex<double>(-3.390589585259365e-01, +3.373499138396203e-17);
    h_ref_cbj1(11) = Kokkos::complex<double>(-3.951932188370151e-02, 0);
    h_ref_cbj1(12) =
        Kokkos::complex<double>(3.951932188370151e-02, +7.988560221984213e-18);
    h_ref_cbj1(13) =
        Kokkos::complex<double>(1.233147100257312e+03, -1.027302265904111e+03);
    h_ref_cbj1(14) =
        Kokkos::complex<double>(1.233147100257312e+03, +1.027302265904111e+03);
    h_ref_cbj1(15) =
        Kokkos::complex<double>(-1.233147100257312e+03, -1.027302265904111e+03);
    h_ref_cbj1(16) =
        Kokkos::complex<double>(-1.233147100257312e+03, +1.027302265904111e+03);
    h_ref_cbj1(17) =
        Kokkos::complex<double>(4.248029136732908e+02, -1.042364939115052e+03);
    h_ref_cbj1(18) =
        Kokkos::complex<double>(4.248029136732908e+02, +1.042364939115052e+03);
    h_ref_cbj1(19) =
        Kokkos::complex<double>(-4.248029136732909e+02, -1.042364939115052e+03);
    h_ref_cbj1(20) =
        Kokkos::complex<double>(-4.248029136732909e+02, +1.042364939115052e+03);
    h_ref_cbj1(21) = Kokkos::complex<double>(1.305514883350938e-01, 0);
    h_ref_cbj1(22) =
        Kokkos::complex<double>(-1.305514883350938e-01, +7.993709105806192e-18);
    h_ref_cbj1(23) = Kokkos::complex<double>(4.659838375816632e-02, 0);
    h_ref_cbj1(24) =
        Kokkos::complex<double>(-4.659838375816632e-02, +6.322680793358811e-18);

    h_ref_cby1(0) = Kokkos::complex<double>(-infinity<double>::value, 0);
    h_ref_cby1(1) =
        Kokkos::complex<double>(1.285849341463599e+00, +7.250812532419394e-01);
    h_ref_cby1(2) =
        Kokkos::complex<double>(1.285849341463599e+00, -7.250812532419394e-01);
    h_ref_cby1(3) =
        Kokkos::complex<double>(1.236114779014097e+00, -8.352164439165690e-01);
    h_ref_cby1(4) =
        Kokkos::complex<double>(1.236114779014097e+00, +8.352164439165690e-01);
    h_ref_cby1(5) =
        Kokkos::complex<double>(1.576608512528508e+03, -7.469476251109801e+02);
    h_ref_cby1(6) =
        Kokkos::complex<double>(1.576608512528508e+03, +7.469476251109801e+02);
    h_ref_cby1(7) =
        Kokkos::complex<double>(1.576608497980113e+03, +7.469476255749524e+02);
    h_ref_cby1(8) =
        Kokkos::complex<double>(1.576608497980113e+03, -7.469476255749524e+02);
    h_ref_cby1(9) = Kokkos::complex<double>(3.246744247918000e-01, 0);
    h_ref_cby1(10) =
        Kokkos::complex<double>(-3.246744247918000e-01, -6.781179170518730e-01);
    h_ref_cby1(11) = Kokkos::complex<double>(1.616692009926331e-01, 0);
    h_ref_cby1(12) =
        Kokkos::complex<double>(-1.616692009926332e-01, +7.903864376740302e-02);
    h_ref_cby1(13) =
        Kokkos::complex<double>(1.027302268200224e+03, +1.233147093992241e+03);
    h_ref_cby1(14) =
        Kokkos::complex<double>(1.027302268200224e+03, -1.233147093992241e+03);
    h_ref_cby1(15) =
        Kokkos::complex<double>(1.027302263607999e+03, -1.233147106522383e+03);
    h_ref_cby1(16) =
        Kokkos::complex<double>(1.027302263607999e+03, +1.233147106522383e+03);
    h_ref_cby1(17) =
        Kokkos::complex<double>(1.042364943073579e+03, +4.248029112344685e+02);
    h_ref_cby1(18) =
        Kokkos::complex<double>(1.042364943073579e+03, -4.248029112344685e+02);
    h_ref_cby1(19) =
        Kokkos::complex<double>(1.042364935156525e+03, -4.248029161121132e+02);
    h_ref_cby1(20) =
        Kokkos::complex<double>(1.042364935156525e+03, +4.248029161121132e+02);
    h_ref_cby1(21) = Kokkos::complex<double>(7.552212658226459e-02, 0);
    h_ref_cby1(22) =
        Kokkos::complex<double>(-7.552212658226459e-02, -2.611029766701876e-01);
    h_ref_cby1(23) = Kokkos::complex<double>(9.186960936986688e-02, 0);
    h_ref_cby1(24) =
        Kokkos::complex<double>(-9.186960936986688e-02, -9.319676751633262e-02);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbj1(i) - h_ref_cbj1(i)),
                Kokkos::abs(h_ref_cbj1(i)) * 1e-13);
    }

// FIXME_SYCL Failing for Intel GPUs
#if !(defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU))
    EXPECT_EQ(h_ref_cby1(0), h_cby1(0));
    for (int i = 1; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cby1(i) - h_ref_cby1(i)),
                Kokkos::abs(h_ref_cby1(i)) * 1e-13);
    }
#endif

    ////Test large arguments
    d_z_large        = ViewType("d_z_large", 6);
    d_cbj1_large     = ViewType("d_cbj1_large", 6);
    d_cby1_large     = ViewType("d_cby1_large", 6);
    h_z_large        = Kokkos::create_mirror_view(d_z_large);
    h_cbj1_large     = Kokkos::create_mirror_view(d_cbj1_large);
    h_cby1_large     = Kokkos::create_mirror_view(d_cby1_large);
    h_ref_cbj1_large = HostViewType("h_ref_cbj1_large", 2);
    h_ref_cby1_large = HostViewType("h_ref_cby1_large", 2);

    h_z_large(0) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(1) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(2) = Kokkos::complex<double>(10000.0, 100.0);
    h_z_large(3) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(4) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(5) = Kokkos::complex<double>(-10000.0, 100.0);

    Kokkos::deep_copy(d_z_large, h_z_large);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj1_large, d_cbj1_large);
    Kokkos::deep_copy(h_cby1_large, d_cby1_large);

    h_ref_cbj1_large(0) =
        Kokkos::complex<double>(4.854515317906369e+40, -9.562049455402486e+40);
    h_ref_cbj1_large(1) =
        Kokkos::complex<double>(-4.854515317906371e+40, -9.562049455402486e+40);

    h_ref_cby1_large(0) =
        Kokkos::complex<double>(9.562049455402486e+40, 4.854515317906369e+40);
    h_ref_cby1_large(1) =
        Kokkos::complex<double>(9.562049455402486e+40, -4.854515317906369e+40);

    EXPECT_TRUE((Kokkos::abs(h_cbj1_large(0) - h_ref_cbj1_large(0)) <
                 Kokkos::abs(h_ref_cbj1_large(0)) * 1e-12) &&
                (Kokkos::abs(h_cbj1_large(0) - h_ref_cbj1_large(0)) >
                 Kokkos::abs(h_ref_cbj1_large(0)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cbj1_large(1) - h_ref_cbj1_large(0)) >
                Kokkos::abs(h_ref_cbj1_large(0)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cbj1_large(2) - h_ref_cbj1_large(0)) <
                Kokkos::abs(h_ref_cbj1_large(0)) * 1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cbj1_large(3) - h_ref_cbj1_large(1)) <
                 Kokkos::abs(h_ref_cbj1_large(1)) * 1e-12) &&
                (Kokkos::abs(h_cbj1_large(3) - h_ref_cbj1_large(1)) >
                 Kokkos::abs(h_ref_cbj1_large(1)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cbj1_large(4) - h_ref_cbj1_large(1)) >
                Kokkos::abs(h_ref_cbj1_large(1)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cbj1_large(5) - h_ref_cbj1_large(1)) <
                Kokkos::abs(h_ref_cbj1_large(1)) * 1e-13);

    EXPECT_TRUE((Kokkos::abs(h_cby1_large(0) - h_ref_cby1_large(0)) <
                 Kokkos::abs(h_ref_cby1_large(0)) * 1e-12) &&
                (Kokkos::abs(h_cby1_large(0) - h_ref_cby1_large(0)) >
                 Kokkos::abs(h_ref_cby1_large(0)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cby1_large(1) - h_ref_cby1_large(0)) >
                Kokkos::abs(h_ref_cby1_large(0)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cby1_large(2) - h_ref_cby1_large(0)) <
                Kokkos::abs(h_ref_cby1_large(0)) * 1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cby1_large(3) - h_ref_cby1_large(1)) <
                 Kokkos::abs(h_ref_cby1_large(1)) * 1e-12) &&
                (Kokkos::abs(h_cby1_large(3) - h_ref_cby1_large(1)) >
                 Kokkos::abs(h_ref_cby1_large(1)) * 1e-13));
    EXPECT_TRUE(Kokkos::abs(h_cby1_large(4) - h_ref_cby1_large(1)) >
                Kokkos::abs(h_ref_cby1_large(1)) * 1e-6);
    EXPECT_TRUE(Kokkos::abs(h_cby1_large(5) - h_ref_cby1_large(1)) <
                Kokkos::abs(h_ref_cby1_large(1)) * 1e-13);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbj1(i) = Kokkos::Experimental::cyl_bessel_j(1, d_z(i));
    d_cby1(i) = Kokkos::Experimental::cyl_bessel_y(1, d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestLargeArgTag&, const int& /*i*/) const {
    d_cbj1_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cbj1_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               11000, 3000);
    d_cbj1_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               11000, 7500);
    d_cbj1_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cbj1_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               11000, 3000);
    d_cbj1_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_j1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               11000, 7500);

    d_cby1_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cby1_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               11000, 3000);
    d_cby1_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               11000, 7500);
    d_cby1_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cby1_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               11000, 3000);
    d_cby1_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_y1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               11000, 7500);
  }
};

template <class ExecSpace>
struct TestComplexBesselI0K0Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbi0, d_cbk0;
  typename ViewType::HostMirror h_z, h_cbi0, h_cbk0;
  HostViewType h_ref_cbi0, h_ref_cbk0;

  ViewType d_z_large, d_cbi0_large, d_cbk0_large;
  typename ViewType::HostMirror h_z_large, h_cbi0_large, h_cbk0_large;
  HostViewType h_ref_cbi0_large, h_ref_cbk0_large;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 26;
    d_z        = ViewType("d_z", N);
    d_cbi0     = ViewType("d_cbi0", N);
    d_cbk0     = ViewType("d_cbk0", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbi0     = Kokkos::create_mirror_view(d_cbi0);
    h_cbk0     = Kokkos::create_mirror_view(d_cbk0);
    h_ref_cbi0 = HostViewType("h_ref_cbi0", N);
    h_ref_cbk0 = HostViewType("h_ref_cbk0", N);

    // Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0, 0.0);
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);
    h_z(25) = Kokkos::complex<double>(7.998015e-5, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbi0, d_cbi0);
    Kokkos::deep_copy(h_cbk0, d_cbk0);

    // Reference values computed with Octave
    h_ref_cbi0(0) = Kokkos::complex<double>(1.000000000000000e+00, 0);
    h_ref_cbi0(1) =
        Kokkos::complex<double>(-4.695171920440706e-01, +4.313788409468920e+00);
    h_ref_cbi0(2) =
        Kokkos::complex<double>(-4.695171920440706e-01, -4.313788409468920e+00);
    h_ref_cbi0(3) =
        Kokkos::complex<double>(-4.695171920440706e-01, -4.313788409468920e+00);
    h_ref_cbi0(4) =
        Kokkos::complex<double>(-4.695171920440706e-01, +4.313788409468920e+00);
    h_ref_cbi0(5) =
        Kokkos::complex<double>(-7.276526052028507e+08, -2.806354803468570e+08);
    h_ref_cbi0(6) =
        Kokkos::complex<double>(-7.276526052028507e+08, +2.806354803468570e+08);
    h_ref_cbi0(7) =
        Kokkos::complex<double>(-7.276526052028507e+08, +2.806354803468570e+08);
    h_ref_cbi0(8) =
        Kokkos::complex<double>(-7.276526052028507e+08, -2.806354803468570e+08);
    h_ref_cbi0(9)  = Kokkos::complex<double>(4.880792585865025e+00, 0);
    h_ref_cbi0(10) = Kokkos::complex<double>(4.880792585865025e+00, 0);
    h_ref_cbi0(11) = Kokkos::complex<double>(8.151421225128924e+08, 0);
    h_ref_cbi0(12) = Kokkos::complex<double>(8.151421225128924e+08, 0);
    h_ref_cbi0(13) =
        Kokkos::complex<double>(-9.775983282455373e+10, -4.159160389327644e+10);
    h_ref_cbi0(14) =
        Kokkos::complex<double>(-9.775983282455373e+10, +4.159160389327644e+10);
    h_ref_cbi0(15) =
        Kokkos::complex<double>(-9.775983282455373e+10, +4.159160389327644e+10);
    h_ref_cbi0(16) =
        Kokkos::complex<double>(-9.775983282455373e+10, -4.159160389327644e+10);
    h_ref_cbi0(17) =
        Kokkos::complex<double>(-5.158377566681892e+24, -2.766704059464302e+24);
    h_ref_cbi0(18) =
        Kokkos::complex<double>(-5.158377566681892e+24, +2.766704059464302e+24);
    h_ref_cbi0(19) =
        Kokkos::complex<double>(-5.158377566681892e+24, +2.766704059464302e+24);
    h_ref_cbi0(20) =
        Kokkos::complex<double>(-5.158377566681892e+24, -2.766704059464302e+24);
    h_ref_cbi0(21) = Kokkos::complex<double>(1.095346047317573e+11, 0);
    h_ref_cbi0(22) = Kokkos::complex<double>(1.095346047317573e+11, 0);
    h_ref_cbi0(23) = Kokkos::complex<double>(5.894077055609803e+24, 0);
    h_ref_cbi0(24) = Kokkos::complex<double>(5.894077055609803e+24, 0);
    h_ref_cbi0(25) = Kokkos::complex<double>(1.0000000015992061009, 0);

    h_ref_cbk0(0) = Kokkos::complex<double>(infinity<double>::value, 0);
    h_ref_cbk0(1) =
        Kokkos::complex<double>(-2.078722558742977e-02, -2.431266356716766e-02);
    h_ref_cbk0(2) =
        Kokkos::complex<double>(-2.078722558742977e-02, +2.431266356716766e-02);
    h_ref_cbk0(3) =
        Kokkos::complex<double>(-1.357295320191579e+01, +1.499344424826928e+00);
    h_ref_cbk0(4) =
        Kokkos::complex<double>(-1.357295320191579e+01, -1.499344424826928e+00);
    h_ref_cbk0(5) =
        Kokkos::complex<double>(-1.820476218131465e-11, +1.795056004780177e-11);
    h_ref_cbk0(6) =
        Kokkos::complex<double>(-1.820476218131465e-11, -1.795056004780177e-11);
    h_ref_cbk0(7) =
        Kokkos::complex<double>(8.816423633943287e+08, +2.285988078870750e+09);
    h_ref_cbk0(8) =
        Kokkos::complex<double>(8.816423633943287e+08, -2.285988078870750e+09);
    h_ref_cbk0(9) = Kokkos::complex<double>(3.473950438627926e-02, 0);
    h_ref_cbk0(10) =
        Kokkos::complex<double>(3.473950438627926e-02, -1.533346213144909e+01);
    h_ref_cbk0(11) = Kokkos::complex<double>(2.667545110351910e-11, 0);
    h_ref_cbk0(12) =
        Kokkos::complex<double>(2.667545110351910e-11, -2.560844503718094e+09);
    h_ref_cbk0(13) =
        Kokkos::complex<double>(-1.163319528590747e-13, +1.073711234918388e-13);
    h_ref_cbk0(14) =
        Kokkos::complex<double>(-1.163319528590747e-13, -1.073711234918388e-13);
    h_ref_cbk0(15) =
        Kokkos::complex<double>(1.306638772421339e+11, +3.071215726177843e+11);
    h_ref_cbk0(16) =
        Kokkos::complex<double>(1.306638772421339e+11, -3.071215726177843e+11);
    h_ref_cbk0(17) =
        Kokkos::complex<double>(-1.111584549467388e-27, +8.581979311477652e-28);
    h_ref_cbk0(18) =
        Kokkos::complex<double>(-1.111584549467388e-27, -8.581979311477652e-28);
    h_ref_cbk0(19) =
        Kokkos::complex<double>(8.691857147870108e+24, +1.620552106793022e+25);
    h_ref_cbk0(20) =
        Kokkos::complex<double>(8.691857147870108e+24, -1.620552106793022e+25);
    h_ref_cbk0(21) = Kokkos::complex<double>(1.630534586888181e-13, 0);
    h_ref_cbk0(22) =
        Kokkos::complex<double>(1.630534586888181e-13, -3.441131095391506e+11);
    h_ref_cbk0(23) = Kokkos::complex<double>(1.413897840559108e-27, 0);
    h_ref_cbk0(24) =
        Kokkos::complex<double>(1.413897840559108e-27, -1.851678917759592e+25);
    h_ref_cbk0(25) = Kokkos::complex<double>(9.5496636116079915979, 0.);

    // FIXME_HIP Disable the test when using ROCm 5.5 and 5.6 due to a known
    // compiler bug
#if !defined(KOKKOS_ENABLE_HIP) || (HIP_VERSION_MAJOR != 5) || \
    ((HIP_VERSION_MAJOR == 5) &&                               \
     !((HIP_VERSION_MINOR == 5) || (HIP_VERSION_MINOR == 6)))
    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbi0(i) - h_ref_cbi0(i)),
                Kokkos::abs(h_ref_cbi0(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cbk0(0), h_cbk0(0));
    int upper_limit_0 = N;
    // FIXME_SYCL Failing for Intel GPUs, 19 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_0 = 19;
#endif
    for (int i = 1; i < upper_limit_0; i++) {
      EXPECT_LE(Kokkos::abs(h_cbk0(i) - h_ref_cbk0(i)),
                Kokkos::abs(h_ref_cbk0(i)) * 1e-13)
          << "at index " << i;
    }
#endif

    ////Test large arguments
    d_z_large        = ViewType("d_z_large", 6);
    d_cbi0_large     = ViewType("d_cbi0_large", 6);
    h_z_large        = Kokkos::create_mirror_view(d_z_large);
    h_cbi0_large     = Kokkos::create_mirror_view(d_cbi0_large);
    h_ref_cbi0_large = HostViewType("h_ref_cbi0_large", 2);

    h_z_large(0) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(1) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(2) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(3) = Kokkos::complex<double>(-100.0, 10.0);
    h_z_large(4) = Kokkos::complex<double>(-100.0, 10.0);
    h_z_large(5) = Kokkos::complex<double>(-100.0, 10.0);

    Kokkos::deep_copy(d_z_large, h_z_large);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbi0_large, d_cbi0_large);

    h_ref_cbi0_large(0) =
        Kokkos::complex<double>(-9.266819049505678e+41, -5.370779383266049e+41);
    h_ref_cbi0_large(1) =
        Kokkos::complex<double>(-9.266819049505678e+41, +5.370779383266049e+41);

    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(0) - h_ref_cbi0_large(0)) <
                Kokkos::abs(h_ref_cbi0_large(0)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(1) - h_ref_cbi0_large(0)) >
                Kokkos::abs(h_ref_cbi0_large(0)) * 1e-4);
    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(2) - h_ref_cbi0_large(0)) <
                Kokkos::abs(h_ref_cbi0_large(0)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(3) - h_ref_cbi0_large(1)) <
                Kokkos::abs(h_ref_cbi0_large(1)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(4) - h_ref_cbi0_large(1)) >
                Kokkos::abs(h_ref_cbi0_large(1)) * 1e-4);
    EXPECT_TRUE(Kokkos::abs(h_cbi0_large(5) - h_ref_cbi0_large(1)) <
                Kokkos::abs(h_ref_cbi0_large(1)) * 1e-15);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbi0(i) = Kokkos::Experimental::cyl_bessel_i(0, d_z(i));
    d_cbk0(i) = Kokkos::Experimental::cyl_bessel_k(0, d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestLargeArgTag&, const int& /*i*/) const {
    d_cbi0_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cbi0_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               110, 35);
    d_cbi0_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               110, 190);
    d_cbi0_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cbi0_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               110, 35);
    d_cbi0_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_i0<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               110, 190);
  }
};

template <class ExecSpace>
struct TestComplexBesselI1K1Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbi1, d_cbk1;
  typename ViewType::HostMirror h_z, h_cbi1, h_cbk1;
  HostViewType h_ref_cbi1, h_ref_cbk1;

  ViewType d_z_large, d_cbi1_large, d_cbk1_large;
  typename ViewType::HostMirror h_z_large, h_cbi1_large, h_cbk1_large;
  HostViewType h_ref_cbi1_large, h_ref_cbk1_large;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_cbi1     = ViewType("d_cbi1", N);
    d_cbk1     = ViewType("d_cbk1", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbi1     = Kokkos::create_mirror_view(d_cbi1);
    h_cbk1     = Kokkos::create_mirror_view(d_cbk1);
    h_ref_cbi1 = HostViewType("h_ref_cbi1", N);
    h_ref_cbk1 = HostViewType("h_ref_cbk1", N);

    // Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0, 0.0);
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbi1, d_cbi1);
    Kokkos::deep_copy(h_cbk1, d_cbk1);

    // Reference values computed with Octave
    h_ref_cbi1(0) = Kokkos::complex<double>(0, 0);
    h_ref_cbi1(1) =
        Kokkos::complex<double>(-8.127809410735776e-01, +3.780682961371298e+00);
    h_ref_cbi1(2) =
        Kokkos::complex<double>(-8.127809410735776e-01, -3.780682961371298e+00);
    h_ref_cbi1(3) =
        Kokkos::complex<double>(8.127809410735776e-01, +3.780682961371298e+00);
    h_ref_cbi1(4) =
        Kokkos::complex<double>(8.127809410735776e-01, -3.780682961371298e+00);
    h_ref_cbi1(5) =
        Kokkos::complex<double>(-7.119745937677552e+08, -2.813616375214342e+08);
    h_ref_cbi1(6) =
        Kokkos::complex<double>(-7.119745937677552e+08, +2.813616375214342e+08);
    h_ref_cbi1(7) =
        Kokkos::complex<double>(7.119745937677552e+08, -2.813616375214342e+08);
    h_ref_cbi1(8) =
        Kokkos::complex<double>(7.119745937677552e+08, +2.813616375214342e+08);
    h_ref_cbi1(9)  = Kokkos::complex<double>(3.953370217402609e+00, 0);
    h_ref_cbi1(10) = Kokkos::complex<double>(-3.953370217402609e+00, 0);
    h_ref_cbi1(11) = Kokkos::complex<double>(7.972200260896506e+08, 0);
    h_ref_cbi1(12) = Kokkos::complex<double>(-7.972200260896506e+08, 0);
    h_ref_cbi1(13) =
        Kokkos::complex<double>(-9.596150723281404e+10, -4.149038020045121e+10);
    h_ref_cbi1(14) =
        Kokkos::complex<double>(-9.596150723281404e+10, +4.149038020045121e+10);
    h_ref_cbi1(15) =
        Kokkos::complex<double>(9.596150723281404e+10, -4.149038020045121e+10);
    h_ref_cbi1(16) =
        Kokkos::complex<double>(9.596150723281404e+10, +4.149038020045121e+10);
    h_ref_cbi1(17) =
        Kokkos::complex<double>(-5.112615594220387e+24, -2.751210232069100e+24);
    h_ref_cbi1(18) =
        Kokkos::complex<double>(-5.112615594220387e+24, +2.751210232069100e+24);
    h_ref_cbi1(19) =
        Kokkos::complex<double>(5.112615594220387e+24, -2.751210232069100e+24);
    h_ref_cbi1(20) =
        Kokkos::complex<double>(5.112615594220387e+24, +2.751210232069100e+24);
    h_ref_cbi1(21) = Kokkos::complex<double>(1.075605042080823e+11, 0);
    h_ref_cbi1(22) = Kokkos::complex<double>(-1.075605042080823e+11, 0);
    h_ref_cbi1(23) = Kokkos::complex<double>(5.844751588390470e+24, 0);
    h_ref_cbi1(24) = Kokkos::complex<double>(-5.844751588390470e+24, 0);

    h_ref_cbk1(0) = Kokkos::complex<double>(infinity<double>::value, 0);
    h_ref_cbk1(1) =
        Kokkos::complex<double>(-2.480952007015153e-02, -2.557074905635180e-02);
    h_ref_cbk1(2) =
        Kokkos::complex<double>(-2.480952007015153e-02, +2.557074905635180e-02);
    h_ref_cbk1(3) =
        Kokkos::complex<double>(-1.185255629692602e+01, +2.527855884398198e+00);
    h_ref_cbk1(4) =
        Kokkos::complex<double>(-1.185255629692602e+01, -2.527855884398198e+00);
    h_ref_cbk1(5) =
        Kokkos::complex<double>(-1.839497240093994e-11, +1.841855854336314e-11);
    h_ref_cbk1(6) =
        Kokkos::complex<double>(-1.839497240093994e-11, -1.841855854336314e-11);
    h_ref_cbk1(7) =
        Kokkos::complex<double>(8.839236534393319e+08, +2.236734153323357e+09);
    h_ref_cbk1(8) =
        Kokkos::complex<double>(8.839236534393319e+08, -2.236734153323357e+09);
    h_ref_cbk1(9) = Kokkos::complex<double>(4.015643112819419e-02, 0);
    h_ref_cbk1(10) =
        Kokkos::complex<double>(-4.015643112819419e-02, -1.241987883191272e+01);
    h_ref_cbk1(11) = Kokkos::complex<double>(2.724930589574976e-11, 0);
    h_ref_cbk1(12) =
        Kokkos::complex<double>(-2.724930589574976e-11, -2.504540577257910e+09);
    h_ref_cbk1(13) =
        Kokkos::complex<double>(-1.175637676331817e-13, +1.097080943197297e-13);
    h_ref_cbk1(14) =
        Kokkos::complex<double>(-1.175637676331817e-13, -1.097080943197297e-13);
    h_ref_cbk1(15) =
        Kokkos::complex<double>(1.303458736323849e+11, +3.014719661500124e+11);
    h_ref_cbk1(16) =
        Kokkos::complex<double>(1.303458736323849e+11, -3.014719661500124e+11);
    h_ref_cbk1(17) =
        Kokkos::complex<double>(-1.119411861396158e-27, +8.666195226392352e-28);
    h_ref_cbk1(18) =
        Kokkos::complex<double>(-1.119411861396158e-27, -8.666195226392352e-28);
    h_ref_cbk1(19) =
        Kokkos::complex<double>(8.643181853549355e+24, +1.606175559143138e+25);
    h_ref_cbk1(20) =
        Kokkos::complex<double>(8.643181853549355e+24, -1.606175559143138e+25);
    h_ref_cbk1(21) = Kokkos::complex<double>(1.659400107332009e-13, 0);
    h_ref_cbk1(22) =
        Kokkos::complex<double>(-1.659400107332009e-13, -3.379112898365253e+11);
    h_ref_cbk1(23) = Kokkos::complex<double>(1.425632026517104e-27, 0);
    h_ref_cbk1(24) =
        Kokkos::complex<double>(-1.425632026517104e-27, -1.836182865214478e+25);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbi1(i) - h_ref_cbi1(i)),
                Kokkos::abs(h_ref_cbi1(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cbk1(0), h_cbk1(0));
    int upper_limit_1 = N;
    // FIXME_SYCL Failing for Intel GPUs, 8 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_1 = 8;
#endif
    for (int i = 1; i < upper_limit_1; i++) {
      EXPECT_LE(Kokkos::abs(h_cbk1(i) - h_ref_cbk1(i)),
                Kokkos::abs(h_ref_cbk1(i)) * 1e-13)
          << "at index " << i;
    }

    ////Test large arguments
    d_z_large        = ViewType("d_z_large", 6);
    d_cbi1_large     = ViewType("d_cbi1_large", 6);
    h_z_large        = Kokkos::create_mirror_view(d_z_large);
    h_cbi1_large     = Kokkos::create_mirror_view(d_cbi1_large);
    h_ref_cbi1_large = HostViewType("h_ref_cbi1_large", 2);

    h_z_large(0) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(1) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(2) = Kokkos::complex<double>(100.0, 10.0);
    h_z_large(3) = Kokkos::complex<double>(-100.0, 10.0);
    h_z_large(4) = Kokkos::complex<double>(-100.0, 10.0);
    h_z_large(5) = Kokkos::complex<double>(-100.0, 10.0);

    Kokkos::deep_copy(d_z_large, h_z_large);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace, Property, TestLargeArgTag>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbi1_large, d_cbi1_large);

    h_ref_cbi1_large(0) =
        Kokkos::complex<double>(-9.218158020154234e+41, -5.348736158968607e+41);
    h_ref_cbi1_large(1) =
        Kokkos::complex<double>(9.218158020154234e+41, -5.348736158968607e+41);

    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(0) - h_ref_cbi1_large(0)) <
                Kokkos::abs(h_ref_cbi1_large(0)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(1) - h_ref_cbi1_large(0)) >
                Kokkos::abs(h_ref_cbi1_large(0)) * 1e-4);
    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(2) - h_ref_cbi1_large(0)) <
                Kokkos::abs(h_ref_cbi1_large(0)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(3) - h_ref_cbi1_large(1)) <
                Kokkos::abs(h_ref_cbi1_large(1)) * 1e-15);
    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(4) - h_ref_cbi1_large(1)) >
                Kokkos::abs(h_ref_cbi1_large(1)) * 1e-4);
    EXPECT_TRUE(Kokkos::abs(h_cbi1_large(5) - h_ref_cbi1_large(1)) <
                Kokkos::abs(h_ref_cbi1_large(1)) * 1e-15);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbi1(i) = Kokkos::Experimental::cyl_bessel_i(1, d_z(i));
    d_cbk1(i) = Kokkos::Experimental::cyl_bessel_k(1, d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TestLargeArgTag&, const int& /*i*/) const {
    d_cbi1_large(0) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(0));
    d_cbi1_large(1) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(1),
                                                               110, 35);
    d_cbi1_large(2) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(2),
                                                               110, 190);
    d_cbi1_large(3) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(3));
    d_cbi1_large(4) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(4),
                                                               110, 35);
    d_cbi1_large(5) =
        Kokkos::Experimental::Impl::cyl_bessel_i1<Kokkos::complex<double>,
                                                  double, int>(d_z_large(5),
                                                               110, 190);
  }
};

template <class ExecSpace>
struct TestComplexBesselH1Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_ch10, d_ch11;
  typename ViewType::HostMirror h_z, h_ch10, h_ch11;
  HostViewType h_ref_ch10, h_ref_ch11;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_ch10     = ViewType("d_ch10", N);
    d_ch11     = ViewType("d_ch11", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_ch10     = Kokkos::create_mirror_view(d_ch10);
    h_ch11     = Kokkos::create_mirror_view(d_ch11);
    h_ref_ch10 = HostViewType("h_ref_ch10", N);
    h_ref_ch11 = HostViewType("h_ref_ch11", N);

    // Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0, 0.0);
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(200.0, 60.0);
    h_z(18) = Kokkos::complex<double>(200.0, -60.0);
    h_z(19) = Kokkos::complex<double>(-200.0, 60.0);
    h_z(20) = Kokkos::complex<double>(-200.0, -60.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(200.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-200.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Hankel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_ch10, d_ch10);
    Kokkos::deep_copy(h_ch11, d_ch11);

    // Reference values computed with Octave
    h_ref_ch10(0) = Kokkos::complex<double>(1.0, -infinity<double>::value);
    h_ref_ch10(1) =
        Kokkos::complex<double>(-1.779327030399459e-02, +5.281940449715537e-02);
    h_ref_ch10(2) =
        Kokkos::complex<double>(-2.480676488910849e+00, +1.948786988612626e+00);
    h_ref_ch10(3) =
        Kokkos::complex<double>(1.779327030399459e-02, +5.281940449715537e-02);
    h_ref_ch10(4) =
        Kokkos::complex<double>(-2.516263029518839e+00, -1.843148179618315e+00);
    h_ref_ch10(5) =
        Kokkos::complex<double>(-7.217716938222564e-06, -1.002796203581228e-07);
    h_ref_ch10(6) =
        Kokkos::complex<double>(-3.204879955218674e+03, -1.446133490498241e+03);
    h_ref_ch10(7) =
        Kokkos::complex<double>(7.217716938222564e-06, -1.002796203581228e-07);
    h_ref_ch10(8) =
        Kokkos::complex<double>(-3.204879969654108e+03, +1.446133490297682e+03);
    h_ref_ch10(9) =
        Kokkos::complex<double>(-2.600519549019334e-01, +3.768500100127903e-01);
    h_ref_ch10(10) =
        Kokkos::complex<double>(2.600519549019334e-01, +3.768500100127903e-01);
    h_ref_ch10(11) =
        Kokkos::complex<double>(-1.624127813134865e-01, -3.598179027370283e-02);
    h_ref_ch10(12) =
        Kokkos::complex<double>(1.624127813134865e-01, -3.598179027370283e-02);
    h_ref_ch10(13) =
        Kokkos::complex<double>(-2.184905481759440e-06, +6.263387166445335e-06);
    h_ref_ch10(14) =
        Kokkos::complex<double>(-2.025824374843011e+03, +2.512479278555672e+03);
    h_ref_ch10(15) =
        Kokkos::complex<double>(2.184905481759440e-06, +6.263387166445335e-06);
    h_ref_ch10(16) =
        Kokkos::complex<double>(-2.025824379212821e+03, -2.512479266028897e+03);
    h_ref_ch10(17) =
        Kokkos::complex<double>(-1.983689762743337e-28, -4.408449940359881e-28);
    h_ref_ch10(18) =
        Kokkos::complex<double>(-8.261945332108929e+23, -6.252486138159269e+24);
    h_ref_ch10(19) =
        Kokkos::complex<double>(1.983689762743337e-28, -4.408449940359881e-28);
    h_ref_ch10(20) =
        Kokkos::complex<double>(-8.261945332108929e+23, +6.252486138159269e+24);
    h_ref_ch10(21) =
        Kokkos::complex<double>(-7.315701054899959e-02, +1.318364704235323e-01);
    h_ref_ch10(22) =
        Kokkos::complex<double>(7.315701054899959e-02, +1.318364704235323e-01);
    h_ref_ch10(23) =
        Kokkos::complex<double>(-1.543743993056510e-02, -5.426577524981793e-02);
    h_ref_ch10(24) =
        Kokkos::complex<double>(1.543743993056510e-02, -5.426577524981793e-02);

    h_ref_ch11(0) = Kokkos::complex<double>(0.0, -infinity<double>::value);
    h_ref_ch11(1) =
        Kokkos::complex<double>(5.506759533731469e-02, +2.486728122475093e-02);
    h_ref_ch11(2) =
        Kokkos::complex<double>(1.505230101821194e+00, +2.546831401702448e+00);
    h_ref_ch11(3) =
        Kokkos::complex<double>(5.506759533731469e-02, -2.486728122475093e-02);
    h_ref_ch11(4) =
        Kokkos::complex<double>(-1.615365292495823e+00, +2.497096839252946e+00);
    h_ref_ch11(5) =
        Kokkos::complex<double>(-2.319863729607219e-07, +7.274197719836158e-06);
    h_ref_ch11(6) =
        Kokkos::complex<double>(-1.493895250453947e+03, +3.153217017782819e+03);
    h_ref_ch11(7) =
        Kokkos::complex<double>(-2.319863729607210e-07, -7.274197719836158e-06);
    h_ref_ch11(8) =
        Kokkos::complex<double>(1.493895250917918e+03, +3.153217003234423e+03);
    h_ref_ch11(9) =
        Kokkos::complex<double>(3.390589585259364e-01, +3.246744247918000e-01);
    h_ref_ch11(10) =
        Kokkos::complex<double>(3.390589585259364e-01, -3.246744247918000e-01);
    h_ref_ch11(11) =
        Kokkos::complex<double>(-3.951932188370152e-02, +1.616692009926331e-01);
    h_ref_ch11(12) =
        Kokkos::complex<double>(-3.951932188370151e-02, -1.616692009926331e-01);
    h_ref_ch11(13) =
        Kokkos::complex<double>(6.265071091331731e-06, +2.296112637347948e-06);
    h_ref_ch11(14) =
        Kokkos::complex<double>(2.466294194249553e+03, +2.054604534104335e+03);
    h_ref_ch11(15) =
        Kokkos::complex<double>(6.265071091331731e-06, -2.296112637347947e-06);
    h_ref_ch11(16) =
        Kokkos::complex<double>(-2.466294206779695e+03, +2.054604529512110e+03);
    h_ref_ch11(17) =
        Kokkos::complex<double>(-4.416040381930448e-28, +1.974955285825768e-28);
    h_ref_ch11(18) =
        Kokkos::complex<double>(-6.250095237987940e+24, +8.112776606830997e+23);
    h_ref_ch11(19) =
        Kokkos::complex<double>(-4.416040381930448e-28, -1.974955285825769e-28);
    h_ref_ch11(20) =
        Kokkos::complex<double>(6.250095237987940e+24, +8.112776606831005e+23);
    h_ref_ch11(21) =
        Kokkos::complex<double>(1.305514883350938e-01, +7.552212658226459e-02);
    h_ref_ch11(22) =
        Kokkos::complex<double>(1.305514883350938e-01, -7.552212658226456e-02);
    h_ref_ch11(23) =
        Kokkos::complex<double>(-5.430453818237824e-02, +1.530182458038999e-02);
    h_ref_ch11(24) =
        Kokkos::complex<double>(-5.430453818237824e-02, -1.530182458039000e-02);

    // FIXME_HIP Disable the test when using ROCm 5.5 and 5.6 due to a known
    // compiler bug
#if !defined(KOKKOS_ENABLE_HIP) || (HIP_VERSION_MAJOR != 5) || \
    ((HIP_VERSION_MAJOR == 5) &&                               \
     !((HIP_VERSION_MINOR == 5) || (HIP_VERSION_MINOR == 6)))
    EXPECT_EQ(h_ref_ch10(0), h_ch10(0));
    int upper_limit_10 = N;
// FIXME_SYCL Failing for Intel GPUs, 17 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_10 = 17;
#endif
    for (int i = 1; i < upper_limit_10; i++) {
      EXPECT_LE(Kokkos::abs(h_ch10(i) - h_ref_ch10(i)),
                Kokkos::abs(h_ref_ch10(i)) * 1e-13)
          << "at index " << i;
    }

    EXPECT_EQ(h_ref_ch11(0), h_ch11(0));
    int upper_limit_11 = N;
    // FIXME_SYCL Failing for Intel GPUs, 2 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_11 = 2;
#endif
    for (int i = 1; i < upper_limit_11; i++) {
      EXPECT_LE(Kokkos::abs(h_ch11(i) - h_ref_ch11(i)),
                Kokkos::abs(h_ref_ch11(i)) * 1e-13)
          << "at index " << i;
    }
#endif
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_ch10(i) = Kokkos::Experimental::cyl_bessel_h1(0, d_z(i));
    d_ch11(i) = Kokkos::Experimental::cyl_bessel_h1(1, d_z(i));
  }
};

template <class ExecSpace>
struct TestComplexBesselH2Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_ch20, d_ch21;
  typename ViewType::HostMirror h_z, h_ch20, h_ch21;
  HostViewType h_ref_ch20, h_ref_ch21;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_ch20     = ViewType("d_ch20", N);
    d_ch21     = ViewType("d_ch21", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_ch20     = Kokkos::create_mirror_view(d_ch20);
    h_ch21     = Kokkos::create_mirror_view(d_ch21);
    h_ref_ch20 = HostViewType("h_ref_ch20", N);
    h_ref_ch21 = HostViewType("h_ref_ch21", N);

    // Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0, 0.0);
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(200.0, 60.0);
    h_z(18) = Kokkos::complex<double>(200.0, -60.0);
    h_z(19) = Kokkos::complex<double>(-200.0, 60.0);
    h_z(20) = Kokkos::complex<double>(-200.0, -60.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(200.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-200.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Hankel functions
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_ch20, d_ch20);
    Kokkos::deep_copy(h_ch21, d_ch21);

    // Reference values computed with Octave
    h_ref_ch20(0) = Kokkos::complex<double>(1.0, infinity<double>::value);
    h_ref_ch20(1) =
        Kokkos::complex<double>(-2.480676488910849e+00, -1.948786988612626e+00);
    h_ref_ch20(2) =
        Kokkos::complex<double>(-1.779327030399459e-02, -5.281940449715537e-02);
    h_ref_ch20(3) =
        Kokkos::complex<double>(-2.516263029518839e+00, +1.843148179618315e+00);
    h_ref_ch20(4) =
        Kokkos::complex<double>(1.779327030399459e-02, -5.281940449715537e-02);
    h_ref_ch20(5) =
        Kokkos::complex<double>(-3.204879955218674e+03, +1.446133490498241e+03);
    h_ref_ch20(6) =
        Kokkos::complex<double>(-7.217716938222564e-06, +1.002796203581228e-07);
    h_ref_ch20(7) =
        Kokkos::complex<double>(-3.204879969654108e+03, -1.446133490297682e+03);
    h_ref_ch20(8) =
        Kokkos::complex<double>(7.217716938222564e-06, +1.002796203581228e-07);
    h_ref_ch20(9) =
        Kokkos::complex<double>(-2.600519549019334e-01, -3.768500100127903e-01);
    h_ref_ch20(10) =
        Kokkos::complex<double>(-7.801558647058006e-01, -3.768500100127903e-01);
    h_ref_ch20(11) =
        Kokkos::complex<double>(-1.624127813134865e-01, +3.598179027370283e-02);
    h_ref_ch20(12) =
        Kokkos::complex<double>(-4.872383439404597e-01, +3.598179027370281e-02);
    h_ref_ch20(13) =
        Kokkos::complex<double>(-2.025824374843011e+03, -2.512479278555672e+03);
    h_ref_ch20(14) =
        Kokkos::complex<double>(-2.184905481759440e-06, -6.263387166445335e-06);
    h_ref_ch20(15) =
        Kokkos::complex<double>(-2.025824379212821e+03, +2.512479266028897e+03);
    h_ref_ch20(16) =
        Kokkos::complex<double>(2.184905481759440e-06, -6.263387166445335e-06);
    h_ref_ch20(17) =
        Kokkos::complex<double>(-8.261945332108929e+23, +6.252486138159269e+24);
    h_ref_ch20(18) =
        Kokkos::complex<double>(-1.983689762743337e-28, +4.408449940359881e-28);
    h_ref_ch20(19) =
        Kokkos::complex<double>(-8.261945332108929e+23, -6.252486138159269e+24);
    h_ref_ch20(20) =
        Kokkos::complex<double>(1.983689762743337e-28, +4.408449940359881e-28);
    h_ref_ch20(21) =
        Kokkos::complex<double>(-7.315701054899959e-02, -1.318364704235323e-01);
    h_ref_ch20(22) =
        Kokkos::complex<double>(-2.194710316469988e-01, -1.318364704235323e-01);
    h_ref_ch20(23) =
        Kokkos::complex<double>(-1.543743993056510e-02, +5.426577524981793e-02);
    h_ref_ch20(24) =
        Kokkos::complex<double>(-4.631231979169528e-02, +5.426577524981793e-02);

    h_ref_ch21(0) = Kokkos::complex<double>(0.0, infinity<double>::value);
    h_ref_ch21(1) =
        Kokkos::complex<double>(1.505230101821194e+00, -2.546831401702448e+00);
    h_ref_ch21(2) =
        Kokkos::complex<double>(5.506759533731469e-02, -2.486728122475093e-02);
    h_ref_ch21(3) =
        Kokkos::complex<double>(-1.615365292495823e+00, -2.497096839252946e+00);
    h_ref_ch21(4) =
        Kokkos::complex<double>(5.506759533731469e-02, +2.486728122475093e-02);
    h_ref_ch21(5) =
        Kokkos::complex<double>(-1.493895250453947e+03, -3.153217017782819e+03);
    h_ref_ch21(6) =
        Kokkos::complex<double>(-2.319863729607219e-07, -7.274197719836158e-06);
    h_ref_ch21(7) =
        Kokkos::complex<double>(1.493895250917918e+03, -3.153217003234423e+03);
    h_ref_ch21(8) =
        Kokkos::complex<double>(-2.319863729607210e-07, +7.274197719836158e-06);
    h_ref_ch21(9) =
        Kokkos::complex<double>(3.390589585259364e-01, -3.246744247918000e-01);
    h_ref_ch21(10) =
        Kokkos::complex<double>(-1.017176875577809e+00, +3.246744247918000e-01);
    h_ref_ch21(11) =
        Kokkos::complex<double>(-3.951932188370152e-02, -1.616692009926331e-01);
    h_ref_ch21(12) =
        Kokkos::complex<double>(1.185579656511045e-01, +1.616692009926332e-01);
    h_ref_ch21(13) =
        Kokkos::complex<double>(2.466294194249553e+03, -2.054604534104335e+03);
    h_ref_ch21(14) =
        Kokkos::complex<double>(6.265071091331731e-06, -2.296112637347948e-06);
    h_ref_ch21(15) =
        Kokkos::complex<double>(-2.466294206779695e+03, -2.054604529512110e+03);
    h_ref_ch21(16) =
        Kokkos::complex<double>(6.265071091331731e-06, +2.296112637347947e-06);
    h_ref_ch21(17) =
        Kokkos::complex<double>(-6.250095237987940e+24, -8.112776606830997e+23);
    h_ref_ch21(18) =
        Kokkos::complex<double>(-4.416040381930448e-28, -1.974955285825768e-28);
    h_ref_ch21(19) =
        Kokkos::complex<double>(6.250095237987940e+24, -8.112776606831005e+23);
    h_ref_ch21(20) =
        Kokkos::complex<double>(-4.416040381930448e-28, +1.974955285825769e-28);
    h_ref_ch21(21) =
        Kokkos::complex<double>(1.305514883350938e-01, -7.552212658226459e-02);
    h_ref_ch21(22) =
        Kokkos::complex<double>(-3.916544650052814e-01, +7.552212658226461e-02);
    h_ref_ch21(23) =
        Kokkos::complex<double>(-5.430453818237824e-02, -1.530182458038999e-02);
    h_ref_ch21(24) =
        Kokkos::complex<double>(1.629136145471347e-01, +1.530182458039000e-02);

    // FIXME_HIP Disable the test when using ROCm 5.5 and 5.6 due to a known
    // compiler bug
#if !defined(KOKKOS_ENABLE_HIP) || (HIP_VERSION_MAJOR != 5) || \
    ((HIP_VERSION_MAJOR == 5) &&                               \
     !((HIP_VERSION_MINOR == 5) || (HIP_VERSION_MINOR == 6)))
    EXPECT_EQ(h_ref_ch20(0), h_ch20(0));
    int upper_limit_20 = N;
// FIXME_SYCL Failing for Intel GPUs, 16 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_20 = 16;
#endif
    for (int i = 1; i < upper_limit_20; i++) {
      EXPECT_LE(Kokkos::abs(h_ch20(i) - h_ref_ch20(i)),
                Kokkos::abs(h_ref_ch20(i)) * 1e-13)
          << "at index " << i;
    }

    EXPECT_EQ(h_ref_ch21(0), h_ch21(0));
    int upper_limit_21 = N;
    // FIXME_SYCL Failing for Intel GPUs, 1 is the first failing test case
#if defined(KOKKOS_ENABLE_SYCL) && defined(KOKKOS_ARCH_INTEL_GPU)
    if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::SYCL>)
      upper_limit_21 = 1;
#endif
    for (int i = 1; i < upper_limit_21; i++) {
      EXPECT_LE(Kokkos::abs(h_ch21(i) - h_ref_ch21(i)),
                Kokkos::abs(h_ref_ch21(i)) * 1e-13)
          << "at index " << i;
    }
#endif
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_ch20(i) = Kokkos::Experimental::cyl_bessel_h2(0, d_z(i));
    d_ch21(i) = Kokkos::Experimental::cyl_bessel_h2(1, d_z(i));
  }
};

template <class ExecSpace>
struct TestComplexBesselJ2Y2Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbj2, d_cby2;
  typename ViewType::HostMirror h_z, h_cbj2, h_cby2;
  HostViewType h_ref_cbj2, h_ref_cby2;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_cbj2     = ViewType("d_cbj2", N);
    d_cby2     = ViewType("d_cby2", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbj2     = Kokkos::create_mirror_view(d_cbj2);
    h_cby2     = Kokkos::create_mirror_view(d_cby2);
    h_ref_cbj2 = HostViewType("h_ref_cbj2", N);
    h_ref_cby2 = HostViewType("h_ref_cby2", N);

    // Generate test inputs
    h_z(0) = Kokkos::complex<double>(0.0, 0.0);
    // abs(z)<=25
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    // abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);

    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj2, d_cbj2);
    Kokkos::deep_copy(h_cby2, d_cby2);

    // Reference values computed with Octave
    h_ref_cbj2(0) = Kokkos::complex<double>(0, 0);
    h_ref_cbj2(1) =
        Kokkos::complex<double>(1.221309098878202e+00, 1.259462723846502e-01);
    h_ref_cbj2(2) =
        Kokkos::complex<double>(1.221309098878202e+00, -1.259462723846502e-01);
    h_ref_cbj2(3) =
        Kokkos::complex<double>(1.221309098878202e+00, -1.259462723846502e-01);
    h_ref_cbj2(4) =
        Kokkos::complex<double>(1.221309098878202e+00, 1.259462723846502e-01);
    h_ref_cbj2(5) =
        Kokkos::complex<double>(1.497683604634948e+03, -8.146168862718569e+02);
    h_ref_cbj2(6) =
        Kokkos::complex<double>(1.497683604634948e+03, 8.146168862718569e+02);
    h_ref_cbj2(7) =
        Kokkos::complex<double>(1.497683604634948e+03, 8.146168862718569e+02);
    h_ref_cbj2(8) =
        Kokkos::complex<double>(1.497683604634948e+03, -8.146168862718569e+02);
    h_ref_cbj2(9) = Kokkos::complex<double>(4.860912605858912e-01, 0.0);
    h_ref_cbj2(10) =
        Kokkos::complex<double>(4.860912605858912e-01, -5.952901063700137e-17);
    h_ref_cbj2(11) = Kokkos::complex<double>(1.589763185409908e-01, 0.0);
    h_ref_cbj2(12) =
        Kokkos::complex<double>(1.589763185409908e-01, 1.387778780781446e-17);
    h_ref_cbj2(13) =
        Kokkos::complex<double>(1067.787971654599, 1163.262408888476);
    h_ref_cbj2(14) =
        Kokkos::complex<double>(1067.787971654599, -1163.262408888476);
    h_ref_cbj2(15) =
        Kokkos::complex<double>(1067.787971654599, -1163.262408888476);
    h_ref_cbj2(16) =
        Kokkos::complex<double>(1067.787971654599, 1163.262408888476);
    h_ref_cbj2(17) =
        Kokkos::complex<double>(1.048358121387835e+03, 3.977175762574223e+02);
    h_ref_cbj2(18) =
        Kokkos::complex<double>(1.048358121387835e+03, -3.977175762574223e+02);
    h_ref_cbj2(19) =
        Kokkos::complex<double>(1.048358121387835e+03, -3.977175762574223e+02);
    h_ref_cbj2(20) =
        Kokkos::complex<double>(1.048358121387835e+03, 3.977175762574223e+02);
    h_ref_cbj2(21) = Kokkos::complex<double>(8.248211685864917e-02, 0.0);
    h_ref_cbj2(22) = Kokkos::complex<double>(8.248211685864917e-02, 0.0);
    h_ref_cbj2(23) = Kokkos::complex<double>(9.302508354766742e-02, 0.0);
    h_ref_cbj2(24) =
        Kokkos::complex<double>(9.302508354766742e-02, -6.938893903907228e-18);

    h_ref_cby2(0) = Kokkos::complex<double>(-infinity<double>::value, 0);
    h_ref_cby2(1) =
        Kokkos::complex<double>(-1.842323456510943e-01, 1.170448544195523e+00);
    h_ref_cby2(2) =
        Kokkos::complex<double>(-1.842323456510943e-01, -1.170448544195523e+00);
    h_ref_cby2(3) =
        Kokkos::complex<double>(6.766019911820612e-02, 1.272169653560880e+00);
    h_ref_cby2(4) =
        Kokkos::complex<double>(6.766019911820612e-02, -1.272169653560880e+00);
    h_ref_cby2(5) =
        Kokkos::complex<double>(8.146168869114892e+02, 1.497683597202903e+03);
    h_ref_cby2(6) =
        Kokkos::complex<double>(8.146168869114892e+02, -1.497683597202903e+03);
    h_ref_cby2(7) =
        Kokkos::complex<double>(-8.146168856322246e+02, 1.497683612066993e+03);
    h_ref_cby2(8) =
        Kokkos::complex<double>(-8.146168856322246e+02, -1.497683612066993e+03);
    h_ref_cby2(9) = Kokkos::complex<double>(-1.604003934849236e-01, 0.0);
    h_ref_cby2(10) =
        Kokkos::complex<double>(-1.604003934849236e-01, 9.721825211717822e-01);
    h_ref_cby2(11) = Kokkos::complex<double>(5.003998166436658e-02, 0.0);
    h_ref_cby2(12) =
        Kokkos::complex<double>(5.003998166436657e-02, 3.179526370819815e-01);
    h_ref_cby2(13) =
        Kokkos::complex<double>(-1163.262415148152, 1067.787969020863);
    h_ref_cby2(14) =
        Kokkos::complex<double>(-1163.262415148152, -1067.787969020863);
    h_ref_cby2(15) =
        Kokkos::complex<double>(1163.262402628800, 1067.787974288336);
    h_ref_cby2(16) =
        Kokkos::complex<double>(1163.262402628800, -1067.787974288336);
    h_ref_cby2(17) =
        Kokkos::complex<double>(-3.977175786094755e+02, 1.048358117354230e+03);
    h_ref_cby2(18) =
        Kokkos::complex<double>(-3.977175786094755e+02, -1.048358117354230e+03);
    h_ref_cby2(19) =
        Kokkos::complex<double>(3.977175739053691e+02, 1.048358125421440e+03);
    h_ref_cby2(20) =
        Kokkos::complex<double>(3.977175739053691e+02, -1.048358125421440e+03);
    h_ref_cby2(21) = Kokkos::complex<double>(-1.264420328105134e-01, 0.0);
    h_ref_cby2(22) =
        Kokkos::complex<double>(-1.264420328105134e-01, 1.649642337172983e-01);
    h_ref_cby2(23) = Kokkos::complex<double>(-4.429663189712049e-02, 0.0);
    h_ref_cby2(24) =
        Kokkos::complex<double>(-4.429663189712048e-02, 1.860501670953348e-01);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbj2(i) - h_ref_cbj2(i)),
                Kokkos::abs(h_ref_cbj2(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cby2(0), h_cby2(0));
    for (int i = 1; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cby2(i) - h_ref_cby2(i)),
                Kokkos::abs(h_ref_cby2(i)) * 1e-13);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbj2(i) = Kokkos::Experimental::cyl_bessel_j(2, d_z(i));
    d_cby2(i) = Kokkos::Experimental::cyl_bessel_y(2, d_z(i));
  }
};

template <class ExecSpace>
struct TestComplexBesselI2K2Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbi2, d_cbk2;
  typename ViewType::HostMirror h_z, h_cbi2, h_cbk2;
  HostViewType h_ref_cbi2, h_ref_cbk2;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 25;
    d_z        = ViewType("d_z", N);
    d_cbi2     = ViewType("d_cbi2", N);
    d_cbk2     = ViewType("d_cbk2", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbi2     = Kokkos::create_mirror_view(d_cbi2);
    h_cbk2     = Kokkos::create_mirror_view(d_cbk2);
    h_ref_cbi2 = HostViewType("h_ref_cbi2", N);
    h_ref_cbk2 = HostViewType("h_ref_cbk2", N);

    // Generate test inputs
    h_z(0) = Kokkos::complex<double>(0.0, 0.0);
    // abs(z)<=25
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    // abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);
    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbi2, d_cbi2);
    Kokkos::deep_copy(h_cbk2, d_cbk2);

    // Reference values computed with Octave
    h_ref_cbi2(0) = Kokkos::complex<double>(0, 0);
    h_ref_cbi2(1) =
        Kokkos::complex<double>(-1.257674591970511, 2.318771368505682);
    h_ref_cbi2(2) =
        Kokkos::complex<double>(-1.257674591970511, -2.318771368505682);
    h_ref_cbi2(3) =
        Kokkos::complex<double>(-1.257674591970511, -2.318771368505682);
    h_ref_cbi2(4) =
        Kokkos::complex<double>(-1.257674591970511, 2.318771368505682);
    h_ref_cbi2(5) =
        Kokkos::complex<double>(-666638194.9266258, -282697255.4650909);
    h_ref_cbi2(6) =
        Kokkos::complex<double>(-666638194.9266258, 282697255.4650909);
    h_ref_cbi2(7) =
        Kokkos::complex<double>(-666638194.9266258, 282697255.4650909);
    h_ref_cbi2(8) =
        Kokkos::complex<double>(-666638194.9266258, -282697255.4650909);
    h_ref_cbi2(9)  = Kokkos::complex<double>(2.245212440929951, 0.0);
    h_ref_cbi2(10) = Kokkos::complex<double>(2.245212440929951, 0.0);
    h_ref_cbi2(11) = Kokkos::complex<double>(745818641.9833573, 0.0);
    h_ref_cbi2(12) = Kokkos::complex<double>(745818641.9833573, 0.0);
    h_ref_cbi2(13) =
        Kokkos::complex<double>(-90742126931.96704, -41134328840.59658);
    h_ref_cbi2(14) =
        Kokkos::complex<double>(-90742126931.96704, 41134328840.59658);
    h_ref_cbi2(15) =
        Kokkos::complex<double>(-90742126931.96704, 41134328840.59658);
    h_ref_cbi2(16) =
        Kokkos::complex<double>(-90742126931.96704, -41134328840.59658);
    h_ref_cbi2(17) =
        Kokkos::complex<double>(-4.977691600209507e+24, -2.705111379474063e+24);
    h_ref_cbi2(18) =
        Kokkos::complex<double>(-4.977691600209507e+24, 2.705111379474063e+24);
    h_ref_cbi2(19) =
        Kokkos::complex<double>(-4.977691600209507e+24, 2.705111379474063e+24);
    h_ref_cbi2(20) =
        Kokkos::complex<double>(-4.977691600209507e+24, -2.705111379474063e+24);
    h_ref_cbi2(21) = Kokkos::complex<double>(101851711574.0371, 0.0);
    h_ref_cbi2(22) = Kokkos::complex<double>(101851711574.0371, 0.0);
    h_ref_cbi2(23) = Kokkos::complex<double>(5.699252002663453e+24, 0.0);
    h_ref_cbi2(24) = Kokkos::complex<double>(5.699252002663453e+24, 0.0);

    h_ref_cbk2(0) = Kokkos::complex<double>(infinity<double>::value, 0);
    h_ref_cbk2(1) =
        Kokkos::complex<double>(-4.010569609868487e-02, -2.848084926389879e-02);
    h_ref_cbk2(2) =
        Kokkos::complex<double>(-4.010569609868487e-02, 2.848084926389879e-02);
    h_ref_cbk2(3) =
        Kokkos::complex<double>(-7.324740792750488, 3.979582108004995);
    h_ref_cbk2(4) =
        Kokkos::complex<double>(-7.324740792750488, -3.979582108004995);
    h_ref_cbk2(5) =
        Kokkos::complex<double>(-1.896437674343862e-11, 1.988244103510463e-11);
    h_ref_cbk2(6) =
        Kokkos::complex<double>(-1.896437674343862e-11, -1.988244103510463e-11);
    h_ref_cbk2(7) =
        Kokkos::complex<double>(8.881196209591267e+08, 2.094305655783848e+09);
    h_ref_cbk2(8) =
        Kokkos::complex<double>(8.881196209591267e+08, -2.094305655783848e+09);
    h_ref_cbk2(9) = Kokkos::complex<double>(6.151045847174205e-02, 0.0);
    h_ref_cbk2(10) =
        Kokkos::complex<double>(6.151045847174205e-02, -7.053542910173942e+00);
    h_ref_cbk2(11) = Kokkos::complex<double>(2.904495596401908e-11, 0.0);
    h_ref_cbk2(12) =
        Kokkos::complex<double>(2.904495596401908e-11, -2.343058366565231e+09);
    h_ref_cbk2(13) =
        Kokkos::complex<double>(-1.212973477697801e-13, 1.169807712685000e-13);
    h_ref_cbk2(14) =
        Kokkos::complex<double>(-1.212973477697801e-13, -1.169807712685000e-13);
    h_ref_cbk2(15) =
        Kokkos::complex<double>(129227305295.9650, 285074799340.5801);
    h_ref_cbk2(16) =
        Kokkos::complex<double>(129227305295.9650, -285074799340.5801);
    h_ref_cbk2(17) =
        Kokkos::complex<double>(-1.143205369174078e-27, 8.923553851868548e-28);
    h_ref_cbk2(18) =
        Kokkos::complex<double>(-1.143205369174078e-27, -8.923553851868548e-28);
    h_ref_cbk2(19) =
        Kokkos::complex<double>(8.498358036897867e+24, 1.563787936305381e+25);
    h_ref_cbk2(20) =
        Kokkos::complex<double>(8.498358036897867e+24, -1.563787936305381e+25);
    h_ref_cbk2(21) = Kokkos::complex<double>(1.749063165983324e-13, 0.0);
    h_ref_cbk2(22) =
        Kokkos::complex<double>(1.749063165983324e-13, -3.199765888365415e+11);
    h_ref_cbk2(23) = Kokkos::complex<double>(1.461418908109678e-27, 0.0);
    h_ref_cbk2(24) =
        Kokkos::complex<double>(1.461418908109678e-27, -1.790472822252442e+25);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbi2(i) - h_ref_cbi2(i)),
                Kokkos::abs(h_ref_cbi2(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cbk2(0), h_cbk2(0));
    for (int i = 1; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbk2(i) - h_ref_cbk2(i)),
                Kokkos::abs(h_ref_cbk2(i)) * 1e-13);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbi2(i) = Kokkos::Experimental::cyl_bessel_i(2, d_z(i));
    d_cbk2(i) = Kokkos::Experimental::cyl_bessel_k(2, d_z(i));
  }
};

template <class ExecSpace>
struct TestComplexBesselH12H22Function {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbh12, d_cbh22;
  typename ViewType::HostMirror h_z, h_cbh12, h_cbh22;
  HostViewType h_ref_cbh12, h_ref_cbh22;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N       = 25;
    d_z         = ViewType("d_z", N);
    d_cbh12     = ViewType("d_cbh12", N);
    d_cbh22     = ViewType("d_cbh22", N);
    h_z         = Kokkos::create_mirror_view(d_z);
    h_cbh12     = Kokkos::create_mirror_view(d_cbh12);
    h_cbh22     = Kokkos::create_mirror_view(d_cbh22);
    h_ref_cbh12 = HostViewType("h_ref_cbh12", N);
    h_ref_cbh22 = HostViewType("h_ref_cbh22", N);

    // Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0, 0.0);
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0, -2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0, -2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0, -10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0, -10.0);
    h_z(9)  = Kokkos::complex<double>(3.0, 0.0);
    h_z(10) = Kokkos::complex<double>(-3.0, 0.0);
    h_z(11) = Kokkos::complex<double>(23.0, 0.0);
    h_z(12) = Kokkos::complex<double>(-23.0, 0.0);
    // abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0, -10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0, -10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0, -10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0, -10.0);
    h_z(21) = Kokkos::complex<double>(28.0, 0.0);
    h_z(22) = Kokkos::complex<double>(-28.0, 0.0);
    h_z(23) = Kokkos::complex<double>(60.0, 0.0);
    h_z(24) = Kokkos::complex<double>(-60.0, 0.0);
    Kokkos::deep_copy(d_z, h_z);

    // Call Bessel functions
#if (HIP_VERSION_MAJOR == 5) && (HIP_VERSION_MINOR == 4)
    using Property =
        Kokkos::Experimental::WorkItemProperty::ImplForceGlobalLaunch_t;
#else
    using Property = Kokkos::Experimental::WorkItemProperty::None_t;
#endif
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, Property>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbh12, d_cbh12);
    Kokkos::deep_copy(h_cbh22, d_cbh22);

    // Reference values computed with Octave
    h_ref_cbh12(0) = Kokkos::complex<double>(0, 0);
    h_ref_cbh12(1) =
        Kokkos::complex<double>(5.086055468267858e-02, -5.828607326644407e-02);
    h_ref_cbh12(2) =
        Kokkos::complex<double>(2.391757643073725e+00, -3.101786180357444e-01);
    h_ref_cbh12(3) =
        Kokkos::complex<double>(-5.086055468267858e-02, -5.828607326644407e-02);
    h_ref_cbh12(4) =
        Kokkos::complex<double>(2.493478752439082e+00, 1.936064715028563e-01);
    h_ref_cbh12(5) =
        Kokkos::complex<double>(7.432045366267922e-06, 6.396322794545899e-07);
    h_ref_cbh12(6) =
        Kokkos::complex<double>(2995.367201837851, 1629.233773183346);
    h_ref_cbh12(7) =
        Kokkos::complex<double>(-7.432045366267922e-06, 6.396322794545899e-07);
    h_ref_cbh12(8) =
        Kokkos::complex<double>(2995.367216701941, -1629.233771904082);
    h_ref_cbh12(9) =
        Kokkos::complex<double>(4.860912605858910e-01, -1.604003934849236e-01);
    h_ref_cbh12(10) =
        Kokkos::complex<double>(-4.860912605858910e-01, -1.604003934849236e-01);
    h_ref_cbh12(11) =
        Kokkos::complex<double>(1.589763185409907e-01, 5.003998166436658e-02);
    h_ref_cbh12(12) =
        Kokkos::complex<double>(-1.589763185409907e-01, 5.003998166436658e-02);
    h_ref_cbh12(13) =
        Kokkos::complex<double>(2.633736063050771e-06, -6.259675757095957e-06);
    h_ref_cbh12(14) =
        Kokkos::complex<double>(2135.575940675463, -2326.524824036628);
    h_ref_cbh12(15) =
        Kokkos::complex<double>(-2.633736063050771e-06, -6.259675757095957e-06);
    h_ref_cbh12(16) =
        Kokkos::complex<double>(2135.575945942935, 2326.524811517277);
    h_ref_cbh12(17) =
        Kokkos::complex<double>(4.033605096112077e-06, -2.352053210899959e-06);
    h_ref_cbh12(18) =
        Kokkos::complex<double>(2.096716238742065e+03, -7.954351548668977e+02);
    h_ref_cbh12(19) =
        Kokkos::complex<double>(-4.033605096112077e-06, -2.352053210899959e-06);
    h_ref_cbh12(20) =
        Kokkos::complex<double>(2.096716246809275e+03, 7.954351501627914e+02);
    h_ref_cbh12(21) =
        Kokkos::complex<double>(8.248211685864913e-02, -1.264420328105134e-01);
    h_ref_cbh12(22) =
        Kokkos::complex<double>(-8.248211685864913e-02, -1.264420328105134e-01);
    h_ref_cbh12(23) =
        Kokkos::complex<double>(9.302508354766741e-02, -4.429663189712049e-02);
    h_ref_cbh12(24) =
        Kokkos::complex<double>(-9.302508354766741e-02, -4.429663189712049e-02);

    h_ref_cbh22(0) = Kokkos::complex<double>(0, 0);

    h_ref_cbh22(1) =
        Kokkos::complex<double>(2.391757643073725e+00, 3.101786180357444e-01);
    h_ref_cbh22(2) =
        Kokkos::complex<double>(5.086055468267858e-02, 5.828607326644407e-02);
    h_ref_cbh22(3) =
        Kokkos::complex<double>(2.493478752439082e+00, -1.936064715028563e-01);
    h_ref_cbh22(4) =
        Kokkos::complex<double>(-5.086055468267858e-02, 5.828607326644407e-02);
    h_ref_cbh22(5) =
        Kokkos::complex<double>(2995.367201837851, -1629.233773183346);
    h_ref_cbh22(6) =
        Kokkos::complex<double>(7.432045366267922e-06, -6.396322794545899e-07);
    h_ref_cbh22(7) =
        Kokkos::complex<double>(2995.367216701941, 1629.233771904082);
    h_ref_cbh22(8) =
        Kokkos::complex<double>(-7.432045366267922e-06, -6.396322794545899e-07);
    h_ref_cbh22(9) =
        Kokkos::complex<double>(4.860912605858910e-01, 1.604003934849236e-01);
    h_ref_cbh22(10) =
        Kokkos::complex<double>(1.458273781757673e+00, 1.604003934849235e-01);
    h_ref_cbh22(11) =
        Kokkos::complex<double>(1.589763185409907e-01, -5.003998166436658e-02);
    h_ref_cbh22(12) =
        Kokkos::complex<double>(4.769289556229723e-01, -5.003998166436656e-02);
    h_ref_cbh22(13) =
        Kokkos::complex<double>(2135.575940675463, 2326.524824036628);
    h_ref_cbh22(14) =
        Kokkos::complex<double>(2.633736063050771e-06, 6.259675757095957e-06);
    h_ref_cbh22(15) =
        Kokkos::complex<double>(2135.575945942935, -2326.524811517277);
    h_ref_cbh22(16) =
        Kokkos::complex<double>(-2.633736063050771e-06, 6.259675757095957e-06);
    h_ref_cbh22(17) =
        Kokkos::complex<double>(2.096716238742065e+03, 7.954351548668977e+02);
    h_ref_cbh22(18) =
        Kokkos::complex<double>(4.033605096112077e-06, 2.352053210899959e-06);
    h_ref_cbh22(19) =
        Kokkos::complex<double>(2.096716246809275e+03, -7.954351501627914e+02);
    h_ref_cbh22(20) =
        Kokkos::complex<double>(-4.033605096112077e-06, 2.352053210899959e-06);
    h_ref_cbh22(21) =
        Kokkos::complex<double>(8.248211685864913e-02, 1.264420328105134e-01);
    h_ref_cbh22(22) =
        Kokkos::complex<double>(2.474463505759475e-01, 1.264420328105134e-01);
    h_ref_cbh22(23) =
        Kokkos::complex<double>(9.302508354766741e-02, 4.429663189712049e-02);
    h_ref_cbh22(24) =
        Kokkos::complex<double>(2.790752506430023e-01, 4.429663189712047e-02);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbh12(i) - h_ref_cbh12(i)),
                Kokkos::abs(h_ref_cbh12(i)) * 1e-13);
    }

    EXPECT_EQ(h_ref_cbh22(0), h_cbh22(0));
    for (int i = 1; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbh22(i) - h_ref_cbh22(i)),
                Kokkos::abs(h_ref_cbh22(i)) * 1e-13);
    }
  }
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const {
    d_cbh12(i) = Kokkos::Experimental::cyl_bessel_h1(2, d_z(i));
    d_cbh22(i) = Kokkos::Experimental::cyl_bessel_h2(2, d_z(i));
  }
};
template <class ExecSpace>
struct TestComplexBesselFunctionsWithLargeIndexAndLargeArguments {
  using ViewType = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbj8, d_cbj9;
  typename ViewType::HostMirror h_z, h_cbj8, h_cbj9;
  HostViewType h_ref_cbj8, h_ref_cbj9;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N      = 5;
    d_z        = ViewType("d_z", N);
    d_cbj8     = ViewType("d_cbj8", N);
    d_cbj9     = ViewType("d_cbj9", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbj8     = Kokkos::create_mirror_view(d_cbj8);
    h_cbj9     = Kokkos::create_mirror_view(d_cbj9);
    h_ref_cbj8 = HostViewType("h_ref_cbj8", N);
    h_ref_cbj9 = HostViewType("h_ref_cbj8", N);
    // Generate test inputs
    h_z(0) = Kokkos::complex<double>(10.0, 8.0);
    h_z(1) = Kokkos::complex<double>(512.0, 0.0);
    h_z(2) = Kokkos::complex<double>(300.0, -200.0);
    h_z(3) = Kokkos::complex<double>(-125.0, 600.0);
    h_z(4) = Kokkos::complex<double>(1000.0, 200.0);

    Kokkos::deep_copy(d_z, h_z);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj8, d_cbj8);
    Kokkos::deep_copy(h_cbj9, d_cbj9);

    h_ref_cbj8(0) =
        Kokkos::complex<double>(35.14290937115415, 55.89195296301610);
    h_ref_cbj8(1) = Kokkos::complex<double>(-2.449916135262541e-02, 0.0);
    h_ref_cbj8(2) =
        Kokkos::complex<double>(-6.155013136843385e+84, -1.307999884019452e+85);
    h_ref_cbj8(3) = Kokkos::complex<double>(4.204904284127386e+258,
                                            -3.962917327640782e+258);
    h_ref_cbj8(4) =
        Kokkos::complex<double>(8.523650188274974e+84, -2.800930999646528e+84);

    h_ref_cbj9(0) =
        Kokkos::complex<double>(-19.74435372008063, 36.71222568036936);
    h_ref_cbj9(1) = Kokkos::complex<double>(2.495433941009553e-02, 0.0);
    h_ref_cbj9(2) =
        Kokkos::complex<double>(-1.302704966918034e+85, 5.820333237321211e+84);
    h_ref_cbj9(3) =
        Kokkos::complex<double>(3.897668823461053e+258, 4.159215325012759e+258);
    h_ref_cbj9(4) =
        Kokkos::complex<double>(2.865837445663514e+84, 8.486616730935941e+84);

    for (int i = 0; i < N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbj8(i) - h_ref_cbj8(i)),
                Kokkos::abs(h_ref_cbj8(i)) * 1e-13);
      EXPECT_LE(Kokkos::abs(h_cbj9(i) - h_ref_cbj9(i)),
                Kokkos::abs(h_ref_cbj9(i)) * 1e-13);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& /*i*/) const {
    d_cbj8(0) = Kokkos::Experimental::cyl_bessel_j(8, d_z(0));
    d_cbj8(1) = Kokkos::Experimental::cyl_bessel_j(8, d_z(1));
    d_cbj8(2) = Kokkos::Experimental::cyl_bessel_j(8, d_z(2));
    d_cbj8(3) = Kokkos::Experimental::cyl_bessel_j(8, d_z(3));
    d_cbj8(4) = Kokkos::Experimental::cyl_bessel_j(8, d_z(4));

    d_cbj9(0) = Kokkos::Experimental::cyl_bessel_j(9, d_z(0));
    d_cbj9(1) = Kokkos::Experimental::cyl_bessel_j(9, d_z(1));
    d_cbj9(2) = Kokkos::Experimental::cyl_bessel_j(9, d_z(2));
    d_cbj9(3) = Kokkos::Experimental::cyl_bessel_j(9, d_z(3));
    d_cbj9(4) = Kokkos::Experimental::cyl_bessel_j(9, d_z(4));
  }
};

template <class ExecSpace>
struct TestBesselCallsWithDeduction {
  using ViewType     = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using ViewRealType = Kokkos::View<double*, ExecSpace>;
  using HostViewType =
      Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;
  using HostViewRealType = Kokkos::View<double*, Kokkos::HostSpace>;

  ViewType d_z, d_cb;
  ViewRealType d_r, d_rb;
  typename ViewType::HostMirror h_z, h_cb;
  typename ViewRealType::HostMirror h_r, h_rb;
  HostViewType h_ref_cb;
  HostViewRealType h_ref_rb;

  void testit() {
    int N_samples = 2;
    int N_complex = N_samples * 6 * 2;  // two indices for each bessel
    int N_real    = N_samples * 4 * 2;  // two indices for each bessel
    d_z           = ViewType("d_z", N_samples);
    d_cb          = ViewType("d_cb", N_complex);
    d_r           = ViewRealType("d_r", N_samples);
    d_rb          = ViewRealType("d_rb", N_real);
    h_z           = Kokkos::create_mirror_view(d_z);
    h_cb          = Kokkos::create_mirror_view(d_cb);
    h_r           = Kokkos::create_mirror_view(d_r);
    h_rb          = Kokkos::create_mirror_view(d_rb);
    h_ref_cb      = HostViewType("h_ref_cb", N_complex);
    h_ref_rb      = HostViewRealType("h_ref_rb", N_real);
    // Generate test input
    h_z(0) = Kokkos::complex<double>(2.0, 1.0);
    h_z(1) = Kokkos::complex<double>(1.0, 2.0);
    h_r(0) = 1.5;
    h_r(1) = 2.0;

    Kokkos::deep_copy(d_z, h_z);
    Kokkos::deep_copy(d_r, h_r);
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 1), *this);
    Kokkos::fence();
    Kokkos::deep_copy(h_cb, d_cb);
    Kokkos::deep_copy(h_rb, d_rb);

    h_ref_cb(0) =
        Kokkos::complex<double>(-2.134746302848219E-03, 1.178707242055155E-03);
    h_ref_cb(1) =
        Kokkos::complex<double>(2.949011783284490E-03, 6.555671631487109E-04);
    h_ref_cb(2) =
        Kokkos::complex<double>(-3.945548817818670E-04, 8.828053172246283E-06);
    h_ref_cb(3) =
        Kokkos::complex<double>(1.074313869455874E-04, 4.637365649395186E-04);
    h_ref_cb(4) =
        Kokkos::complex<double>(-2.949011783284490E-03, 6.555671631487109E-04);
    h_ref_cb(5) =
        Kokkos::complex<double>(2.134746302848219E-03, 1.178707242055155E-03);
    h_ref_cb(6) =
        Kokkos::complex<double>(-4.637365649395186E-04, -1.074313869455874E-04);
    h_ref_cb(7) =
        Kokkos::complex<double>(-8.828053172246260E-06, 3.945548817818670E-04);
    h_ref_cb(8) =
        Kokkos::complex<double>(-2.603425050549171E+01, -4.384521266763712E+00);
    h_ref_cb(9) =
        Kokkos::complex<double>(3.000607145820604E+01, -1.912077136644432E+01);
    h_ref_cb(10) =
        Kokkos::complex<double>(-1.402157222471133E+02, 3.828717669039141E+01);
    h_ref_cb(11) =
        Kokkos::complex<double>(-1.248245425959870E+01, -1.861086169343677E+02);
    h_ref_cb(12) =
        Kokkos::complex<double>(1.910127967412645E+01, 1.217052636849550E+01);
    h_ref_cb(13) =
        Kokkos::complex<double>(-1.657457419772987E+01, 2.794221942571219E+00);
    h_ref_cb(14) =
        Kokkos::complex<double>(1.184804165203494E+02, 7.946182634452692E+00);
    h_ref_cb(15) =
        Kokkos::complex<double>(-2.437483744579928E+01, 8.926420861070027E+01);
    h_ref_cb(16) =
        Kokkos::complex<double>(-1.217266111479835E+01, 1.910245838136851E+01);
    h_ref_cb(17) =
        Kokkos::complex<double>(-2.791272930787934E+00, -1.657391863056672E+01);
    h_ref_cb(18) =
        Kokkos::complex<double>(-7.946577189334481E+00, 1.184804253484026E+02);
    h_ref_cb(19) =
        Kokkos::complex<double>(-8.926410117931331E+01, -2.437437370923435E+01);
    h_ref_cb(20) =
        Kokkos::complex<double>(1.216839162219265E+01, -1.910010096688440E+01);
    h_ref_cb(21) =
        Kokkos::complex<double>(2.797170954354503E+00, 1.657522976489302E+01);
    h_ref_cb(22) =
        Kokkos::complex<double>(7.945788079570903E+00, -1.184804076922962E+02);
    h_ref_cb(23) =
        Kokkos::complex<double>(8.926431604208722E+01, 2.437530118236421E+01);

    h_ref_rb(0)  = 0.00022801269539361233;
    h_ref_rb(1)  = 0.001202428971789993;
    h_ref_rb(2)  = 2.467979578828794e-5;
    h_ref_rb(3)  = 0.00017494407486827413;
    h_ref_rb(4)  = 0.0002677691439944764;
    h_ref_rb(5)  = 0.001600173363521726;
    h_ref_rb(6)  = 2.8406417141745867e-5;
    h_ref_rb(7)  = 0.0002246391420013424;
    h_ref_rb(8)  = 301.7040785018287;
    h_ref_rb(9)  = 49.3511614303943;
    h_ref_rb(10) = 2457.70040917393;
    h_ref_rb(11) = 305.5380176829622;
    h_ref_rb(12) = -240.5734174668407;
    h_ref_rb(13) = -46.91400241607929;
    h_ref_rb(14) = -1887.3970313392276;
    h_ref_rb(15) = -271.5480253679938;

    for (int i = 0; i < N_complex; i++) {
      EXPECT_LE(Kokkos::abs(h_cb(i) - h_ref_cb(i)),
                Kokkos::abs(h_ref_cb(i)) * 1e-9);  // TODO
    }

    for (int i = 0; i < N_real; i++) {
      EXPECT_LE(Kokkos::abs(h_rb(i) - h_ref_rb(i)),
                Kokkos::abs(h_ref_rb(i)) * 1e-7);  // TODO
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& /*i*/) const {
    d_cb(0) = Kokkos::Experimental::cyl_bessel_j(6, d_z(0));
    d_cb(1) = Kokkos::Experimental::cyl_bessel_j(6, d_z(1));
    d_cb(2) = Kokkos::Experimental::cyl_bessel_j(7, d_z(0));
    d_cb(3) = Kokkos::Experimental::cyl_bessel_j(7, d_z(1));

    d_cb(4) = Kokkos::Experimental::cyl_bessel_i(6, d_z(0));
    d_cb(5) = Kokkos::Experimental::cyl_bessel_i(6, d_z(1));
    d_cb(6) = Kokkos::Experimental::cyl_bessel_i(7, d_z(0));
    d_cb(7) = Kokkos::Experimental::cyl_bessel_i(7, d_z(1));

    d_cb(8)  = Kokkos::Experimental::cyl_bessel_k(6, d_z(0));
    d_cb(9)  = Kokkos::Experimental::cyl_bessel_k(6, d_z(1));
    d_cb(10) = Kokkos::Experimental::cyl_bessel_k(7, d_z(0));
    d_cb(11) = Kokkos::Experimental::cyl_bessel_k(7, d_z(1));

    d_cb(12) = Kokkos::Experimental::cyl_bessel_y(6, d_z(0));
    d_cb(13) = Kokkos::Experimental::cyl_bessel_y(6, d_z(1));
    d_cb(14) = Kokkos::Experimental::cyl_bessel_y(7, d_z(0));
    d_cb(15) = Kokkos::Experimental::cyl_bessel_y(7, d_z(1));

    d_cb(16) = Kokkos::Experimental::cyl_bessel_h1(6, d_z(0));
    d_cb(17) = Kokkos::Experimental::cyl_bessel_h1(6, d_z(1));
    d_cb(18) = Kokkos::Experimental::cyl_bessel_h1(7, d_z(0));
    d_cb(19) = Kokkos::Experimental::cyl_bessel_h1(7, d_z(1));

    d_cb(20) = Kokkos::Experimental::cyl_bessel_h2(6, d_z(0));
    d_cb(21) = Kokkos::Experimental::cyl_bessel_h2(6, d_z(1));
    d_cb(22) = Kokkos::Experimental::cyl_bessel_h2(7, d_z(0));
    d_cb(23) = Kokkos::Experimental::cyl_bessel_h2(7, d_z(1));

    // Call to vendor functions of possible and ret real type
    d_rb(0) = Kokkos::Experimental::cyl_bessel_j(6, d_r(0));
    d_rb(1) = Kokkos::Experimental::cyl_bessel_j(6, d_r(1));
    d_rb(2) = Kokkos::Experimental::cyl_bessel_j(7, d_r(0));
    d_rb(3) = Kokkos::Experimental::cyl_bessel_j(7, d_r(1));

    d_rb(4) = Kokkos::Experimental::cyl_bessel_i(6, d_r(0));
    d_rb(5) = Kokkos::Experimental::cyl_bessel_i(6, d_r(1));
    d_rb(6) = Kokkos::Experimental::cyl_bessel_i(7, d_r(0));
    d_rb(7) = Kokkos::Experimental::cyl_bessel_i(7, d_r(1));

    d_rb(8)  = Kokkos::Experimental::cyl_bessel_k(6, d_r(0));
    d_rb(9)  = Kokkos::Experimental::cyl_bessel_k(6, d_r(1));
    d_rb(10) = Kokkos::Experimental::cyl_bessel_k(7, d_r(0));
    d_rb(11) = Kokkos::Experimental::cyl_bessel_k(7, d_r(1));

    d_rb(12) = Kokkos::Experimental::cyl_bessel_y(6, d_r(0));
    d_rb(13) = Kokkos::Experimental::cyl_bessel_y(6, d_r(1));
    d_rb(14) = Kokkos::Experimental::cyl_bessel_y(7, d_r(0));
    d_rb(15) = Kokkos::Experimental::cyl_bessel_y(7, d_r(1));
  }
};

TEST(TEST_CATEGORY, mathspecialfunc_expint1) {
  TestExponentialIntergral1Function<TEST_EXECSPACE> test;
  test.testit();
}

// FIXME_OPENMPTARGET: This unit test fails with a misaligned address error at
// runtime with LLVM/13.
#ifndef KOKKOS_ENABLE_OPENMPTARGET
TEST(TEST_CATEGORY, mathspecialfunc_errorfunc) {
  TestComplexErrorFunction<TEST_EXECSPACE> test;
  test.testit();
}
#endif

TEST(TEST_CATEGORY, mathspecialfunc_cbesselj0y0) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselJ0Y0Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselj1y1) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselJ1Y1Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesseli0k0) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselI0K0Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesseli1k1) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselI1K1Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselh1stkind) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselH1Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselh2ndkind) {
#if defined(KOKKOS_ENABLE_OPENMPTARGET) && defined(KOKKOS_ARCH_INTEL_GPU)
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since test is known to fail with OpenMPTarget on "
                    "Intel GPUs";  // FIXME_OPENMPTARGET
#endif
  TestComplexBesselH2Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselj2y2) {
  TestComplexBesselJ2Y2Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesseli2k2) {
  TestComplexBesselI2K2Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselh12h22) {
  TestComplexBesselH12H22Function<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_largeindexesandargs) {
  TestComplexBesselFunctionsWithLargeIndexAndLargeArguments<TEST_EXECSPACE>
      test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_besseldeduction) {
  TestBesselCallsWithDeduction<TEST_EXECSPACE> test;
  test.testit();
}
}  // namespace Test
