#include <fstream>
#include <gtest/gtest.h>
#include "Kokkos_Core.hpp"

namespace Test {
	
struct TestLargeArgTag {};

template <class ExecSpace>
struct TestExponentialIntergralFunction {
  using ViewType     = Kokkos::View<double*, ExecSpace>;
  using HostViewType = Kokkos::View<double*, Kokkos::HostSpace>;

  ViewType d_x, d_expint;
  typename ViewType::HostMirror h_x, h_expint;
  HostViewType h_ref;

  void testit() {
    using Kokkos::Experimental::infinity;
    using Kokkos::Experimental::fabs;

    d_x      = ViewType("d_x", 15);
    d_expint = ViewType("d_expint", 15);
    h_x      = Kokkos::create_mirror_view(d_x);
    h_expint = Kokkos::create_mirror_view(d_expint);
    h_ref    = HostViewType("h_ref", 15);

    //Generate test inputs
    h_x(0)  = -0.2;
    h_x(1)  =  0.0;
    h_x(2)  =  0.2;
    h_x(3)  =  0.8;
    h_x(4)  =  1.6;
    h_x(5)  =  5.1;
    h_x(6)  =  0.01;
    h_x(7)  =  0.001;
    h_x(8)  =  1.0;
    h_x(9)  =  1.001;
    h_x(10) =  1.01;
    h_x(11) =  1.1;
    h_x(12) =  7.2;
    h_x(13) =  10.3;
    h_x(14) =  15.4;
    Kokkos::deep_copy(d_x, h_x);

    //Call exponential integral function
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 15), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_expint, d_expint);

    //Reference values computed with Octave
    h_ref(0) = -infinity<double>::value;//x(0)=-0.2
    h_ref(1) =  infinity<double>::value;//x(1)= 0.0
    h_ref(2) = 1.222650544183893e+00;//x(2) =0.2
    h_ref(3) = 3.105965785455429e-01;//x(3) =0.8
    h_ref(4) = 8.630833369753976e-02;//x(4) =1.6 
    h_ref(5) = 1.021300107861738e-03;//x(5) =5.1
    h_ref(6) = 4.037929576538113e+00;//x(6) =0.01
    h_ref(7) = 6.331539364136149e+00;//x(7) =0.001
    h_ref(8) = 2.193839343955205e-01;//x(8) =1.0
    h_ref(9) = 2.190164225274689e-01;//x(9) =1.001
    h_ref(10)= 2.157416237944899e-01;//x(10)=1.01
    h_ref(11)= 1.859909045360401e-01;//x(11)=1.1
    h_ref(12)= 9.218811688716196e-05;//x(12)=7.2
    h_ref(13)= 2.996734771597901e-06;//x(13)=10.3
    h_ref(14)= 1.254522935050609e-08;//x(14)=15.4

    EXPECT_EQ(h_expint(0), h_ref(0));
    EXPECT_EQ(h_expint(1), h_ref(1));
    for (int i=2; i<15; i++) {
      EXPECT_LE(fabs(h_expint(i) - h_ref(i)), fabs(h_ref(i))*1e-15);
      //printf("%d. %.15e vs. %.15e\n",i, fabs(h_expint(i) - h_ref(i)), fabs(h_ref(i))*1e-15);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & i) const {
    d_expint(i)  = Kokkos::Experimental::expint(d_x(i));
  }
};

template <class ExecSpace>
struct TestComplexErrorFunction {
  using ViewType     = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType = Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_erf, d_erfcx;
  typename ViewType::HostMirror h_z, h_erf, h_erfcx;
  HostViewType h_ref_erf, h_ref_erfcx;

  void testit() {
    d_z     = ViewType("d_z", 52);
    d_erf   = ViewType("d_erf", 52);
    d_erfcx = ViewType("d_erfcx", 52);
    h_z     = Kokkos::create_mirror_view(d_z);
    h_erf   = Kokkos::create_mirror_view(d_erf);
    h_erfcx = Kokkos::create_mirror_view(d_erfcx);
    h_ref_erf   = HostViewType("h_ref_erf", 52);
    h_ref_erfcx = HostViewType("h_ref_erfcx", 52);

    //Generate test inputs
    //abs(z)<=2
    h_z(0)  = Kokkos::complex<double>( 0.0011, 0);
    h_z(1)  = Kokkos::complex<double>(-0.0011, 0);
    h_z(2)  = Kokkos::complex<double>( 1.4567, 0);
    h_z(3)  = Kokkos::complex<double>(-1.4567, 0);
    h_z(4)  = Kokkos::complex<double>(0,  0.0011);
    h_z(5)  = Kokkos::complex<double>(0, -0.0011);
    h_z(6)  = Kokkos::complex<double>(0,  1.4567);
    h_z(7)  = Kokkos::complex<double>(0, -1.4567);
    h_z(8)  = Kokkos::complex<double>( 1.4567,  0.0011);
    h_z(9)  = Kokkos::complex<double>( 1.4567, -0.0011);
    h_z(10) = Kokkos::complex<double>(-1.4567,  0.0011);
    h_z(11) = Kokkos::complex<double>(-1.4567, -0.0011);
    h_z(12) = Kokkos::complex<double>( 1.4567,  0.5942);
    h_z(13) = Kokkos::complex<double>( 1.4567, -0.5942);
    h_z(14) = Kokkos::complex<double>(-1.4567,  0.5942);
    h_z(15) = Kokkos::complex<double>(-1.4567, -0.5942);
    h_z(16) = Kokkos::complex<double>( 0.0011,  0.5942);
    h_z(17) = Kokkos::complex<double>( 0.0011, -0.5942);
    h_z(18) = Kokkos::complex<double>(-0.0011,  0.5942);
    h_z(19) = Kokkos::complex<double>(-0.0011, -0.5942);
    h_z(20) = Kokkos::complex<double>( 0.0011,  0.0051);
    h_z(21) = Kokkos::complex<double>( 0.0011, -0.0051);
    h_z(22) = Kokkos::complex<double>(-0.0011,  0.0051);
    h_z(23) = Kokkos::complex<double>(-0.0011, -0.0051);
    //abs(z)>2.0 and x>1
    h_z(24) = Kokkos::complex<double>( 3.5,  0.0011);
    h_z(25) = Kokkos::complex<double>( 3.5, -0.0011);
    h_z(26) = Kokkos::complex<double>(-3.5,  0.0011);
    h_z(27) = Kokkos::complex<double>(-3.5, -0.0011);
    h_z(28) = Kokkos::complex<double>( 3.5,  9.7);
    h_z(29) = Kokkos::complex<double>( 3.5, -9.7);
    h_z(30) = Kokkos::complex<double>(-3.5,  9.7);
    h_z(31) = Kokkos::complex<double>(-3.5, -9.7);
    h_z(32) = Kokkos::complex<double>( 18.9,  9.7);
    h_z(33) = Kokkos::complex<double>( 18.9, -9.7);
    h_z(34) = Kokkos::complex<double>(-18.9,  9.7);
    h_z(35) = Kokkos::complex<double>(-18.9, -9.7);
    //abs(z)>2.0 and 0<=x<=1 and abs(y)<6
    h_z(36) = Kokkos::complex<double>( 0.85,  3.5);
    h_z(37) = Kokkos::complex<double>( 0.85, -3.5);
    h_z(38) = Kokkos::complex<double>(-0.85,  3.5);
    h_z(39) = Kokkos::complex<double>(-0.85, -3.5);
    h_z(40) = Kokkos::complex<double>( 0.0011,  3.5);
    h_z(41) = Kokkos::complex<double>( 0.0011, -3.5);
    h_z(42) = Kokkos::complex<double>(-0.0011,  3.5);
    h_z(43) = Kokkos::complex<double>(-0.0011, -3.5);
    //abs(z)>2.0 and 0<=x<=1 and abs(y)>=6
    h_z(44) = Kokkos::complex<double>( 0.85,  7.5);
    h_z(45) = Kokkos::complex<double>( 0.85, -7.5);
    h_z(46) = Kokkos::complex<double>(-0.85,  7.5);
    h_z(47) = Kokkos::complex<double>(-0.85, -7.5);
    h_z(48) = Kokkos::complex<double>( 0.85,  19.7);
    h_z(49) = Kokkos::complex<double>( 0.85, -19.7);
    h_z(50) = Kokkos::complex<double>(-0.85,  19.7);
    h_z(51) = Kokkos::complex<double>(-0.85, -19.7);

    Kokkos::deep_copy(d_z, h_z);

    //Call erf and erfcx functions
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, 52), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_erf, d_erf);
    Kokkos::deep_copy(h_erfcx, d_erfcx);

    //Reference values computed with Octave
    h_ref_erf(0)  = Kokkos::complex<double>( 0.001241216583181022, 0);
    h_ref_erf(1)  = Kokkos::complex<double>(-0.001241216583181022, 0);
    h_ref_erf(2)  = Kokkos::complex<double>( 0.9606095744865353, 0);
    h_ref_erf(3)  = Kokkos::complex<double>(-0.9606095744865353, 0);
    h_ref_erf(4)  = Kokkos::complex<double>(0,  0.001241217584429469);
    h_ref_erf(5)  = Kokkos::complex<double>(0, -0.001241217584429469);
    h_ref_erf(6)  = Kokkos::complex<double>(0,  4.149756424218223);
    h_ref_erf(7)  = Kokkos::complex<double>(0, -4.149756424218223);
    h_ref_erf(8)  = Kokkos::complex<double>( 0.960609812745064,  0.0001486911741082233);
    h_ref_erf(9)  = Kokkos::complex<double>( 0.960609812745064, -0.0001486911741082233);
    h_ref_erf(10) = Kokkos::complex<double>(-0.960609812745064,  0.0001486911741082233);
    h_ref_erf(11) = Kokkos::complex<double>(-0.960609812745064, -0.0001486911741082233);
    h_ref_erf(12) = Kokkos::complex<double>( 1.02408827958197,  0.04828570635603527);
    h_ref_erf(13) = Kokkos::complex<double>( 1.02408827958197, -0.04828570635603527);
    h_ref_erf(14) = Kokkos::complex<double>(-1.02408827958197,  0.04828570635603527);
    h_ref_erf(15) = Kokkos::complex<double>(-1.02408827958197, -0.04828570635603527);
    h_ref_erf(16) = Kokkos::complex<double>( 0.001766791817179109,  0.7585038120712589);
    h_ref_erf(17) = Kokkos::complex<double>( 0.001766791817179109, -0.7585038120712589);
    h_ref_erf(18) = Kokkos::complex<double>(-0.001766791817179109,  0.7585038120712589);
    h_ref_erf(19) = Kokkos::complex<double>(-0.001766791817179109, -0.7585038120712589);
    h_ref_erf(20) = Kokkos::complex<double>( 0.001241248867618165,  0.005754776682713324);
    h_ref_erf(21) = Kokkos::complex<double>( 0.001241248867618165, -0.005754776682713324);
    h_ref_erf(22) = Kokkos::complex<double>(-0.001241248867618165,  0.005754776682713324);
    h_ref_erf(23) = Kokkos::complex<double>(-0.001241248867618165, -0.005754776682713324);
    h_ref_erf(24) = Kokkos::complex<double>( 0.9999992569244941,  5.939313159932013e-09);
    h_ref_erf(25) = Kokkos::complex<double>( 0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(26) = Kokkos::complex<double>(-0.9999992569244941,  5.939313159932013e-09);
    h_ref_erf(27) = Kokkos::complex<double>(-0.9999992569244941, -5.939313159932013e-09);
    h_ref_erf(28) = Kokkos::complex<double>(-1.915595842013002e+34,  1.228821279117683e+32);
    h_ref_erf(29) = Kokkos::complex<double>(-1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(30) = Kokkos::complex<double>( 1.915595842013002e+34,  1.228821279117683e+32);
    h_ref_erf(31) = Kokkos::complex<double>( 1.915595842013002e+34, -1.228821279117683e+32);
    h_ref_erf(32) = Kokkos::complex<double>( 1,  5.959897539826596e-117);
    h_ref_erf(33) = Kokkos::complex<double>( 1, -5.959897539826596e-117);
    h_ref_erf(34) = Kokkos::complex<double>(-1,  5.959897539826596e-117);
    h_ref_erf(35) = Kokkos::complex<double>(-1, -5.959897539826596e-117);
    h_ref_erf(36) = Kokkos::complex<double>(-9211.077162784413,  13667.93825589455);
    h_ref_erf(37) = Kokkos::complex<double>(-9211.077162784413, -13667.93825589455);
    h_ref_erf(38) = Kokkos::complex<double>( 9211.077162784413,  13667.93825589455);
    h_ref_erf(39) = Kokkos::complex<double>( 9211.077162784413, -13667.93825589455);
    h_ref_erf(40) = Kokkos::complex<double>( 259.38847811225,  35281.28906479814);
    h_ref_erf(41) = Kokkos::complex<double>( 259.38847811225, -35281.28906479814);
    h_ref_erf(42) = Kokkos::complex<double>(-259.38847811225,  35281.28906479814);
    h_ref_erf(43) = Kokkos::complex<double>(-259.38847811225, -35281.28906479814);
    h_ref_erf(44) = Kokkos::complex<double>( 6.752085728270252e+21,  9.809477366939276e+22);
    h_ref_erf(45) = Kokkos::complex<double>( 6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(46) = Kokkos::complex<double>(-6.752085728270252e+21,  9.809477366939276e+22);
    h_ref_erf(47) = Kokkos::complex<double>(-6.752085728270252e+21, -9.809477366939276e+22);
    h_ref_erf(48) = Kokkos::complex<double>( 4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(49) = Kokkos::complex<double>( 4.37526734926942e+166,  2.16796709605852e+166);
    h_ref_erf(50) = Kokkos::complex<double>(-4.37526734926942e+166, -2.16796709605852e+166);
    h_ref_erf(51) = Kokkos::complex<double>(-4.37526734926942e+166,  2.16796709605852e+166);

    h_ref_erfcx(0)  = Kokkos::complex<double>(0.9987599919156778, 0);
    h_ref_erfcx(1)  = Kokkos::complex<double>(1.001242428085786,  0);
    h_ref_erfcx(2)  = Kokkos::complex<double>(0.3288157848563544, 0);
    h_ref_erfcx(3)  = Kokkos::complex<double>(16.36639786516915,  0);
    h_ref_erfcx(4)  = Kokkos::complex<double>(0.999998790000732, -0.001241216082557101);
    h_ref_erfcx(5)  = Kokkos::complex<double>(0.999998790000732,  0.001241216082557101);
    h_ref_erfcx(6)  = Kokkos::complex<double>(0.1197948131677216, -0.4971192955307743);
    h_ref_erfcx(7)  = Kokkos::complex<double>(0.1197948131677216,  0.4971192955307743);
    h_ref_erfcx(8)  = Kokkos::complex<double>(0.3288156873503045, -0.0001874479383970247);
    h_ref_erfcx(9)  = Kokkos::complex<double>(0.3288156873503045,  0.0001874479383970247);
    h_ref_erfcx(10) = Kokkos::complex<double>(16.36629202874158, -0.05369111060785572);
    h_ref_erfcx(11) = Kokkos::complex<double>(16.36629202874158,  0.05369111060785572);
    h_ref_erfcx(12) = Kokkos::complex<double>(0.3020886508118801, -0.09424097887578842);
    h_ref_erfcx(13) = Kokkos::complex<double>(0.3020886508118801,  0.09424097887578842);
    h_ref_erfcx(14) = Kokkos::complex<double>(-2.174707722732267, -11.67259764091796);
    h_ref_erfcx(15) = Kokkos::complex<double>(-2.174707722732267,  11.67259764091796);
    h_ref_erfcx(16) = Kokkos::complex<double>(0.7019810779371267, -0.5319516793968513);
    h_ref_erfcx(17) = Kokkos::complex<double>(0.7019810779371267,  0.5319516793968513);
    h_ref_erfcx(18) = Kokkos::complex<double>(0.7030703366403597, -0.5337884198542978);
    h_ref_erfcx(19) = Kokkos::complex<double>(0.7030703366403597,  0.5337884198542978);
    h_ref_erfcx(20) = Kokkos::complex<double>(0.9987340467266177, -0.005743428170378673);
    h_ref_erfcx(21) = Kokkos::complex<double>(0.9987340467266177,  0.005743428170378673);
    h_ref_erfcx(22) = Kokkos::complex<double>(1.001216353762532, -0.005765867613873103);
    h_ref_erfcx(23) = Kokkos::complex<double>(1.001216353762532,  0.005765867613873103);
    h_ref_erfcx(24) = Kokkos::complex<double>(0.1552936427089241, -4.545593205871305e-05);
    h_ref_erfcx(25) = Kokkos::complex<double>(0.1552936427089241,  4.545593205871305e-05);
    h_ref_erfcx(26) = Kokkos::complex<double>(417949.5262869648, -3218.276197742372);
    h_ref_erfcx(27) = Kokkos::complex<double>(417949.5262869648,  3218.276197742372);
    h_ref_erfcx(28) = Kokkos::complex<double>( 0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(29) = Kokkos::complex<double>( 0.01879467905925653,  0.0515934271478583);
    h_ref_erfcx(30) = Kokkos::complex<double>(-0.01879467905925653, -0.0515934271478583);
    h_ref_erfcx(31) = Kokkos::complex<double>(-0.01879467905925653,  0.0515934271478583);
    h_ref_erfcx(32) = Kokkos::complex<double>(0.02362328821805, -0.01209735551897239);
    h_ref_erfcx(33) = Kokkos::complex<double>(0.02362328821805,  0.01209735551897239);
    h_ref_erfcx(34) = Kokkos::complex<double>(-2.304726099084567e+114, -2.942443198107089e+114);
    h_ref_erfcx(35) = Kokkos::complex<double>(-2.304726099084567e+114,  2.942443198107089e+114);
    h_ref_erfcx(36) = Kokkos::complex<double>(0.04174017523145063, -0.1569865319886248);
    h_ref_erfcx(37) = Kokkos::complex<double>(0.04174017523145063,  0.1569865319886248);
    h_ref_erfcx(38) = Kokkos::complex<double>(-0.04172154858670504, -0.156980085534407);
    h_ref_erfcx(39) = Kokkos::complex<double>(-0.04172154858670504,  0.156980085534407);
    h_ref_erfcx(40) = Kokkos::complex<double>(6.355803055239174e-05, -0.1688298297427782);
    h_ref_erfcx(41) = Kokkos::complex<double>(6.355803055239174e-05,  0.1688298297427782);
    h_ref_erfcx(42) = Kokkos::complex<double>(-5.398806789669434e-05, -0.168829903432947);
    h_ref_erfcx(43) = Kokkos::complex<double>(-5.398806789669434e-05,  0.168829903432947);
    h_ref_erfcx(44) = Kokkos::complex<double>( 0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(45) = Kokkos::complex<double>( 0.008645103282302355,  0.07490521021566741);
    h_ref_erfcx(46) = Kokkos::complex<double>(-0.008645103282302355, -0.07490521021566741);
    h_ref_erfcx(47) = Kokkos::complex<double>(-0.008645103282302355,  0.07490521021566741);
    h_ref_erfcx(48) = Kokkos::complex<double>( 0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(49) = Kokkos::complex<double>( 0.001238176693606428,  0.02862247416909219);
    h_ref_erfcx(50) = Kokkos::complex<double>(-0.001238176693606428, -0.02862247416909219);
    h_ref_erfcx(51) = Kokkos::complex<double>(-0.001238176693606428,  0.02862247416909219);
	  
    for (int i=0; i<52; i++) {
      //printf("Test erf: %d\n",i);
      EXPECT_LE(Kokkos::abs(h_erf(i) - h_ref_erf(i)), Kokkos::abs(h_ref_erf(i))*1e-13);
      //printf("cerf %d. %.15e vs. %.15e\n", i, Kokkos::abs(h_erf(i)-h_ref_erf(i)), Kokkos::abs(h_ref_erf(i))*1e-13);
    }

    for (int i=0; i<52; i++) {
      //printf("Test erfcx: %d\n",i);
      EXPECT_LE(Kokkos::abs(h_erfcx(i) - h_ref_erfcx(i)), Kokkos::abs(h_ref_erfcx(i))*1e-13);
      //printf("cerfcx %d. %.15e vs. %.15e\n", i, Kokkos::abs(h_erfcx(i)-h_ref_erfcx(i)), Kokkos::abs(h_ref_erfcx(i))*1e-13);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & i) const {
    d_erf(i)   = Kokkos::Experimental::cerf(d_z(i));
    d_erfcx(i) = Kokkos::Experimental::cerfcx(d_z(i));
  }
};

template <class ExecSpace>
struct TestComplexBesselJ0Y0Function {
  using ViewType     = Kokkos::View<Kokkos::complex<double>*, ExecSpace>;
  using HostViewType = Kokkos::View<Kokkos::complex<double>*, Kokkos::HostSpace>;

  ViewType d_z, d_cbj0, d_cby0;
  typename ViewType::HostMirror h_z, h_cbj0, h_cby0;
  HostViewType h_ref_cbj0, h_ref_cby0;

  ViewType d_z_large, d_cbj0_large, d_cby0_large;
  typename ViewType::HostMirror h_z_large, h_cbj0_large, h_cby0_large;
  HostViewType h_ref_cbj0_large, h_ref_cby0_large;

  void testit() {
    using Kokkos::Experimental::infinity;

    int N = 25;
    d_z        = ViewType("d_z", N);
    d_cbj0     = ViewType("d_cbj0", N);
    d_cby0     = ViewType("d_cby0", N);
    h_z        = Kokkos::create_mirror_view(d_z);
    h_cbj0     = Kokkos::create_mirror_view(d_cbj0);
    h_cby0     = Kokkos::create_mirror_view(d_cby0);
    h_ref_cbj0 = HostViewType("h_ref_cbj0", N);
    h_ref_cby0 = HostViewType("h_ref_cby0", N);

    //Generate test inputs
    h_z(0)  = Kokkos::complex<double>(0.0,0.0);
    //abs(z)<=25
    h_z(1)  = Kokkos::complex<double>(3.0, 2.0);
    h_z(2)  = Kokkos::complex<double>(3.0,-2.0);
    h_z(3)  = Kokkos::complex<double>(-3.0, 2.0);
    h_z(4)  = Kokkos::complex<double>(-3.0,-2.0);
    h_z(5)  = Kokkos::complex<double>(23.0, 10.0);
    h_z(6)  = Kokkos::complex<double>(23.0,-10.0);
    h_z(7)  = Kokkos::complex<double>(-23.0, 10.0);
    h_z(8)  = Kokkos::complex<double>(-23.0,-10.0);
    h_z(9)  = Kokkos::complex<double>( 3.0,0.0);
    h_z(10) = Kokkos::complex<double>(-3.0,0.0);
    h_z(11) = Kokkos::complex<double>( 23.0,0.0);
    h_z(12) = Kokkos::complex<double>(-23.0,0.0);
    //abs(z)>25
    h_z(13) = Kokkos::complex<double>(28.0, 10.0);
    h_z(14) = Kokkos::complex<double>(28.0,-10.0);
    h_z(15) = Kokkos::complex<double>(-28.0, 10.0);
    h_z(16) = Kokkos::complex<double>(-28.0,-10.0);
    h_z(17) = Kokkos::complex<double>(60.0, 10.0);
    h_z(18) = Kokkos::complex<double>(60.0,-10.0);
    h_z(19) = Kokkos::complex<double>(-60.0, 10.0);
    h_z(20) = Kokkos::complex<double>(-60.0,-10.0);
    h_z(21) = Kokkos::complex<double>( 28.0,0.0);
    h_z(22) = Kokkos::complex<double>(-28.0,0.0);
    h_z(23) = Kokkos::complex<double>( 60.0,0.0);
    h_z(24) = Kokkos::complex<double>(-60.0,0.0);

    Kokkos::deep_copy(d_z, h_z);

    //Call Bessel functions
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, N), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj0, d_cbj0);
    Kokkos::deep_copy(h_cby0, d_cby0);

    //Reference values computed with Octave
    h_ref_cbj0(0)  = Kokkos::complex<double>( 1.000000000000000e+00, 0);
    h_ref_cbj0(1)  = Kokkos::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
    h_ref_cbj0(2)  = Kokkos::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
    h_ref_cbj0(3)  = Kokkos::complex<double>(-1.249234879607422e+00, +9.479837920577351e-01);
    h_ref_cbj0(4)  = Kokkos::complex<double>(-1.249234879607422e+00, -9.479837920577351e-01);
    h_ref_cbj0(5)  = Kokkos::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
    h_ref_cbj0(6)  = Kokkos::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
    h_ref_cbj0(7)  = Kokkos::complex<double>(-1.602439981218195e+03, -7.230667451989807e+02);
    h_ref_cbj0(8)  = Kokkos::complex<double>(-1.602439981218195e+03, +7.230667451989807e+02);
    h_ref_cbj0(9)  = Kokkos::complex<double>(-2.600519549019335e-01, 0);
    h_ref_cbj0(10) = Kokkos::complex<double>(-2.600519549019335e-01, +9.951051106466461e-18);
    h_ref_cbj0(11) = Kokkos::complex<double>(-1.624127813134866e-01, 0);
    h_ref_cbj0(12) = Kokkos::complex<double>(-1.624127813134866e-01, -1.387778780781446e-17);
    h_ref_cbj0(13) = Kokkos::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
    h_ref_cbj0(14) = Kokkos::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
    h_ref_cbj0(15) = Kokkos::complex<double>(-1.012912188513958e+03, +1.256239636146142e+03);
    h_ref_cbj0(16) = Kokkos::complex<double>(-1.012912188513958e+03, -1.256239636146142e+03);
    h_ref_cbj0(17) = Kokkos::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
    h_ref_cbj0(18) = Kokkos::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
    h_ref_cbj0(19) = Kokkos::complex<double>(-1.040215134669324e+03, +4.338202386810095e+02);
    h_ref_cbj0(20) = Kokkos::complex<double>(-1.040215134669324e+03, -4.338202386810095e+02);
    h_ref_cbj0(21) = Kokkos::complex<double>(-7.315701054899962e-02, 0);
    h_ref_cbj0(22) = Kokkos::complex<double>(-7.315701054899962e-02, -6.938893903907228e-18);
    h_ref_cbj0(23) = Kokkos::complex<double>(-9.147180408906189e-02, 0);
    h_ref_cbj0(24) = Kokkos::complex<double>(-9.147180408906189e-02, +1.387778780781446e-17);
  
    h_ref_cby0(0)  = Kokkos::complex<double>(-infinity<double>::value, 0);
    h_ref_cby0(1)  = Kokkos::complex<double>( 1.000803196554890e+00, -1.231441609303427e+00);
    h_ref_cby0(2)  = Kokkos::complex<double>( 1.000803196554890e+00, +1.231441609303427e+00);
    h_ref_cby0(3)  = Kokkos::complex<double>(-8.951643875605797e-01, -1.267028149911417e+00);
    h_ref_cby0(4)  = Kokkos::complex<double>(-8.951643875605797e-01, +1.267028149911417e+00);
    h_ref_cby0(5)  = Kokkos::complex<double>(-7.230667452992603e+02, -1.602439974000479e+03);
    h_ref_cby0(6)  = Kokkos::complex<double>(-7.230667452992603e+02, +1.602439974000479e+03);
    h_ref_cby0(7)  = Kokkos::complex<double>( 7.230667450987011e+02, -1.602439988435912e+03);
    h_ref_cby0(8)  = Kokkos::complex<double>( 7.230667450987011e+02, +1.602439988435912e+03);
    h_ref_cby0(9)  = Kokkos::complex<double>( 3.768500100127903e-01, 0);
    h_ref_cby0(10) = Kokkos::complex<double>( 3.768500100127903e-01, -5.201039098038670e-01);
    h_ref_cby0(11) = Kokkos::complex<double>(-3.598179027370283e-02, 0);
    h_ref_cby0(12) = Kokkos::complex<double>(-3.598179027370282e-02, -3.248255626269732e-01);
    h_ref_cby0(13) = Kokkos::complex<double>( 1.256239642409530e+03, -1.012912186329053e+03);
    h_ref_cby0(14) = Kokkos::complex<double>( 1.256239642409530e+03, +1.012912186329053e+03);
    h_ref_cby0(15) = Kokkos::complex<double>(-1.256239629882755e+03, -1.012912190698863e+03);
    h_ref_cby0(16) = Kokkos::complex<double>(-1.256239629882755e+03, +1.012912190698863e+03);
    h_ref_cby0(17) = Kokkos::complex<double>( 4.338202411482646e+02, -1.040215130736213e+03);
    h_ref_cby0(18) = Kokkos::complex<double>( 4.338202411482646e+02, +1.040215130736213e+03);
    h_ref_cby0(19) = Kokkos::complex<double>(-4.338202362137545e+02, -1.040215138602435e+03);
    h_ref_cby0(20) = Kokkos::complex<double>(-4.338202362137545e+02, +1.040215138602435e+03);
    h_ref_cby0(21) = Kokkos::complex<double>( 1.318364704235323e-01, 0);
    h_ref_cby0(22) = Kokkos::complex<double>( 1.318364704235323e-01, -1.463140210979992e-01);
    h_ref_cby0(23) = Kokkos::complex<double>( 4.735895220944939e-02, 0);
    h_ref_cby0(24) = Kokkos::complex<double>( 4.735895220944938e-02, -1.829436081781237e-01);
   
    for (int i=0; i<N; i++) {
      EXPECT_LE(Kokkos::abs(h_cbj0(i) - h_ref_cbj0(i)), Kokkos::abs(h_ref_cbj0(i))*1e-13);
    }

    EXPECT_EQ(h_cby0(0), h_ref_cby0(0));
    for (int i=1; i<N; i++) {
      EXPECT_LE(Kokkos::abs(h_cby0(i) - h_ref_cby0(i)), Kokkos::abs(h_ref_cby0(i))*1e-13);
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

    h_z_large(0) = Kokkos::complex<double>( 10000.0, 100.0);
    h_z_large(1) = Kokkos::complex<double>( 10000.0, 100.0);
    h_z_large(2) = Kokkos::complex<double>( 10000.0, 100.0);
    h_z_large(3) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(4) = Kokkos::complex<double>(-10000.0, 100.0);
    h_z_large(5) = Kokkos::complex<double>(-10000.0, 100.0);

    Kokkos::deep_copy(d_z_large, h_z_large);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace, TestLargeArgTag>(0, 1), *this);
    Kokkos::fence();

    Kokkos::deep_copy(h_cbj0_large, d_cbj0_large);
    Kokkos::deep_copy(h_cby0_large, d_cby0_large);

    h_ref_cbj0_large(0) = Kokkos::complex<double>(-9.561811498244175e+40, -4.854995782103029e+40);
    h_ref_cbj0_large(1) = Kokkos::complex<double>(-9.561811498244175e+40, +4.854995782103029e+40);

    h_ref_cby0_large(0) = Kokkos::complex<double>( 4.854995782103029e+40, -9.561811498244175e+40);
    h_ref_cby0_large(1) = Kokkos::complex<double>(-4.854995782103029e+40, -9.561811498244175e+40);
  
    EXPECT_TRUE((Kokkos::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) < Kokkos::abs(h_ref_cbj0_large(0))*1e-12)&&
                (Kokkos::abs(h_cbj0_large(0) - h_ref_cbj0_large(0)) > Kokkos::abs(h_ref_cbj0_large(0))*1e-13));
    EXPECT_TRUE (Kokkos::abs(h_cbj0_large(1) - h_ref_cbj0_large(0)) > Kokkos::abs(h_ref_cbj0_large(0))*1e-6);
    EXPECT_TRUE (Kokkos::abs(h_cbj0_large(2) - h_ref_cbj0_large(0)) < Kokkos::abs(h_ref_cbj0_large(0))*1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) < Kokkos::abs(h_ref_cbj0_large(1))*1e-12)&&
                (Kokkos::abs(h_cbj0_large(3) - h_ref_cbj0_large(1)) > Kokkos::abs(h_ref_cbj0_large(1))*1e-13));
    EXPECT_TRUE (Kokkos::abs(h_cbj0_large(4) - h_ref_cbj0_large(1)) > Kokkos::abs(h_ref_cbj0_large(1))*1e-6);
    EXPECT_TRUE (Kokkos::abs(h_cbj0_large(5) - h_ref_cbj0_large(1)) < Kokkos::abs(h_ref_cbj0_large(1))*1e-13);
	
    EXPECT_TRUE((Kokkos::abs(h_cby0_large(0) - h_ref_cby0_large(0)) < Kokkos::abs(h_ref_cby0_large(0))*1e-12)&&
                (Kokkos::abs(h_cby0_large(0) - h_ref_cby0_large(0)) > Kokkos::abs(h_ref_cby0_large(0))*1e-13));
    EXPECT_TRUE (Kokkos::abs(h_cby0_large(1) - h_ref_cby0_large(0)) > Kokkos::abs(h_ref_cby0_large(0))*1e-6);
    EXPECT_TRUE (Kokkos::abs(h_cby0_large(2) - h_ref_cby0_large(0)) < Kokkos::abs(h_ref_cby0_large(0))*1e-13);
    EXPECT_TRUE((Kokkos::abs(h_cby0_large(3) - h_ref_cby0_large(1)) < Kokkos::abs(h_ref_cby0_large(1))*1e-12)&&
                (Kokkos::abs(h_cby0_large(3) - h_ref_cby0_large(1)) > Kokkos::abs(h_ref_cby0_large(1))*1e-13));
    EXPECT_TRUE (Kokkos::abs(h_cby0_large(4) - h_ref_cby0_large(1)) > Kokkos::abs(h_ref_cby0_large(1))*1e-6);
    EXPECT_TRUE (Kokkos::abs(h_cby0_large(5) - h_ref_cby0_large(1)) < Kokkos::abs(h_ref_cby0_large(1))*1e-13);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & i) const {
    d_cbj0(i) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z(i));
    d_cby0(i) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z(i));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()( const TestLargeArgTag&, const int & i ) const {
    d_cbj0_large(0) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(0));
    d_cbj0_large(1) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(1),11000,3000);
    d_cbj0_large(2) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(2),11000,7500);
    d_cbj0_large(3) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(3));
    d_cbj0_large(4) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(4),11000,3000);
    d_cbj0_large(5) = Kokkos::Experimental::cbesselj0<Kokkos::complex<double>,double,int>(d_z_large(5),11000,7500);

    d_cby0_large(0) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(0));
    d_cby0_large(1) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(1),11000,3000);
    d_cby0_large(2) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(2),11000,7500);
    d_cby0_large(3) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(3));
    d_cby0_large(4) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(4),11000,3000);
    d_cby0_large(5) = Kokkos::Experimental::cbessely0<Kokkos::complex<double>,double,int>(d_z_large(5),11000,7500);
  }
};

TEST(TEST_CATEGORY, mathspecialfunc_expint) {
  TestExponentialIntergralFunction<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cmplxerror) {
  TestComplexErrorFunction<TEST_EXECSPACE> test;
  test.testit();
}

TEST(TEST_CATEGORY, mathspecialfunc_cbesselj0y0) {
  TestComplexBesselJ0Y0Function<TEST_EXECSPACE> test;
  test.testit();
}

}//namespace Test
