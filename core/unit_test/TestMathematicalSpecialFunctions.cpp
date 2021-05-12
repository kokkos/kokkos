#include <fstream>
#include <gtest/gtest.h>
#include "Kokkos_Core.hpp"

namespace Test {

template <class ExecSpace>
struct TestExponentialIntergralFunction {
  using ViewType     = Kokkos::View<double*, Kokkos::LayoutLeft, ExecSpace>;
  using HostViewType = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using ATR          = Kokkos::Details::ArithTraits<double>;
  using Kokkos::Experimental::infinity;

  ViewType d_x, d_expint;
  typename ViewType::HostMirror h_x, h_expint;
  HostViewType h_ref;

  void testit() {
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
    h_ref(0)  = -infinity<double>::value;//x(0)  = -0.2
    h_ref(1)  =  infinity<double>::value;//x(1)  =  0.0
    h_ref(2)  = 1.222650544183893e+00;//x(2)  =  0.2
    h_ref(3)  = 3.105965785455429e-01;//x(3)  =  0.8
    h_ref(4)  = 8.630833369753976e-02;//x(4)  =  1.6 
    h_ref(5)  = 1.021300107861738e-03;//x(5)  =  5.1
    h_ref(6)  = 4.037929576538113e+00;//x(6)  =  0.01
    h_ref(7)  = 6.331539364136149e+00;//x(7)  =  0.001
    h_ref(8)  = 2.193839343955205e-01;//x(8)  =  1.0
    h_ref(9)  = 2.190164225274689e-01;//x(9)  =  1.001
    h_ref(10) = 2.157416237944899e-01;//x(10) =  1.01
    h_ref(11) = 1.859909045360401e-01;//x(11) =  1.1
    h_ref(12) = 9.218811688716196e-05;//x(12) =  7.2
    h_ref(13) = 2.996734771597901e-06;//x(13) =  10.3
    h_ref(14) = 1.254522935050609e-08;//x(14) =  15.4

    EXPECT_EQ(h_expint(0), h_ref(0));
    EXPECT_EQ(h_expint(1), h_ref(1));
    EXPECT_FLOAT_EQ(h_expint(2) , h_ref(2) );
    EXPECT_FLOAT_EQ(h_expint(3) , h_ref(3) );
    EXPECT_FLOAT_EQ(h_expint(4) , h_ref(4) );
    EXPECT_FLOAT_EQ(h_expint(5) , h_ref(5) );
    EXPECT_FLOAT_EQ(h_expint(6) , h_ref(6) );
    EXPECT_FLOAT_EQ(h_expint(7) , h_ref(7) );
    EXPECT_FLOAT_EQ(h_expint(8) , h_ref(8) );
    EXPECT_FLOAT_EQ(h_expint(9) , h_ref(9) );
    EXPECT_FLOAT_EQ(h_expint(10), h_ref(10));
    EXPECT_FLOAT_EQ(h_expint(11), h_ref(11));
    EXPECT_FLOAT_EQ(h_expint(12), h_ref(12));
    EXPECT_FLOAT_EQ(h_expint(13), h_ref(13));
    EXPECT_FLOAT_EQ(h_expint(14), h_ref(14));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int & i) const {
    d_expint(i)  = Kokkos::Experimental::expint(d_x(i));
  }
};


TEST(TEST_CATEGORY, expint_func) {
  TestExponentialIntergralFunction<TEST_EXECSPACE> test;
  test.testit();
}

}//namespace Test
