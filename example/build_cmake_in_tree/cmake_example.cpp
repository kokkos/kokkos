#include <Kokkos_Core.hpp>

template <class V>
struct MyFunctor {
  // using value_type = typename V::value_type;
  V view;

  template <class I0>
  KOKKOS_FUNCTION void operator()(I0 i0, typename V::value_type &res) const {
    res += view(i0);
  }

 template <class I0>
  KOKKOS_FUNCTION void operator()(I0 i0, typename V::value_type &res, bool) const {
    res += view(i0);
  }

};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    size_t Nx = 4;
    Kokkos::View<double *> myview("myview", Nx);

    MyFunctor<decltype(myview)> func = {myview};
    double res                       = 0;
    Kokkos::parallel_reduce("sum", myview.extent(0), func, res);
    Kokkos::parallel_scan(myview.extent(0), func, res);
  }
  Kokkos::finalize();
}
