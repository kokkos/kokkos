#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double*> test("test", 100);
    Kokkos::parallel_for("memset", 100, KOKKOS_LAMBDA (int idx){
      test(idx) = 0;
    });
    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}
