// export OMP_PROC_BIND=spread ; export OMP_PLACES=threads
// c++  -O2 -g -DNDEBUG  -fopenmp
// ../core/perf_test/test_atomic_minmax_simple.cpp -I../core/src/ -I. -o
// test_atomic_minmax_simple.x  containers/src/libkokkoscontainers.a
// core/src/libkokkoscore.a -ldl && OMP_NUM_THREADS=1
// ./test_atomic_minmax_simple.x 10000000

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <typeinfo>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

using exec_space = Kokkos::DefaultExecutionSpace;

template <typename T>
void test(const int length) {
  Kokkos::Impl::Timer timer;

  using vector = Kokkos::View<T*, exec_space>;

  vector inp("input", length);

  // input is max values - all min atomics will replace
  {
    T max = std::numeric_limits<T>::max();

    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) { inp(i) = max; });
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) {
          (void)Kokkos::atomic_fetch_min(&(inp(i)), (T)i);
        });
    Kokkos::fence();
    double time = timer.seconds();

    int errors(0);
    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int i, int& inner) { inner += (inp(i) != (T)i); },
        errors);
    Kokkos::fence();

    if (errors) {
      std::cerr << "Error in 100% min replacements: " << errors << std::endl;
      std::cerr << "inp(0)=" << inp(0) << std::endl;
    }
    std::cout << "Time for 100% min replacements: " << time << std::endl;
  }

  // input is min values - all max atomics will replace
  {
    T min = std::numeric_limits<T>::lowest();

    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) { inp(i) = min; });
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) {
          (void)Kokkos::atomic_max_fetch(&(inp(i)), (T)i);
        });
    Kokkos::fence();
    double time = timer.seconds();

    int errors(0);
    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int i, int& inner) { inner += (inp(i) != (T)i); },
        errors);
    Kokkos::fence();

    if (errors) {
      std::cerr << "Error in 100% max replacements: " << errors << std::endl;
      std::cerr << "inp(0)=" << inp(0) << std::endl;
    }
    std::cout << "Time for 100% max replacements: " << time << std::endl;
  }

  // input is max values - all max atomics will early exit
  {
    T max = std::numeric_limits<T>::max();

    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) { inp(i) = max; });
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) {
          (void)Kokkos::atomic_max_fetch(&(inp(i)), (T)i);
        });
    Kokkos::fence();
    double time = timer.seconds();

    int errors(0);
    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int i, int& inner) {
          T ref = max;
          inner += (inp(i) != ref);
        },
        errors);
    Kokkos::fence();

    if (errors) {
      std::cerr << "Error in 100% max early exits: " << errors << std::endl;
      std::cerr << "inp(0)=" << inp(0) << std::endl;
    }
    std::cout << "Time for 100% max early exits: " << time << std::endl;
  }

  // input is min values - all min atomics will early exit
  {
    T min = std::numeric_limits<T>::lowest();

    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) { inp(i) = min; });
    Kokkos::fence();

    timer.reset();
    Kokkos::parallel_for(
        length, KOKKOS_LAMBDA(const int i) {
          (void)Kokkos::atomic_min_fetch(&(inp(i)), (T)i);
        });
    Kokkos::fence();
    double time = timer.seconds();

    int errors(0);
    Kokkos::parallel_reduce(
        length,
        KOKKOS_LAMBDA(const int i, int& inner) {
          T ref = min;
          inner += (inp(i) != ref);
        },
        errors);
    Kokkos::fence();

    if (errors) {
      std::cerr << "Error in 100% min early exits: " << errors << std::endl;
      std::cerr << "inp(0)=" << inp(0) << std::endl;
      if (length > 9) std::cout << "inp(9)=" << inp(9) << std::endl;
    }
    std::cout << "Time for 100% min early exits: " << time << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int length = 1000000;
    if (argc == 2) {
      length = std::atoi(argv[1]);
    }

    if (length < 1) return 0;

    std::cout << "================ int" << std::endl;
    test<int>(length);
    std::cout << "================ long" << std::endl;
    test<long>(length);
    std::cout << "================ long long" << std::endl;
    test<long long>(length);

    std::cout << "================ unsigned int" << std::endl;
    test<unsigned int>(length);
    std::cout << "================ unsigned long" << std::endl;
    test<unsigned long>(length);
    std::cout << "================ unsigned long long" << std::endl;
    test<unsigned long long>(length);

    std::cout << "================ float" << std::endl;
    test<float>(length);
    std::cout << "================ double" << std::endl;
    test<double>(length);
  }
  Kokkos::finalize();
  return 0;
}
