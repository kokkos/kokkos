// export OMP_PROC_BIND=spread ; export OMP_PLACES=threads
// c++  -O2 -g -DNDEBUG  -fopenmp ../core/perf_test/test_atomic_minmax_simple.cpp -I../core/src/ -I. -o test_atomic_minmax_simple.x  containers/src/libkokkoscontainers.a core/src/libkokkoscore.a -ldl && OMP_NUM_THREADS=1 ./test_atomic_minmax_simple.x 10000000


#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <typeinfo>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

using exec_space = Kokkos::DefaultExecutionSpace;

template<typename T>
void sanity(void)
{
    T a = 0;
    T b = 1;
    T c;

    std::cout << "atomic_fetch_min:\n";
    c = Kokkos::atomic_fetch_min(&a, b);
    std::cout << "a=" << a << "\n";
    std::cout << "b=" << b << "\n";
    std::cout << "c=" << c << "\n";

    std::cout << "atomic_fetch_max:\n";
    c = Kokkos::atomic_fetch_max(&a, b);
    std::cout << "a=" << a << "\n";
    std::cout << "b=" << b << "\n";
    std::cout << "c=" << c << "\n";

    std::cout << "atomic_fetch_max:\n";
    c = Kokkos::atomic_fetch_max(&a, b);
    std::cout << "a=" << a << "\n";
    std::cout << "b=" << b << "\n";
    std::cout << "c=" << c << "\n";

}

template<typename T>
void test(const int length)
{
    Kokkos::Impl::Timer timer;

    typedef Kokkos::View<T*, exec_space> vector;

    vector inp("input", length);

    // input is max values - all min atomics will replace
    {
        T max = std::numeric_limits<T>::max();

        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            inp(i) = max;
        });
        Kokkos::fence();

        Kokkos::fence();
        timer.reset();

        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            //std:: cout << "BEFORE inp(" << i << ")=" << inp(i) << "\n";
            T out = Kokkos::atomic_fetch_min(&(inp(i)), (T)i);
            //std:: cout << "AFTER  inp(" << i << ")=" << inp(i) << ", out=" << out << "\n";
        });
        Kokkos::fence();
        double time = timer.seconds();

        int errors(0);
        Kokkos::parallel_reduce(length, KOKKOS_LAMBDA(const int i, int & inner) {
            inner += ( inp(i) != i );
        }, errors);
        Kokkos::fence();

        if (errors) {
            std::cout << "Error in 100% min replacements: " << errors << std::endl;
            std::cout << "inp(0)=" << inp(0) << std::endl;
        }
        std::cout << "Time for 100% min replacements: " << time << std::endl;
    }

    // input is min values - all max atomics will replace
    {
        T min = std::numeric_limits<T>::min();
        if (min > 0) min = -1; // for floats, min is positive epsilon...

        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            inp(i) = min;
        });
        Kokkos::fence();

        timer.reset();
        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            T out = Kokkos::atomic_max_fetch(&(inp(i)), (T)i);
        });
        Kokkos::fence();
        double time = timer.seconds();

        int errors(0);
        Kokkos::parallel_reduce(length, KOKKOS_LAMBDA(const int i, int & inner) {
            inner += ( inp(i) != i );
        }, errors);
        Kokkos::fence();

        if (errors) {
            std::cout << "Error in 100% max replacements: " << errors << std::endl;
            std::cout << "inp(0)=" << inp(0) << std::endl;
        }
        std::cout << "Time for 100% max replacements: " << time << std::endl;
    }

    // input is max values - all max atomics will early exit
    {
        T max = std::numeric_limits<T>::max();

        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            inp(i) = max;
        });
        Kokkos::fence();

        timer.reset();
        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            T out = Kokkos::atomic_max_fetch(&(inp(i)), (T)i);
        });
        Kokkos::fence();
        double time = timer.seconds();

        int errors(0);
        Kokkos::parallel_reduce(length, KOKKOS_LAMBDA(const int i, int & inner) {
            T ref = max;
            inner += ( inp(i) != ref);
        }, errors);
        Kokkos::fence();

        if (errors) {
            std::cout << "Error in 100% max early exits: " << errors << std::endl;
            std::cout << "inp(0)=" << inp(0) << std::endl;
        }
        std::cout << "Time for 100% max early exits: " << time << std::endl;
    }

    // input is min values - all min atomics will early exit
    {
        T min = std::numeric_limits<T>::min();
        if (min > 0) min = -1; // for floats, min is positive epsilon...

        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            inp(i) = min;
        });
        Kokkos::fence();

        timer.reset();
        Kokkos::parallel_for(length, KOKKOS_LAMBDA(const int i) {
            T out = Kokkos::atomic_min_fetch(&(inp(i)), (T)i);
        });
        Kokkos::fence();
        double time = timer.seconds();

        int errors(0);
        Kokkos::parallel_reduce(length, KOKKOS_LAMBDA(const int i, int & inner) {
            T ref = min;
            inner += ( inp(i) != ref);
        }, errors);
        Kokkos::fence();

        if (errors) {
            std::cout << "Error in 100% min early exits: " << errors << std::endl;
            std::cout << "inp(0)=" << inp(0) << std::endl;
            if (length>9) std::cout << "inp(9)=" << inp(9) << std::endl;
        }
        std::cout << "Time for 100% min early exits: " << time << std::endl;
    }
}

int main(int argc, char * argv[])
{
    Kokkos::initialize(argc, argv);
    {
        sanity<int>();

        if (argc != 2) {
            std::cout << "arguments: <elements>" << std::endl;
            std::abort();
        }
        int length      = std::atoi(argv[1]);

        if (length<1) return 0;

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


