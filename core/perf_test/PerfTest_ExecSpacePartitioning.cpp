#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <PerfTest_Category.hpp>


namespace Test {

namespace {
  template<class ExecSpace>
  struct SpaceInstance {
    static ExecSpace create() {
      return ExecSpace();
    }
    static void destroy(ExecSpace&) {
    }
    static bool overlap() {
      return false;
    }
  };

  #ifdef KOKKOS_ENABLE_CUDA
  template<>
  struct SpaceInstance<Kokkos::Cuda> {
    static Kokkos::Cuda create() {
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      return Kokkos::Cuda(stream);
    }
    static void destroy(Kokkos::Cuda& space) {
      cudaStream_t stream = space.cuda_stream();
      cudaStreamDestroy(stream);
    }
    static bool overlap() {
      bool value = true;
      auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
      if(local_rank_str) {
        value = (std::atoi(local_rank_str)==0);
      }
      return value;
    }
  };
  #endif
}

struct functor {
  int M,R;
  Kokkos::View<double**,TEST_EXECSPACE> a;
  functor(int M_, int R_, Kokkos::View<double**,TEST_EXECSPACE> a_):M(M_),R(R_),a(a_){}
  KOKKOS_INLINE_FUNCTION
  void operator() (const int i) const {
    for(int r=0;r<R;r++)
    for(int j=0;j<M;j++) {
      a(i,j)+=1.0;
    }
  }
};


TEST_F( default_exec, overlap_range_policy ) {
  int N = 2000;
   int M = 10000;
   int R =  10;

   TEST_EXECSPACE space;
   TEST_EXECSPACE space1 = SpaceInstance<TEST_EXECSPACE>::create();
   TEST_EXECSPACE space2 = SpaceInstance<TEST_EXECSPACE>::create();

   Kokkos::View<double**,TEST_EXECSPACE> a("A",N,M);
   functor f(M,R,a);
   Kokkos::parallel_for(N, functor(M,R,a));

   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space1,0,N), f);
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space2,0,N), f);
   Kokkos::fence();

   Kokkos::Timer timer;
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space,0,N), f);
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space,0,N), f);
   Kokkos::fence();
   double time_start = timer.seconds();

   timer.reset();
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space1,0,N), functor(M,R,a));
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space2,0,N), functor(M,R,a));
   Kokkos::fence();
   double time_overlap = timer.seconds();

   timer.reset();
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space,0,N), f);
   Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(space,0,N), f);
   Kokkos::fence();
   double time_end = timer.seconds();

   SpaceInstance<TEST_EXECSPACE>::destroy(space1);
   SpaceInstance<TEST_EXECSPACE>::destroy(space2);

   if(SpaceInstance<TEST_EXECSPACE>::overlap()) {
     ASSERT_TRUE( (time_end > 1.5*time_overlap) );
   }
   printf("Time NonOverlap: %lf Time Overlap: %lf\n",time_end,time_overlap);
}
}
