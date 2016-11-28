#include<Kokkos_Core.hpp>
#include<impl/Kokkos_Timer.hpp>
#include<bench.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize();
  

  if(argc<10) { 
    printf("Arguments: N K R D U F T S\n");
    printf("  P:   Precision (1==float, 2==double)\n");
    printf("  N,K: dimensions of the 2D array to allocate\n");
    printf("  R:   how often to loop through the K dimension with each team\n");
    printf("  D:   distance between loaded elements (stride)\n");
    printf("  U:   how many independent flops to do per load\n");
    printf("  F:   how many times to repeat the U unrolled operations before reading next element\n");
    printf("  T:   team size\n");
    printf("  S:   shared memory per team (used to control occupancy on GPUs)\n");
    printf("Example Input GPU:\n");
    printf("  Bandwidth Bound : 2 100000 1024 1 1 1 1 256 6000\n");
    printf("  Cache Bound     : 2 100000 1024 64 1 1 1 512 20000\n");
    printf("  Compute Bound   : 2 100000 1024 1 1 8 64 256 6000\n");
    printf("  Load Slots Used : 2 20000 256 32 16 1 1 256 6000\n");
    printf("  Inefficient Load: 2 20000 256 32 2 1 1 256 20000\n");
    Kokkos::finalize();
    return 0;
  }
  

  int P = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int R = atoi(argv[4]);
  int D = atoi(argv[5]);
  int U = atoi(argv[6]);
  int F = atoi(argv[7]);
  int T = atoi(argv[8]);
  int S = atoi(argv[9]);

  if(U>8) {printf("U must be 1-8\n"); return 0;} 
  if( (D!=1) && (D!=2) && (D!=4) && (D!=8) && (D!=16) && (D!=32)) {printf("D must be one of 1,2,4,8,16,32\n"); return 0;}
  if( (P!=1) && (P!=2) ) {printf("P must be one of 1,2\n"); return 0;}

  if(P==1) {
    run_stride_unroll<float>(N,K,R,D,U,F,T,S);
  }
  if(P==2) {
    run_stride_unroll<double>(N,K,R,D,U,F,T,S);
  }

  Kokkos::finalize();
}

