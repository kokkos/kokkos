#include <Kokkos_Core.hpp>

template <int N>
struct TestFunctor {
  double values[N];
  Kokkos::View<double*> a;
  int K;
  TestFunctor(Kokkos::View<double*> a_, int K_) : a(a_), K(K_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (int j = 0; j < K; j++) a(i) += 1.0 * i * values[j];
  }
};

template <int N>
struct TestRFunctor {
  double values[N];
  Kokkos::View<double*> a;
  int K;
  TestRFunctor(Kokkos::View<double*> a_, int K_) : a(a_), K(K_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, double& lsum) const {
    for (int j = 0; j < K; j++) a(i) += 1.0 * i * values[j];
    lsum += a(i);
  }
};

template <int V>
void run(int N, int M, int K) {
  std::string l_no_fence, l_fence, l_red_no_fence, l_red_fence,
      l_red_view_no_fence, l_red_view_fence;
  {
    std::ostringstream ostream;
    ostream << "RunNoFence_" << N << "_" << K << std::endl;
    l_no_fence = ostream.str();
  }
  {
    std::ostringstream ostream;
    ostream << "RunFence_" << N << "_" << K << std::endl;
    l_fence = ostream.str();
  }
  {
    std::ostringstream ostream;
    ostream << "RunReduceNoFence_" << N << "_" << K << std::endl;
    l_red_no_fence = ostream.str();
  }
  {
    std::ostringstream ostream;
    ostream << "RunReduceFence_" << N << "_" << K << std::endl;
    l_red_fence = ostream.str();
  }
  {
    std::ostringstream ostream;
    ostream << "RunReduceViewNoFence_" << N << "_" << K << std::endl;
    l_red_view_no_fence = ostream.str();
  }
  {
    std::ostringstream ostream;
    ostream << "RunReduceViewFence_" << N << "_" << K << std::endl;
    l_red_view_fence = ostream.str();
  }

  double result;
  Kokkos::View<double*> a("A", N);
  Kokkos::View<double> v_result("result");
  TestFunctor<V> f(a, K);
  TestRFunctor<V> rf(a, K);

  // warmup
  Kokkos::parallel_for(l_no_fence, N, f);
  Kokkos::parallel_for(l_no_fence, N, f);
  Kokkos::parallel_for(l_no_fence, N, f);
  Kokkos::parallel_for(l_no_fence, N, f);

  Kokkos::fence();
  Kokkos::Timer timer;
  for (int i = 0; i < M; i++) {
    Kokkos::parallel_for(l_no_fence, N, f);
  }
  double time_no_fence = timer.seconds();
  Kokkos::fence();
  double time_no_fence_fenced = timer.seconds();
  timer.reset();

  for (int i = 0; i < M; i++) {
    Kokkos::parallel_for(l_fence, N, f);
    Kokkos::fence();
  }
  double time_fence = timer.seconds();
  Kokkos::fence();
  double time_fence_fenced = timer.seconds();

  // warmup
  Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
  Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
  Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
  Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);

  timer.reset();

  for (int i = 0; i < M; i++) {
    Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
  }
  double time_red_no_fence = timer.seconds();
  Kokkos::fence();
  double time_red_no_fence_fenced = timer.seconds();
  timer.reset();

  for (int i = 0; i < M; i++) {
    Kokkos::parallel_reduce(l_red_fence, N, rf, result);
    Kokkos::fence();
  }
  double time_red_fence = timer.seconds();
  Kokkos::fence();
  double time_red_fence_fenced = timer.seconds();

  // warmup
  Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
  Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
  Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
  Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);

  timer.reset();

  for (int i = 0; i < M; i++) {
    Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
  }
  double time_red_view_no_fence = timer.seconds();
  Kokkos::fence();
  double time_red_view_no_fence_fenced = timer.seconds();
  timer.reset();

  for (int i = 0; i < M; i++) {
    Kokkos::parallel_reduce(l_red_view_fence, N, rf, v_result);
    Kokkos::fence();
  }
  double time_red_view_fence = timer.seconds();
  Kokkos::fence();
  double time_red_view_fence_fenced = timer.seconds();
  timer.reset();

  double x = 1.e6 / M;
  printf(
      "%i %i %i %i parallel_for: %lf %lf ( %lf %lf ) parallel_reduce: %lf %lf "
      "( %lf %lf ) parallel_reduce(view): %lf %lf ( %lf %lf )\n",
      N, V, K, M, x * time_no_fence, x * time_fence, x * time_no_fence_fenced,
      x * time_fence_fenced, x * time_red_no_fence, x * time_red_fence,
      x * time_red_no_fence_fenced, x * time_red_fence_fenced,
      x * time_red_view_no_fence, x * time_red_view_fence,
      x * time_red_view_no_fence_fenced, x * time_red_view_fence_fenced);
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    int M = (argc > 2) ? atoi(argv[2]) : 1;
    int K = (argc > 3) ? atoi(argv[3]) : 1;

    printf("==========================\n");
    printf("Kokkos Launch Latency Test\n");
    printf("==========================\n");
    printf("\n\n");
    printf("Arguments: N M K\n");
    printf("  N: loop length\n");
    printf("  M: how many kernels to dispatch\n");
    printf("  K: nested loop length (capped by size of functor member array\n");
    printf("\n\n");
    printf("  Output V is the size of the functor member array\n");
    printf("\n\n");
    printf(
        "N V K M time_no_fence time_fence (time_no_fence_fenced "
        "time_fence_fenced)\n");
    run<1>(N, M, K <= 1 ? K : 1);
    run<16>(N, M, K <= 16 ? K : 16);
    run<200>(N, M, K <= 200 ? K : 200);
    run<3000>(N, M, K <= 3000 ? K : 3000);
    run<30000>(N, M, K <= 30000 ? K : 30000);
  }
  Kokkos::finalize();
}