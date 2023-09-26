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

struct Opts {
  bool no_par_for         = false;
  bool no_par_reduce      = false;
  bool no_par_reduce_view = false;
};

template <int V>
void run(int N, int M, int K, const Opts& opts) {
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
  Kokkos::Timer timer;

  // initialize to a obviously wrong value
  double time_no_fence        = -1;  // launch loop
  double time_no_fence_fenced = -1;  // launch loop then fence
  double time_fence           = -1;  // launch&fence loop
  double time_fence_fenced    = -1;  // launch&fence loop then fence

  double time_red_no_fence        = -1;
  double time_red_no_fence_fenced = -1;
  double time_red_fence           = -1;
  double time_red_fence_fenced    = -1;

  double time_red_view_no_fence        = -1;
  double time_red_view_no_fence_fenced = -1;
  double time_red_view_fence           = -1;
  double time_red_view_fence_fenced    = -1;

  if (!opts.no_par_for) {
    // warmup
    Kokkos::parallel_for(l_no_fence, N, f);
    Kokkos::parallel_for(l_no_fence, N, f);
    Kokkos::parallel_for(l_no_fence, N, f);
    Kokkos::parallel_for(l_no_fence, N, f);

    Kokkos::fence();

    timer.reset();
    for (int i = 0; i < M; i++) {
      Kokkos::parallel_for(l_no_fence, N, f);
    }
    time_no_fence = timer.seconds();
    Kokkos::fence();
    time_no_fence_fenced = timer.seconds();
    timer.reset();

    for (int i = 0; i < M; i++) {
      Kokkos::parallel_for(l_fence, N, f);
      Kokkos::fence();
    }
    time_fence = timer.seconds();
    Kokkos::fence();
    time_fence_fenced = timer.seconds();
  }

  if (!opts.no_par_reduce) {
    // warmup
    Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
    Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
    Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
    Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);

    timer.reset();

    for (int i = 0; i < M; i++) {
      Kokkos::parallel_reduce(l_red_no_fence, N, rf, result);
    }
    time_red_no_fence = timer.seconds();
    Kokkos::fence();
    time_red_no_fence_fenced = timer.seconds();
    timer.reset();

    for (int i = 0; i < M; i++) {
      Kokkos::parallel_reduce(l_red_fence, N, rf, result);
      Kokkos::fence();
    }
    time_red_fence = timer.seconds();
    Kokkos::fence();
    time_red_fence_fenced = timer.seconds();
  }

  if (!opts.no_par_reduce_view) {
    // warmup
    Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
    Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
    Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
    Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);

    timer.reset();

    for (int i = 0; i < M; i++) {
      Kokkos::parallel_reduce(l_red_view_no_fence, N, rf, v_result);
    }
    time_red_view_no_fence = timer.seconds();
    Kokkos::fence();
    time_red_view_no_fence_fenced = timer.seconds();
    timer.reset();

    for (int i = 0; i < M; i++) {
      Kokkos::parallel_reduce(l_red_view_fence, N, rf, v_result);
      Kokkos::fence();
    }
    time_red_view_fence = timer.seconds();
    Kokkos::fence();
    time_red_view_fence_fenced = timer.seconds();
    timer.reset();
  }

  const double x = 1.e6 / M;
  printf("%i %i %i %i", N, V, K, M);
  if (!opts.no_par_for) {
    printf(" parallel_for: %lf %lf ( %lf %lf )", x * time_no_fence,
           x * time_fence, x * time_no_fence_fenced, x * time_fence_fenced);
  }
  if (!opts.no_par_reduce) {
    printf(" parallel_reduce: %lf %lf ( %lf %lf )", x * time_red_no_fence,
           x * time_red_fence, x * time_red_no_fence_fenced,
           x * time_red_fence_fenced);
  }
  if (!opts.no_par_reduce_view) {
    printf(" parallel_reduce(view): %lf %lf ( %lf %lf )",
           x * time_red_view_no_fence, x * time_red_view_fence,
           x * time_red_view_no_fence_fenced, x * time_red_view_fence_fenced);
  }
  printf("\n");
}
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = 10000;
    int M = 1;
    int K = 1;

    Opts opts;

    printf("==========================\n");
    printf("Kokkos Launch Latency Test\n");
    printf("==========================\n");
    printf("\n\n");
    printf("Usage: %s ARGUMENTS [OPTIONS...]\n\n", argv[0]);
    printf("Arguments: N M K\n");
    printf("  N: loop length\n");
    printf("  M: how many kernels to dispatch\n");
    printf(
        "  K: nested loop length (capped by size of functor member array\n\n");
    printf("Options:\n");
    printf("  --no-parallel-for:         skip parallel_for benchmark\n");
    printf("  --no-parallel-reduce:      skip parallel_reduce benchmark\n");
    printf(
        "  --no-parallel-reduce-view: skip parallel_reduce into view "
        "benchmark\n");
    printf("\n\n");
    printf("  Output V is the size of the functor member array\n");
    printf("\n\n");

    for (int i = 1; i < argc; ++i) {
      const std::string_view arg(argv[i]);

      // anything that doesn't start with --
      if (arg.size() >= 2 && arg[0] != '-' && arg[1] != '-') {
        if (i == 1)
          N = atoi(arg.data());
        else if (i == 2)
          M = atoi(arg.data());
        else if (i == 3)
          K = atoi(arg.data());
        else {
          throw std::runtime_error("unexpected argument!");
        }
      } else if (arg == "--no-parallel-for") {
        opts.no_par_for = true;
      } else if (arg == "--no-parallel-reduce") {
        opts.no_par_reduce = true;
      } else if (arg == "--no-parallel-reduce-view") {
        opts.no_par_reduce_view = true;
      } else {
        std::stringstream ss;
        ss << "unexpected option " << arg;
        throw std::runtime_error(ss.str());
      }
    }

    printf(
        "N V K M time_no_fence time_fence (time_no_fence_fenced "
        "time_fence_fenced)\n");

    run<1>(N, M, K <= 1 ? K : 1, opts);
    run<16>(N, M, K <= 16 ? K : 16, opts);
    run<200>(N, M, K <= 200 ? K : 200, opts);
    run<3000>(N, M, K <= 3000 ? K : 3000, opts);
    run<30000>(N, M, K <= 30000 ? K : 30000, opts);
  }
  Kokkos::finalize();
}