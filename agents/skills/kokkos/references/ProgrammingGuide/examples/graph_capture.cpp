#include "cublas_v2.h"

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#define ASSERT_EQ(a, b) if(a != b) throw std::runtime_error("There was a problem: " #a " is not equal to " #b);

#define CHECK_CUBLAS_CALL(call)                           \
  {                                                       \
    const auto error_code = call;                         \
    if(error_code != CUBLAS_STATUS_SUCCESS)               \
    {                                                     \
      printf("%s:%d: failure of statement %s: %s (%d)\n", \
        __FILE__, __LINE__,                               \
        #call,                                            \
        cublasGetStatusName(error_code), error_code);     \
      std::abort();                                       \
    }                                                     \
  }

template <typename MatrixType, typename VectorType>
void initialize_system(const Kokkos::Cuda& exec, const MatrixType& matrix, const VectorType& vector) {
  Kokkos::parallel_for(
    Kokkos::RangePolicy(exec, 0, matrix.extent(0)),
    KOKKOS_LAMBDA(const Kokkos::Cuda::size_type irow) {
      matrix(irow, 0) = 2 * irow + 1;
      matrix(irow, 1) = 2 * irow + 2;

      vector(irow) = irow + 5;
    }
  );
};

auto create_cublas_handle()
{
  cublasHandle_t handle = nullptr;

  CHECK_CUBLAS_CALL(cublasCreate(&handle));

  return std::shared_ptr<cublasContext>(handle, [](cublasHandle_t ptr) {
    CHECK_CUBLAS_CALL(cublasDestroy(ptr));
  });
}

template <typename Exec, typename Predecessor, typename MatrixType, typename VectorType, typename ResultType>
auto gemv(const Exec& exec, const Predecessor& predecessor, const MatrixType& matrix, const VectorType& vector, const ResultType& result)
{
  static_assert(std::is_same_v<typename MatrixType::value_type, double>);
  static_assert(std::is_same_v<typename VectorType::value_type, double>);
  static_assert(std::is_same_v<typename ResultType::value_type, double>);

  auto handle = create_cublas_handle();

  const double alpha = 1., beta = 1.;

  /// The @c handle is a shared resource stored in the lambda.
  /// Since the lambda is stored by the node, and the node won't be destroyed
  /// until the @c Kokkos::Graph is destroyed, the @c cuBLAS handle is
  /// guaranteed to stay alive for the whole graph lifetime.
  return predecessor.cuda_capture(
    exec,
    [=](const Kokkos::Cuda& exec_) {
      CHECK_CUBLAS_CALL(cublasSetStream(handle.get(), exec_.cuda_stream()));
      CHECK_CUBLAS_CALL(cublasDgemv(
        handle.get(),
        CUBLAS_OP_N,
        vector.size(),
        result.size(),
        &alpha,
        matrix.data(), vector.size(),
        vector.data(), 1,
        &beta,
        result.data(), 1
      ));
    }
  );
}

void test_capture()
{
  constexpr size_t nrows = 2, ncols = 2;

  using value_t    = double;
  using matrix_t   = Kokkos::View<value_t[nrows][ncols], Kokkos::LayoutRight, Kokkos::CudaSpace>;
  using vector_d_t = Kokkos::View<value_t[nrows],                             Kokkos::CudaSpace>;
  using vector_s_t = Kokkos::View<value_t[nrows],                             Kokkos::SharedSpace>;

  const Kokkos::Cuda exec {};

  const matrix_t   matrix(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "matrix"));
  const vector_d_t vector(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec, "vector"));
  const vector_s_t result(Kokkos::view_alloc(                             exec, "result"));

  initialize_system(exec, matrix, vector);

  auto graph = Kokkos::Experimental::create_graph([&](const auto& root) {
    auto node_gemv = gemv(exec, root, matrix, vector, result);
  });

  graph.instantiate();

  //! The views are stored in the graph node. No kernel ran yet.
  ASSERT_EQ(matrix.use_count(), 2);
  ASSERT_EQ(vector.use_count(), 2);
  ASSERT_EQ(result.use_count(), 2);
  ASSERT_EQ(result(0), 0);
  ASSERT_EQ(result(1), 0);

  //! Let's submit the graph twice, to ensure that the captured node behaves well.
  graph.submit(exec);

  Kokkos::deep_copy(exec, vector, result);

  graph.submit(exec);

  exec.fence();

  ASSERT_EQ(result(0), 23 + 125);
  ASSERT_EQ(result(1), 34 + 182);
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  test_capture();
  Kokkos::finalize();
  return EXIT_SUCCESS;
}
