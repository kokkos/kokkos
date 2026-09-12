// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestCuda_Category.hpp>
#include <TestMultiGPU.hpp>

namespace {

struct StreamsAndDevices {
  std::array<cudaStream_t, 2> streams;
  std::array<int, 2> devices;

  StreamsAndDevices() {
    int n_devices;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaGetDeviceCount(&n_devices));

    devices = {0, n_devices - 1};
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
    }
  }
  StreamsAndDevices(const StreamsAndDevices &)            = delete;
  StreamsAndDevices &operator=(const StreamsAndDevices &) = delete;
  ~StreamsAndDevices() {
    for (int i = 0; i < 2; ++i) {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[i]));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
    }
  }
};

struct TEST_CATEGORY_FIXTURE(multi_gpu) : public ::testing::Test {
  void SetUp() override {
    auto execs = Kokkos::create_device_space();
    if (execs.size() <= 1) {
      GTEST_SKIP() << "Skipping Cuda multi-gpu testing since current machine "
                      "only contains a single GPU.\n";
    }

    ASSERT_GE(execs.size(), 2);
    ASSERT_NE(execs[0].cuda_device(), execs[1].cuda_device());
  }
};

TEST_F(TEST_CATEGORY_FIXTURE(multi_gpu), managed_views) {
  auto execs = Kokkos::create_device_space();

  Kokkos::View<int *, TEST_EXECSPACE> view0(Kokkos::view_alloc("v0", execs[0]),
                                            100);
  Kokkos::View<int *, TEST_EXECSPACE> view1(Kokkos::view_alloc("v1", execs[1]),
                                            100);

  test_policies(execs[0], view0, execs[1], view1);
}

TEST_F(TEST_CATEGORY_FIXTURE(multi_gpu), unmanaged_views) {
  auto execs = Kokkos::create_device_space();

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[0].cuda_device()));
  int *p0;
  void *p0_void_ptr = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&p0_void_ptr, sizeof(int) * 100));
  p0 = static_cast<int *>(p0_void_ptr);
  Kokkos::View<int *, TEST_EXECSPACE> view0(p0, 100);

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(execs[1].cuda_device()));
  int *p;
  void *p_void_ptr = nullptr;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(&p_void_ptr, sizeof(int) * 100));
  p = static_cast<int *>(p_void_ptr);
  Kokkos::View<int *, TEST_EXECSPACE> view(p, 100);

  test_policies(execs[0], view0, execs[1], view);
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p0));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(p));
}

TEST_F(TEST_CATEGORY_FIXTURE(multi_gpu), scratch_space) {
  auto execs = Kokkos::create_device_space();

  test_scratch(execs[0], execs[1]);
}

TEST_F(TEST_CATEGORY_FIXTURE(multi_gpu), stream_sync_semantics_raw_cuda) {
  // Test that stream synchronization behavior for various GPU APIs matches the
  // assumptions made in Kokkos for multi gpu support, namely, that any stream
  // (no matter which device it is created on) can be synced from any device.

  StreamsAndDevices streams_and_devices;
  {
    auto streams = streams_and_devices.streams;
    auto devices = streams_and_devices.devices;

    // Allocate data.
    int *value;
    void *value_void_ptr = nullptr;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMallocHost(&value_void_ptr, 1 * sizeof(int)));
    value = static_cast<int *>(value_void_ptr);

    int *check;
    void *check_void_ptr = nullptr;
    KOKKOS_IMPL_CUDA_SAFE_CALL(
        cudaMallocHost(&check_void_ptr, 1 * sizeof(int)));
    check = static_cast<int *>(check_void_ptr);

    // Launch "long" kernel on device 0.
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[0]));
    constexpr int size = 10000;
    accumulate_kernel<size><<<1, 1, 0, streams[0]>>>(value);

    // Wait for the kernel running on device 0 while we are on device 1, then
    // check the value.
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaSetDevice(devices[1]));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(streams[0]));
    copy_kernel<<<1, 1, 0, streams[1]>>>(check, value);
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamSynchronize(streams[1]));
    ASSERT_EQ(check[0], size);

    // Cleanup.
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(value));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeHost(check));
  }
}

}  // namespace
