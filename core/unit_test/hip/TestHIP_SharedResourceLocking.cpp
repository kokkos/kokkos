//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <random>

#include <Kokkos_Core.hpp>
#include <TestHIP_Category.hpp>

struct ThreadWorkOnSharedResource {
  int device;
  size_t value;
  size_t* counter;

  // Detached work that ends up releasing the shared resource lock.
  void callback() const {
    std::this_thread::sleep_for(std::chrono::microseconds{value});
    *counter += value;
    Kokkos::Impl::HIPInternal::constantMemReusable[device].release();
  }

  // This is what is typically done in
  // HIPParallelLaunchKernelInvoker::invoke_kernel. First, acquire the lock,
  // then spawn some asynchronous work that will release the shared resource
  // lock.
  void operator()() const {
    [[maybe_unused]] const auto lock =
        Kokkos::Impl::HIPInternal::constantMemReusable[device].acquire();
    std::thread(&ThreadWorkOnSharedResource::callback, *this).detach();
  }
};

TEST(TEST_CATEGORY, shared_resource_locking) {
  constexpr size_t nthreads_per_device = 2 << 5;

  int num_devices = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceCount(&num_devices));

  ASSERT_GT(num_devices, 0);

  std::random_device rnd_dev;
  std::mt19937 twister{rnd_dev()};
  std::uniform_int_distribution<size_t> uniform_dist(1, nthreads_per_device);

  std::vector<std::array<std::thread, nthreads_per_device>> threads_per_device(
      num_devices);

  std::vector<size_t> accu_per_device(num_devices);
  std::vector<size_t> expd_per_device(num_devices);

  for (int device = 0; device < num_devices; ++device) {
    for (auto& t : threads_per_device[device]) {
      const size_t value = uniform_dist(twister);
      expd_per_device.at(device) += value;
      t = std::thread(ThreadWorkOnSharedResource{
          device, value, std::addressof(accu_per_device.at(device))});
    }
  }

  // Wait for all threads to finish.
  for (int device = 0; device < num_devices; ++device) {
    for (auto& t : threads_per_device.at(device)) {
      t.join();
    }
  }

  // Since threads have spawned detached work, we need to ensure all shared
  // resource locks have been released before reading the accumulated values.
  for (int device = 0; device < num_devices; ++device) {
    Kokkos::Impl::HIPInternal::constantMemReusable[device].acquire();
    Kokkos::Impl::HIPInternal::constantMemReusable[device].release();
  }

  ASSERT_EQ(accu_per_device, expd_per_device);
}
