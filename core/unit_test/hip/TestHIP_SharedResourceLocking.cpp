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

template <typename ViewType>
struct AddTo {
  static_assert(ViewType::rank() == 0);

  ViewType data;
  typename ViewType::value_type value;

  template <typename T>
  KOKKOS_FUNCTION void operator()(const T) const {
    data() += value;
  }

  std::byte unused[Kokkos::Impl::HIPTraits::ConstantMemoryUseThreshold] = {};
};

template <typename ViewType>
struct ThreadWorkOnConstantMemory {
  using functor_t = AddTo<ViewType>;

  static_assert(sizeof(functor_t) >
                Kokkos::Impl::HIPTraits::ConstantMemoryUseThreshold);
  static_assert(sizeof(functor_t) <
                Kokkos::Impl::HIPTraits::ConstantMemoryUsage);

  std::vector<Kokkos::HIP> execs;
  ViewType data;
  typename ViewType::value_type value;

  void operator()() && {
    for (auto&& exec : execs) {
      // Sleep as long as the launch latency.
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
      Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 1),
                           functor_t{data, value});
    }
  }
};

TEST(TEST_CATEGORY, constant_memory) {
  constexpr size_t nthreads_per_device = 2 << 5;
  constexpr size_t nreps_per_thread    = 2 << 3;

  using view_t = Kokkos::View<size_t[nthreads_per_device], Kokkos::SharedSpace>;
  using view_r0_t = Kokkos::View<size_t, Kokkos::SharedSpace>;

  int num_devices = 0;
  KOKKOS_IMPL_HIP_SAFE_CALL(hipGetDeviceCount(&num_devices));

  ASSERT_GT(num_devices, 0);

  std::random_device rnd_dev;
  std::mt19937 twister{rnd_dev()};
  std::uniform_int_distribution<size_t> uniform_dist(1, nthreads_per_device);

  std::vector<std::array<std::thread, nthreads_per_device>> threads_per_device(
      num_devices);

  const view_t accu_per_device(Kokkos::view_alloc("per-device counter"));
  std::vector<size_t> expd_per_device(num_devices);

  std::vector<hipStream_t> streams_parent(num_devices);

  for (int device = 0; device < num_devices; ++device) {
    KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(device));
    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreate(&streams_parent[device]));

    const auto execs = Kokkos::Experimental::partition_space(
        Kokkos::HIP(streams_parent.at(device), Kokkos::Impl::ManageStream::no),
        std::vector<size_t>(nreps_per_thread, 1));

    for (auto& t : threads_per_device[device]) {
      const size_t value = uniform_dist(twister);
      expd_per_device.at(device) += nreps_per_thread * value;

      t = std::thread(ThreadWorkOnConstantMemory<view_r0_t>{
          execs, Kokkos::subview(accu_per_device, device), value});
    }
  }

  // Wait for all threads to finish.
  for (int device = 0; device < num_devices; ++device) {
    for (auto& t : threads_per_device.at(device)) {
      t.join();
    }

    KOKKOS_IMPL_HIP_SAFE_CALL(hipSetDevice(device));
    KOKKOS_IMPL_HIP_SAFE_CALL(hipDeviceSynchronize());

    ASSERT_EQ(accu_per_device(device), expd_per_device.at(device)) << device;

    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamDestroy(streams_parent.at(device)));
  }
}
