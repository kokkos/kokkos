/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <cmath>

struct DOT {
  using view_t = Kokkos::View<double*>;
  int N;
  view_t x, y;

  bool fence_all;
  DOT(int N_, bool fence_all_)
      : N(N_), x(view_t("X", N)), y(view_t("Y", N)), fence_all(fence_all_) {}

  KOKKOS_FUNCTION
  void operator()(int i, double& lsum) const { lsum += x(i) * y(i); }

  double kk_dot(int R) {
    // Warmup
    double result;
    Kokkos::parallel_reduce("kk_dot_wup", N, *this, result);
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_reduce("kk_dot", N, *this, result);
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

#ifdef KOKKOS_ENABLE_SYCL
  double sycl_dot(int R) {
    DOT f(*this);
    const auto y_ = y.data();
    const auto x_ = x.data();

    auto sycl_queue = cl::sycl::queue(cl::sycl::gpu_selector());

    // Warmup
    {
      double result   = 0.;
      auto result_ptr = static_cast<double*>(
          sycl::malloc(sizeof(result), sycl_queue, sycl::usm::alloc::shared));
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto reduction =
            cl::sycl::ext::oneapi::reduction(result_ptr, std::plus<>());
        cgh.parallel_for(cl::sycl::nd_range<1>(N, 128), reduction,
                         [=](cl::sycl::nd_item<1> itemId, auto& sum) {
                           const int i = itemId.get_global_id();
                           sum.combine(x_[i] * y_[i]);
                         });
      });
      sycl_queue.wait();
      sycl_queue.memcpy(&result, result_ptr, sizeof(result));
      sycl_queue.wait();
    }

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      double result   = 0.;
      auto result_ptr = static_cast<double*>(
          sycl::malloc(sizeof(result), sycl_queue, sycl::usm::alloc::shared));
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        auto reduction =
            cl::sycl::ext::oneapi::reduction(result_ptr, std::plus<>());
        cgh.parallel_for(cl::sycl::nd_range<1>(N, 128), reduction,
                         [=](cl::sycl::nd_item<1> itemId, auto& sum) {
                           const int i = itemId.get_global_id();
                           sum.combine(x_[i] * y_[i]);
                         });
      });
      sycl_queue.wait();
      sycl_queue.memcpy(&result, result_ptr, sizeof(result));
      sycl_queue.wait();
    }
    double time = timer.seconds();
    return time;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMPTARGET
  double openmptarget_dot(int R) {
    DOT f(*this);
    const auto y_ = y.data();
    const auto x_ = x.data();

    // Warmup
    {
      double result = 0.;
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_)
      for (int i = 0; i < N; ++i) {
        result += x_[i] * y_[i];
      }
    }

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      double result = 0.;
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_)
      for (int i = 0; i < N; ++i) {
        result += x_[i] * y_[i];
      }
    }
    double time = timer.seconds();
    return time;
  }
#endif

  void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 2 * R;
    double GB          = bytes_moved / 1024 / 1024 / 1024;
    double time_kk     = kk_dot(R);
    printf("DOT KK: %e s %e GB/s\n", time_kk, GB / time_kk);
#ifdef KOKKOS_ENABLE_SYCL
    double time_sycl = sycl_dot(R);
    printf("DOT SYCL: %e s %e GB/s\n", time_sycl, GB / time_sycl);
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    double time_openmptarget = openmptarget_dot(R);
    printf("DOT OpenMPTarget: %e s %e GB/s\n", time_openmptarget,
           GB / time_openmptarget);
#endif
  }
};
