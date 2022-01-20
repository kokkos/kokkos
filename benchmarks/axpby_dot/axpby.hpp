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

struct AXPBY {
  using view_t = Kokkos::View<double*>;
  int N;
  view_t x, y, z;

  bool fence_all;
  AXPBY(int N_, bool fence_all_)
      : N(N_),
        x(view_t("X", N)),
        y(view_t("Y", N)),
        z(view_t("Z", N)),
        fence_all(fence_all_) {}

  KOKKOS_FUNCTION
  void operator()(int i) const { z(i) = x(i) + y(i); }

  double kk_axpby(int R) {
    // Warmup
    Kokkos::parallel_for("kk_axpby_wup", N, *this);
    Kokkos::fence();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      Kokkos::parallel_for("kk_axpby", N, *this);
      if (fence_all) Kokkos::fence();
    }
    Kokkos::fence();
    double time = timer.seconds();
    return time;
  }

#ifdef KOKKOS_ENABLE_SYCL
  double sycl_axpby(int R) {
    auto sycl_queue = cl::sycl::queue(cl::sycl::gpu_selector());

    double* x_ = sycl::malloc_device<double>(N, sycl_queue);
    double* y_ = sycl::malloc_device<double>(N, sycl_queue);
    double* z_ = sycl::malloc_device<double>(N, sycl_queue);

    // Initialization
    sycl_queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::item<1> itemId) {
        const int i = itemId.get_id();
        z_[i] = x_[i] = y_[i] = 0;
      });
    });
    sycl_queue.wait();

    // Warmup
    sycl_queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for(cl::sycl::range<1>(N), [=](cl::sycl::item<1> itemId) {
        const int i = itemId.get_id();
        z_[i]       = x_[i] + y_[i];
      });
    });
    sycl_queue.wait();

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
      sycl_queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class vec_add>(cl::sycl::range<1>(N),
                                        [=](cl::sycl::item<1> itemId) {
                                          const int i = itemId.get_id();
                                          z_[i]       = x_[i] + y_[i];
                                        });
      });
      if (fence_all) sycl_queue.wait();
    }
    sycl_queue.wait();
    double time = timer.seconds();
    return time;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMPTARGET
  double openmptarget_axpby(int R) {
    double* x_ = x.data();
    double* y_ = y.data();
    double* z_ = z.data();

    // Initialization
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_, z_)
    for (int i = 0; i < N; ++i) {
      z_[i] = x_[i] = y_[i] = 0;
    }

    // Warmup
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_, z_)
    for (int i = 0; i < N; ++i) {
      z_[i] = x_[i] + y_[i];
    }

    Kokkos::Timer timer;
    for (int r = 0; r < R; r++) {
#pragma omp target teams distribute parallel for is_device_ptr(x_, y_, z_)
      for (int i = 0; i < N; ++i) {
        z_[i] = x_[i] + y_[i];
      }
    }
    double time = timer.seconds();
    return time;
  }
#endif

  void run_test(int R) {
    double bytes_moved = 1. * sizeof(double) * N * 3 * R;
    double GB          = bytes_moved / 1024 / 1024 / 1024;
    double time_kk     = kk_axpby(R);
    printf("AXPBY KK: %e s %e GB/s\n", time_kk, GB / time_kk);
#ifdef KOKKOS_ENABLE_SYCL
    double time_sycl = sycl_axpby(R);
    printf("AXPBY SYCL: %e s %e GB/s\n", time_sycl, GB / time_sycl);
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    double time_openmptarget = openmptarget_axpby(R);
    printf("AXPBY OpenMPTarget: %e s %e GB/s\n", time_openmptarget,
           GB / time_openmptarget);
#endif
  }
};
