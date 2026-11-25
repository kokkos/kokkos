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

/*! \file md_view.cpp
 *  \brief Microbenchmark for multidimensional view index calculation overhead
 *
 *  This benchmark measures the performance impact of using different integer
 *  types (int32_t vs int64_t) for indexing operations in deeply nested loops
 *  accessing an 8-dimensional Kokkos view.
 *
 *  The test is designed to stress the address calculation machinery by:
 *  - Using an 8D view with dynamic dimensions to prevent compile-time
 * optimizations
 *  - Accessing data out of scratch to remove memory access overheads
 */

#include <chrono>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

using Clock  = std::chrono::steady_clock;
using Dur    = std::chrono::duration<double>;
using Scalar = double;

template <typename IndexType>
struct S {
  using member_type   = Kokkos::TeamPolicy<>::member_type;
  using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using ScratchView =
      Kokkos::View<Scalar*, scratch_space, Kokkos::MemoryUnmanaged>;

  Kokkos::View<Scalar*> output_;
  int iterations_;
  int dim0_;
  int dim1_;
  int dim2_;
  int dim3_;
  int dim4_;
  int dim5_;

  S(const Kokkos::View<Scalar*>& output, int iterations, int dim0, int dim1,
    int dim2, int dim3, int dim4, int dim5)
      : output_(output),
        iterations_(iterations),
        dim0_(dim0),
        dim1_(dim1),
        dim2_(dim2),
        dim3_(dim3),
        dim4_(dim4),
        dim5_(dim5) {}

  KOKKOS_INLINE_FUNCTION int total_size() const {
    return dim0_ * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * 3 * 3;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type& team_member) const {
    const int i = team_member.league_rank() * team_member.team_size() +
                  team_member.team_rank();

    // Get a 1D scratch view for the team
    ScratchView arr(team_member.team_scratch(0),
                    total_size() * team_member.team_size());

    // put something in scratch
    for (int idx = team_member.team_rank(); idx < total_size();
         idx += team_member.team_size()) {
      arr(idx) = 17 - idx;
    }
    team_member.team_barrier();

    // dynamic 8D view to maximize address calculation pressure
    using View8D = Kokkos::View<Scalar********, Kokkos::LayoutLeft,
                                Kokkos::DefaultExecutionSpace::memory_space,
                                Kokkos::MemoryUnmanaged>;
    View8D v(arr.data(), dim0_, dim1_, dim2_, dim3_, dim4_, dim5_, 3, 3);

    // generate lots and lots of integer instructions relative to memory and
    // floats
    Scalar sum = 0;
    for (int repeat = 0; repeat < iterations_; ++repeat) {
      for (IndexType i0 = 0; i0 < dim0_; ++i0) {
        for (IndexType i1 = 0; i1 < dim1_; ++i1) {
          for (IndexType i2 = 0; i2 < dim2_; ++i2) {
            for (IndexType i3 = 0; i3 < dim3_; ++i3) {
              for (IndexType i4 = 0; i4 < dim4_; ++i4) {
                for (IndexType i5 = 0; i5 < dim5_; ++i5) {
                  sum += v(i0, i1, i2, i3, i4, i5, 0, 0);
                  sum += v(i0, i1, i2, i3, i4, i5, 0, 1);
                  sum += v(i0, i1, i2, i3, i4, i5, 0, 2);
                  sum += v(i0, i1, i2, i3, i4, i5, 1, 0);
                  sum += v(i0, i1, i2, i3, i4, i5, 1, 1);
                  sum += v(i0, i1, i2, i3, i4, i5, 1, 2);
                  sum += v(i0, i1, i2, i3, i4, i5, 2, 0);
                  sum += v(i0, i1, i2, i3, i4, i5, 2, 1);
                  sum += v(i0, i1, i2, i3, i4, i5, 2, 2);
                }
              }
            }
          }
        }
      }
    }
    if (size_t(i) < output_.size()) {
      output_(i) = sum;
    }
  }

  size_t team_shmem_size(int teamSize) const {
    return ScratchView::shmem_size(total_size() * teamSize);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int n              = Kokkos::DefaultExecutionSpace{}.concurrency();
    constexpr int iterations = 100000;

    // header
    std::cout << "========================================\n";
    std::cout << "Multidimensional View Index Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Concurrency:     " << n << "\n";
    std::cout << "Iterations:      " << iterations << "\n";

    Kokkos::View<Scalar*> output("output", n);
    auto output_h = Kokkos::create_mirror_view(output);

    int teamSize = 1;
#if defined(KOKKOS_ENABLE_CUDA)
    teamSize = 128;
    std::cout << "Backend:         CUDA\n";
#elif defined(KOKKOS_ENABLE_HIP)
    teamSize = 64;  // 1.5x-ish as fast as 128 on MI300A
    std::cout << "Backend:         HIP\n";
#elif defined(KOKKOS_ENABLE_SYCL)
    teamSize = 64;  // untried
    std::cout << "Backend:         SYCL\n";
#else
    std::cout << "Backend:         Other\n";
#endif

    const int leagueSize = (output.size() + teamSize - 1) / teamSize;
    std::cout << "Team size:       " << teamSize << "\n";
    std::cout << "League size:     " << leagueSize << "\n";
    std::cout << "========================================\n";

    Kokkos::TeamPolicy<> policy(leagueSize, teamSize);

    // warmup
    std::cerr << "warming up int32_t/int64_t...\n";
    Kokkos::parallel_for(policy, S<int32_t>{output, 1, 4, 1, 1, 1, 1, 1});
    Kokkos::parallel_for(policy, S<int64_t>{output, 1, 4, 1, 1, 1, 1, 1});
    Kokkos::fence();
    std::cerr << "done\n";

    // actual test
    std::cerr << "Running int32_t test...\n";
    auto start = Clock::now();
    Kokkos::parallel_for(policy,
                         S<int32_t>{output, iterations, 4, 1, 1, 1, 1, 1});
    Kokkos::fence();
    const Dur elapsed32 = Clock::now() - start;
    Kokkos::deep_copy(output_h, output);
    std::cerr << "done (output[0]=" << output_h(0) << ")\n";

    // actual test
    std::cerr << "Running int64_t test...\n";
    start = Clock::now();
    Kokkos::parallel_for(policy,
                         S<int64_t>{output, iterations, 4, 1, 1, 1, 1, 1});
    Kokkos::fence();
    const Dur elapsed64 = Clock::now() - start;
    Kokkos::deep_copy(output_h, output);
    std::cerr << "done (output[0]=" << output_h(0) << ")\n";

    // Results
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "int32_t time:    " << elapsed32.count() << " s\n";
    std::cout << "int64_t time:    " << elapsed64.count() << " s\n";
    std::cout << "int32_t speedup: " << std::setprecision(2)
              << elapsed64.count() / elapsed32.count() << "x\n";
    std::cout << "========================================\n";
  }

  Kokkos::finalize();
  return 0;
}
