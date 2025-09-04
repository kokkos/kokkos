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

#ifndef KOKKOS_TIMER_HPP
#define KOKKOS_TIMER_HPP

#include <Kokkos_Macros.hpp>

#include <chrono>

namespace Kokkos {

class Timer {
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;

 public:
  Timer(const Timer&)            = delete;
  Timer& operator=(const Timer&) = delete;

  Timer() { reset(); }

  void reset() { start_ = Clock::now(); }

  double seconds() const {
    using namespace std::chrono;
    auto const now = Clock::now();
    return duration_cast<duration<double>>(now - start_).count();
  }
};

}  // namespace Kokkos

#endif
