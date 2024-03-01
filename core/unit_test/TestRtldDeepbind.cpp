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

#include <Kokkos_Core.hpp>
#include <dlfcn.h>
#include <iostream>

double time_step(size_t N, double dx, const Kokkos::View<double*>& u,
                 const Kokkos::View<double*>& c);
using time_step_fptr_t = decltype(time_step)*;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  void* kernel_lib = dlopen("./libRtldDeepbind_TestKernel.so",
                            RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
  if (kernel_lib == nullptr) {
    std::cout << "Could not dlopen libRtldDeepbind_TestKernel.so.\n";
    return 1;
  }

  auto time_step_fptr =
      reinterpret_cast<time_step_fptr_t>(dlsym(kernel_lib, "time_step"));
  if (time_step_fptr == nullptr) return 2;

  {
    size_t N  = 10;
    double dx = 0.1;

    Kokkos::View<double*> u("u", N), c("c", N);

    time_step_fptr(N, dx, u, c);
  }

  Kokkos::finalize();
}
