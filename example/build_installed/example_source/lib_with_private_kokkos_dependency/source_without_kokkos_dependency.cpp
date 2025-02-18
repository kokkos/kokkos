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

#include "header_without_kokkos_dependency.h"
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif
#include <iostream>

#if defined __CUDACC__ || defined __HIPCC__
namespace cuda_hip_functions_without_kokkos_dependency{
__global__
void print_from_device();
}
#endif

namespace lib_with_private_kokkos_dependency {

void print_non_kokkos() {
  std::cout << "Hello from non-kokkos function inside "
               "lib_with_private_kokkos_dependency\n";
#if defined __CUDACC__ || defined __HIPCC__
  std::cout << "Calling additional device function without kokkos dependency\n";
  cuda_hip_functions_without_kokkos_dependency::print_from_device<<<1,1>>>();
#endif
}

}  // namespace lib_with_private_kokkos_dependency
