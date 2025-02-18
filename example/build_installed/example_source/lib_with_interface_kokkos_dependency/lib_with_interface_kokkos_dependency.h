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

#ifndef LIB_WITH_INTERFACE_KOKKOS_DEPENDENCY
#define LIB_WITH_INTERFACE_KOKKOS_DEPENDENCY

#include <Kokkos_Core.hpp>
#if defined __CUDACC__ || defined __HIPCC__
#include <cuda_hip_header_without_kokkos_dependency.cuh>
#endif

#include <iostream>

namespace lib_with_interface_kokkos_dependency {

template<typename ViewType>
void print(ViewType a)
    {
  static_assert(std::is_same_v<Kokkos::View<int*>, ViewType>, "ViewType must match Kokkos::View<int*>");
  std::cout << "Hello from lib_with_interface_kokkos_dependency, printig view a(0) " << a(0) << "\n";
#if defined __CUDACC__ || defined __HIPCC__
  std::cout << "Calling additional device function without kokkos dependency\n";
  cuda_hip_header_without_kokkos_dependency::print_from_device<<<1,1>>>(1.0);
#endif
  }

template<typename ViewType>
struct StructOfLibWithInterfaceKokkosDependency {
  ViewType value;
};

}  // namespace lib_with_interface_kokkos_dependency

#endif
