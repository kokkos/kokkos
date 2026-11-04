// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <arm_sve.h>

int main() {
  const int vl = svcntb() * 8;
  std::cout << "SVE_HW_VL=" << vl << std::endl;
  return 0;
}
