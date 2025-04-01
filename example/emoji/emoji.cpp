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

#include <iostream>

#include <Kokkos_Emoji.hpp>

int main(int 🧹🧮, char* 🧹[]) {
  🫘::🔭💂 🫘🔭(🧹🧮, 🧹);

  🫘::🖨️🛠️(std::cout);

  🫘::👁️<int*> 🧍👁️("my view", 10);
  auto 🧍🪞👁️ = 🫘::🏗️🪞👁️(🧍👁️);

  🫘::🚅🚅(
      "initialization", 🫘::📏🚔<🫘::💘🏠🔪🌌>(0, 10), 🫘🐑 (int const 👆) {
        🧍🪞👁️(👆) = 👆;
      });
  🫘::🚧("wait initialization");

  🫘::🕳️👯(🧍👁️, 🧍🪞👁️);

  int 💰;
  🫘::🚅🌱(
      "reduction",
      10, 🫘🐑 (int const 👆, int& 🏡💰) { 🏡💰 += 🧍👁️(👆); }, 💰);

  🫘::🖨️📄("sum is %i 💅\n", 💰);
}
