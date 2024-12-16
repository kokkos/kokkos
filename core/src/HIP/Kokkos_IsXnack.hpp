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

#ifndef KOKKOS_IMPL_ISXNACK_HPP
#define KOKKOS_IMPL_ISXNACK_HPP

#include <cstdlib>  //getenv
#include <string>
#include <fstream>
#include <optional>

#include <impl/Kokkos_StringManipulation.hpp>

namespace Kokkos::Impl {

// return true if HSA_XNACK is set in the environment
inline bool is_xnack_enabled_from_env() {
  static std::optional<bool> result;
  if (!result.has_value()) {
    const char* hsa_xnack = std::getenv("HSA_XNACK");
    if (!hsa_xnack) {
      result = false;
    } else if (Kokkos::Impl::strcmp(hsa_xnack, "1") != 0) {
      result = false;
    } else {
      result = true;
    }
  }
  return result.value();
}

// return true if /proc/cmdline is readable and contains "amdgpu.noretry=0"
inline bool is_xnack_enabled_from_cmdline() {
  static std::optional<bool> result;
  if (!result.has_value()) {
    std::ifstream cmdline("/proc/cmdline");

    if (!cmdline.is_open()) {
      result = false;
    } else {
      std::string line;
      if (!std::getline(cmdline, line)) {
        result = false;
      } else {
        result = line.find("amdgpu.noretry=0") != std::string::npos;
      }
      cmdline.close();
    }
  }
  return result.value();
}

inline bool is_xnack() {
  return is_xnack_enabled_from_env() || is_xnack_enabled_from_cmdline();
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_ISXNACK_HPP
