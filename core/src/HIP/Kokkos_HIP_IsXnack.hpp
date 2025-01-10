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

/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_HIP_ISXNACK_HPP
#define KOKKOS_HIP_ISXNACK_HPP

#include <string>
#include <optional>
#include <fstream>

#ifdef __linux__
#include <stdio.h>
#include <sys/utsname.h>
#endif

namespace Kokkos::Impl {

inline bool is_hsa_xnack_1_impl() {
  const char* envVar = std::getenv("HSA_XNACK");
  return envVar != nullptr && std::string(envVar) == "1";
}

// returns true iff environment variable HSA_XNACK=1
inline bool is_hsa_xnack_1() {
  static bool cache = is_hsa_xnack_1_impl();
  return cache;
}

#ifdef __linux__
// try to get `uname -r`. Returns an empty optional for any problem
inline std::optional<std::string> uname_r() {
  struct utsname buffer;
  if (uname(&buffer) != 0) {
    return std::optional<std::string>{};
  }
  return std::optional<std::string>{buffer.release};
}
#endif

inline bool is_boot_config_hmm_mirror_y_impl() {
#ifdef __linux__
  // figure out the boot config file name
  std::optional<std::string> unameR = uname_r();
  if (!unameR) {
    // couldn't figure out linux release name
    return false;
  }
  std::string bootConfigPath = std::string("/boot/config-") + unameR.value();

  std::ifstream file(bootConfigPath);
  if (!file.is_open()) {
    // couldn't open file for whatever reason
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.find("CONFIG_HMM_MIRROR=y") != std::string::npos) {
      return true;  // Found the string
    }
  }
  return false;

#else
  return false;
#endif
}

// return true iff `CONFIG_HMM_MIRROR=y` is definitely in /boot/config-$(uname
// -r) returns false for non-linux platforms or any other problem
inline bool is_boot_config_hmm_mirror_y() {
  static bool cache = is_boot_config_hmm_mirror_y_impl();
  return cache;
}

}  // namespace Kokkos::Impl

#endif  // KOKKOS_HIP_ISXNACK_HPP
