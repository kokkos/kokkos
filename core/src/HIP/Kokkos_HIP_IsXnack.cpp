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

#include <string>
#include <string_view>
#include <optional>
#include <fstream>

#ifdef __linux__
#include <stdio.h>
#include <sys/utsname.h>
#endif

namespace {

#ifdef __linux__
// try to get `uname -r`. Returns an empty optional for any problem
std::optional<std::string> uname_r() {
  struct utsname buffer;
  if (uname(&buffer) != 0) {
    return std::nullopt;
  }
  return std::optional<std::string>{buffer.release};
}
#endif

// returns true iff environment variable HSA_XNACK=1
bool hsa_xnack_enabled_in_host_environment() {
  const char* var = std::getenv("HSA_XNACK");
  return var && std::string_view{var} == "1";
}

// return true iff `CONFIG_HMM_MIRROR=y` is definitely in /boot/config-$(uname
// -r) returns false for non-linux platforms or any other problem
bool config_hmm_mirror_in_boot_config() {
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
#else   // __linux__
  return false;
#endif  // __linux__
}

}  // namespace

namespace Kokkos::Impl {

bool xnack_enabled() {
  static bool cache = [] {
    return config_hmm_mirror_in_boot_config() &&
           hsa_xnack_enabled_in_host_environment();
  }();
  return cache;
}

}  // namespace Kokkos::Impl
