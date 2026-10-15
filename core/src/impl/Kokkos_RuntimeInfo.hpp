// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_RUNTIME_INFO_HPP
#define KOKKOS_RUNTIME_INFO_HPP

#include <iostream>
#include <string>

namespace Kokkos {
/** \brief Print "Bill of Materials" */
void print_configuration(std::ostream& os, bool verbose = false);

[[nodiscard]] int device_id() noexcept;
[[nodiscard]] int num_devices() noexcept;
[[nodiscard]] int num_threads() noexcept;

bool show_warnings() noexcept;
bool tune_internals() noexcept;
}  // namespace Kokkos

namespace Kokkos::Impl {
void declare_configuration_metadata(const std::string& category,
                                    const std::string& key,
                                    const std::string& value);

}  // namespace Kokkos::Impl

#endif
