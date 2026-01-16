// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INSTANCE_HPP
#define KOKKOS_NEXTSILICON_INSTANCE_HPP

#include <cstdint>
#include <iosfwd>
#include <string>
#include <memory>

namespace Kokkos::Experimental::Impl {

class NextSiliconInternal {
  bool m_is_initialized = false;

  NextSiliconInternal()                                      = default;
  NextSiliconInternal(const NextSiliconInternal&)            = delete;
  NextSiliconInternal& operator=(const NextSiliconInternal&) = delete;

 public:
  const static uint32_t m_ns_device_id = 0;
  static NextSiliconInternal* singleton();

  void initialize();
  void finalize();
  bool is_initialized() const;

  void print_configuration(std::ostream& os) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;
};

}  // namespace Kokkos::Experimental::Impl

#endif
