// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INSTANCE_HPP
#define KOKKOS_NEXTSILICON_INSTANCE_HPP

#include <impl/Kokkos_HostSharedPtr.hpp>

#include <cstdint>
#include <iosfwd>
#include <string>

namespace Kokkos::Experimental::Impl {

class NextSiliconInternal {
  NextSiliconInternal(const NextSiliconInternal&)            = delete;
  NextSiliconInternal& operator=(const NextSiliconInternal&) = delete;

 public:
  static Kokkos::Impl::HostSharedPtr<NextSiliconInternal> default_instance;

  NextSiliconInternal();
  ~NextSiliconInternal();

  void print_configuration(std::ostream& os) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;
};

}  // namespace Kokkos::Experimental::Impl

#endif
