// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_INSTANCE_HPP
#define KOKKOS_NEXTSILICON_INSTANCE_HPP

#include <impl/Kokkos_HostSharedPtr.hpp>

#include <cstdint>
#include <cstddef>
#include <iosfwd>
#include <mutex>
#include <string>
#include <memory>

#include <NextSilicon/Kokkos_NextSilicon_HeapBuffer.hpp>
#include <NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp>

namespace Kokkos::Experimental {
// Forward declaration prevents cyclic dependency imports
class NextSilicon;
}  // namespace Kokkos::Experimental

namespace Kokkos::Experimental::Impl {

class NextSiliconInternal {
  Impl::NextSiliconHeapBuffer functorBuffer_;
  Impl::NextSiliconHeapBuffer leagueScratchBuffer_;
  Impl::NextSiliconHeapBuffer reducePartialBuffer_;
  ::Kokkos::Impl::PageAlignedData<std::recursive_mutex,
                                  ::Kokkos::Impl::PageLocation::Host>
      device_mutex_;

  NextSiliconInternal(const NextSiliconInternal&)            = delete;
  NextSiliconInternal& operator=(const NextSiliconInternal&) = delete;

  std::byte* resize_functor_buffer(size_t requested);

 public:
  static Kokkos::Impl::HostSharedPtr<NextSiliconInternal> default_instance;

  NextSiliconInternal();
  ~NextSiliconInternal();

  template <class Driver>
  auto clone_driver(const Driver& driver) {
    // Helper to clone the driver before going into the handoff function. This
    // prevents the stack from getting migrated into device.
    size_t functor_size = sizeof(Driver);
    std::byte* buffer   = resize_functor_buffer(functor_size);
    auto deleter        = [](Driver* const p) {
      if (p) p->~Driver();
    };
    return std::unique_ptr<Driver, decltype(deleter)>(
        new (buffer) Driver(driver), deleter);
  }

  std::byte* resize_league_scratch_buffer(size_t requested);
  std::byte* resize_reduce_partial_buffer(size_t requested);

  void print_configuration(std::ostream& os) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;

  [[nodiscard]] std::lock_guard<std::recursive_mutex> lock_device();
};

}  // namespace Kokkos::Experimental::Impl

#endif
