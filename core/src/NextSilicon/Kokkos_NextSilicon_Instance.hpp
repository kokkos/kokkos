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

#include <nextapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon_HeapBuffer.hpp>
#include <NextSilicon/Kokkos_NextSilicon_PageAlignedData.hpp>

namespace Kokkos::Experimental::Impl {

class NextSiliconInternal {
  Impl::NextSiliconHeapBuffer m_functor_buffer;
  Impl::NextSiliconHeapBuffer m_league_scratch_buffer;
  Impl::NextSiliconHeapBuffer m_reduce_partial_buffer;
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
  Driver* clone_driver(const Driver& driver) {
    // Helper to clone the driver before going into the handoff function. This
    // prevents the stack from getting migrated into device.
    // Kokkos does not require:
    // - functor copy-ctor run when functor moves to device
    // - functor dtor runs at some point on the device
    // nor does CUDA do either of those things.
    // Just memcpy the functor bytes before execution is okay.
    static_assert(alignof(Driver) >= 64,
                  "Driver must be at least 64B aligned for BMT-enabled "
                  "nextapi_memory_copy");
    size_t functor_size = sizeof(Driver);
    std::byte* buffer   = resize_functor_buffer(functor_size);
    nextapi_memory_copy(buffer, &driver, functor_size);
    return reinterpret_cast<Driver*>(buffer);
  }

  std::byte* resize_league_scratch_buffer(size_t requested);
  std::byte* resize_reduce_partial_buffer(size_t requested);

  void print_configuration(std::ostream& os) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;

  [[nodiscard]] auto lock_device() {
    KOKKOS_IF_ON_DEVICE(
        (KOKKOS_ASSERT(false &&
                       "lock_device should never be called on device");))
    return std::lock_guard<std::recursive_mutex>(this->device_mutex_);
  }
};

}  // namespace Kokkos::Experimental::Impl

#endif
