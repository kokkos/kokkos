// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_HPP
#define KOKKOS_NEXTSILICON_HPP

#include <nsapi/intrinsics.h>

#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>

namespace Kokkos::Experimental::Impl {
class NextSiliconInternal;
}  // namespace Kokkos::Experimental::Impl

namespace Kokkos::Experimental {

class NextSilicon {
  Impl::NextSiliconInternal* m_space_instance;

  friend bool operator==(NextSilicon const& lhs, NextSilicon const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(NextSilicon const& lhs, NextSilicon const& rhs) {
    return !(lhs == rhs);
  }

 public:
  using execution_space = NextSilicon;
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_UNIFIED_MEMORY)
  using memory_space = Kokkos::Experimental::NextSiliconSharedSpace;
#else
  using memory_space = Kokkos::Experimental::NextSiliconSpace;
#endif
  using device_type = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<NextSilicon>;

  NextSilicon();

  static void impl_initialize(InitializationSettings const& settings);
  static void impl_finalize();
  static bool impl_is_initialized();

  void print_configuration(std::ostream& os, bool verbose = false) const;

  void fence(std::string const& name =
                 "Kokkos::NextSilicon::fence(): Unnamed Instance Fence") const;
  static void impl_static_fence(std::string const& name);

  static char const* name() { return "NextSilicon"; }
  static int concurrency() {
    return 64 * 1024; /* FIXME_NEXTSILICON - move to nsapi call */
  }
  static bool in_parallel() {
    // FIXME_NEXTSILICON: being on the grid is not the same as being in a
    // parallel region
    return __nsapi_is_on_cg();
  }

  static int impl_hardware_thread_id() noexcept;

  uint32_t impl_instance_id() const noexcept;
  Impl::NextSiliconInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

  int ns_device_id() const noexcept;
};

}  // namespace Kokkos::Experimental

namespace Kokkos::Impl {

template <>
struct MemorySpaceAccess<
    Kokkos::Experimental::NextSiliconSpace,
    Kokkos::Experimental::NextSilicon::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

template <>
struct MemorySpaceAccess<
    Kokkos::Experimental::NextSiliconSharedSpace,
    Kokkos::Experimental::NextSilicon::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};

}  // namespace Kokkos::Impl

template <>
struct Kokkos::Tools::Experimental::DeviceTypeTraits<
    ::Kokkos::Experimental::NextSilicon> {
  static constexpr DeviceType id =
      ::Kokkos::Profiling::Experimental::DeviceType::NextSilicon;

  static int device_id(const Kokkos::Experimental::NextSilicon& exec) {
    return exec.ns_device_id();
  }
};

#endif  // KOKKOS_NEXTSILICON_HPP
