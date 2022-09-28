/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>

template <typename InputView, typename OutputView>
void check_host_mirror(const InputView&, const OutputView&) {
  using InputSpace  = typename InputView::memory_space;
  using OutputSpace = typename OutputView::memory_space;
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                           InputSpace>::accessible)
    static_assert(std::is_same_v<OutputSpace, InputSpace>);
  else
    static_assert(std::is_same_v<OutputSpace, Kokkos::HostSpace>);
}

template <typename DeviceView, typename HostView>
void test_create_mirror_properties(const DeviceView& device_view,
                                   const HostView& host_view) {
  using DeviceMemorySpace    = typename DeviceView::memory_space;
  using HostMemorySpace      = typename HostView::memory_space;
  using DeviceExecutionSpace = typename DeviceView::execution_space;
  using HostExecutionSpace   = typename HostView::execution_space;

  // create_mirror
  {
    auto mirror =
        Kokkos::create_mirror(Kokkos::WithoutInitializing, device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing, host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing,
                                        DeviceExecutionSpace{}, device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(DeviceExecutionSpace{}, device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::WithoutInitializing,
                                        DeviceExecutionSpace{}, host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(DeviceExecutionSpace{}, host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view
  {
    auto mirror =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing, host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, DeviceExecutionSpace{}, device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror =
        Kokkos::create_mirror_view(DeviceExecutionSpace{}, device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                             DeviceExecutionSpace{}, host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(DeviceExecutionSpace{}, host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror view_alloc
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::view_alloc(), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::view_alloc(), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::view_alloc(DeviceMemorySpace{}),
                                        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(Kokkos::view_alloc(DeviceMemorySpace{}),
                                        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view view_alloc
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(Kokkos::WithoutInitializing), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(Kokkos::view_alloc(), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceMemorySpace{}), device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceMemorySpace{}), host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror view_alloc + execution space
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing),
        device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(HostExecutionSpace{}), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing),
        host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(HostExecutionSpace{}), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                           DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                           DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view view_alloc + execution space
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing),
        device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(HostExecutionSpace{}), device_view);
    check_host_mirror(device_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing),
        host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}), host_view);
    check_host_mirror(host_view, mirror);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                           DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                           DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view_and_copy
  {
    auto mirror =
        Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror =
        Kokkos::create_mirror_view_and_copy(HostMemorySpace{}, host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view_and_copy view_alloc
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(HostMemorySpace{}), device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(HostMemorySpace{}), host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceMemorySpace{}), device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceMemorySpace{}), host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }

  // create_mirror_view_and_copy view_alloc + execution space
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(HostMemorySpace{}, DeviceExecutionSpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(HostMemorySpace{}, DeviceExecutionSpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 HostMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        device_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
  {
    auto mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}),
        host_view);
    static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                                 DeviceMemorySpace>);
  }
}

TEST(TEST_CATEGORY, create_mirror_view_properties) {
  Kokkos::View<int*, TEST_EXECSPACE> device_view("device view", 10);
  Kokkos::View<int*, Kokkos::HostSpace> host_view("host view", 10);

  test_create_mirror_properties(device_view, host_view);
}
