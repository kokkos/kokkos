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
#include <Kokkos_DynamicView.hpp>
#include <Kokkos_DynRankView.hpp>
#include <Kokkos_OffsetView.hpp>

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

template <typename View, typename... Args>
void check_create_mirror_host(const View& view, Args&&... args) {
  auto mirror = Kokkos::create_mirror(std::forward<Args>(args)..., view);
  check_host_mirror(view, mirror);
}

template <typename DeviceMemorySpace, typename View, typename... Args>
void check_create_mirror_device(const View& view, Args&&... args) {
  auto mirror = Kokkos::create_mirror(std::forward<Args>(args)..., view);
  static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                               DeviceMemorySpace>);
}

template <typename View, typename... Args>
void check_create_mirror_view_host(const View& view, Args&&... args) {
  auto mirror = Kokkos::create_mirror_view(std::forward<Args>(args)..., view);
  check_host_mirror(view, mirror);
}

template <typename DeviceMemorySpace, typename View, typename... Args>
void check_create_mirror_view_device(const View& view, Args&&... args) {
  auto mirror = Kokkos::create_mirror_view(std::forward<Args>(args)..., view);
  static_assert(std::is_same_v<typename decltype(mirror)::memory_space,
                               DeviceMemorySpace>);
}

template <typename MemorySpace, typename View, typename... Args>
void check_create_mirror_view_and_copy(const View& view, Args&&... args) {
  auto mirror =
      Kokkos::create_mirror_view_and_copy(std::forward<Args>(args)..., view);
  static_assert(
      std::is_same_v<typename decltype(mirror)::memory_space, MemorySpace>);
}

template <typename DeviceView, typename HostView>
void test_create_mirror_properties(const DeviceView& device_view,
                                   const HostView& host_view) {
  using DeviceMemorySpace    = typename DeviceView::memory_space;
  using HostMemorySpace      = typename HostView::memory_space;
  using DeviceExecutionSpace = typename DeviceView::execution_space;
  using HostExecutionSpace   = typename HostView::execution_space;

  // create_mirror
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_host(device_view, Kokkos::WithoutInitializing);
    check_create_mirror_host(device_view);
  }
  check_create_mirror_host(host_view, Kokkos::WithoutInitializing);
  check_create_mirror_host(host_view);
  check_create_mirror_device<DeviceMemorySpace>(
      device_view, Kokkos::WithoutInitializing, DeviceExecutionSpace{});
  check_create_mirror_device<DeviceMemorySpace>(device_view,
                                                DeviceExecutionSpace{});
  check_create_mirror_device<DeviceMemorySpace>(
      host_view, Kokkos::WithoutInitializing, DeviceExecutionSpace{});
  check_create_mirror_device<DeviceMemorySpace>(host_view,
                                                DeviceExecutionSpace{});

  // create_mirror_view
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_view_host(device_view, Kokkos::WithoutInitializing);
    check_create_mirror_view_host(device_view);
  }
  check_create_mirror_view_host(host_view, Kokkos::WithoutInitializing);
  check_create_mirror_view_host(host_view);
  check_create_mirror_view_device<DeviceMemorySpace>(
      device_view, Kokkos::WithoutInitializing, DeviceExecutionSpace{});
  check_create_mirror_view_device<DeviceMemorySpace>(device_view,
                                                     DeviceExecutionSpace{});
  check_create_mirror_view_device<DeviceMemorySpace>(
      host_view, Kokkos::WithoutInitializing, DeviceExecutionSpace{});
  check_create_mirror_view_device<DeviceMemorySpace>(host_view,
                                                     DeviceExecutionSpace{});

  // create_mirror view_alloc
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_host(device_view,
                             Kokkos::view_alloc(Kokkos::WithoutInitializing));
    check_create_mirror_host(device_view, Kokkos::view_alloc());
  }
  check_create_mirror_host(host_view,
                           Kokkos::view_alloc(Kokkos::WithoutInitializing));
  check_create_mirror_host(host_view, Kokkos::view_alloc());
  check_create_mirror_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      device_view, Kokkos::view_alloc(DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      host_view, Kokkos::view_alloc(DeviceMemorySpace{}));

  // create_mirror_view view_alloc
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_view_host(
        device_view, Kokkos::view_alloc(Kokkos::WithoutInitializing));
    check_create_mirror_view_host(device_view, Kokkos::view_alloc());
  }
  check_create_mirror_view_host(
      host_view, Kokkos::view_alloc(Kokkos::WithoutInitializing));
  check_create_mirror_view_host(host_view, Kokkos::view_alloc());
  check_create_mirror_view_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      device_view, Kokkos::view_alloc(DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(Kokkos::WithoutInitializing, DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      host_view, Kokkos::view_alloc(DeviceMemorySpace{}));

  // create_mirror view_alloc + execution space
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_host(device_view,
                             Kokkos::view_alloc(DeviceExecutionSpace{},
                                                Kokkos::WithoutInitializing));
    check_create_mirror_host(device_view,
                             Kokkos::view_alloc(HostExecutionSpace{}));
  }
  check_create_mirror_host(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing));
  check_create_mirror_host(host_view, Kokkos::view_alloc(HostExecutionSpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                         DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                         DeviceMemorySpace{}));
  check_create_mirror_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}));

  // create_mirror_view view_alloc + execution space
  // FIXME DynamicView: HostMirror is the same type
  if constexpr (!Kokkos::is_dynamic_view<DeviceView>::value) {
    check_create_mirror_view_host(
        device_view, Kokkos::view_alloc(DeviceExecutionSpace{},
                                        Kokkos::WithoutInitializing));
    check_create_mirror_view_host(device_view,
                                  Kokkos::view_alloc(HostExecutionSpace{}));
  }
  check_create_mirror_view_host(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing));
  check_create_mirror_view_host(host_view,
                                Kokkos::view_alloc(HostExecutionSpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                         DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, Kokkos::WithoutInitializing,
                         DeviceMemorySpace{}));
  check_create_mirror_view_device<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(DeviceExecutionSpace{}, DeviceMemorySpace{}));

  // create_mirror_view_and_copy
  check_create_mirror_view_and_copy<HostMemorySpace>(device_view,
                                                     HostMemorySpace{});
  check_create_mirror_view_and_copy<HostMemorySpace>(host_view,
                                                     HostMemorySpace{});
  check_create_mirror_view_and_copy<DeviceMemorySpace>(device_view,
                                                       DeviceMemorySpace{});
  check_create_mirror_view_and_copy<DeviceMemorySpace>(host_view,
                                                       DeviceMemorySpace{});

  // create_mirror_view_and_copy view_alloc
  check_create_mirror_view_and_copy<HostMemorySpace>(
      device_view, Kokkos::view_alloc(HostMemorySpace{}));
  check_create_mirror_view_and_copy<HostMemorySpace>(
      host_view, Kokkos::view_alloc(HostMemorySpace{}));
  check_create_mirror_view_and_copy<DeviceMemorySpace>(
      device_view, Kokkos::view_alloc(DeviceMemorySpace{}));
  check_create_mirror_view_and_copy<DeviceMemorySpace>(
      host_view, Kokkos::view_alloc(DeviceMemorySpace{}));

  // create_mirror_view_and_copy view_alloc + execution space
  check_create_mirror_view_and_copy<HostMemorySpace>(
      device_view, Kokkos::view_alloc(HostMemorySpace{}, HostExecutionSpace{}));
  check_create_mirror_view_and_copy<HostMemorySpace>(
      host_view, Kokkos::view_alloc(HostMemorySpace{}, HostExecutionSpace{}));
  check_create_mirror_view_and_copy<DeviceMemorySpace>(
      device_view,
      Kokkos::view_alloc(DeviceMemorySpace{}, DeviceExecutionSpace{}));
  check_create_mirror_view_and_copy<DeviceMemorySpace>(
      host_view,
      Kokkos::view_alloc(DeviceMemorySpace{}, DeviceExecutionSpace{}));
}

TEST(TEST_CATEGORY, create_mirror_dynrankview) {
  Kokkos::DynRankView<int, TEST_EXECSPACE> device_view("device view", 10);
  Kokkos::DynRankView<int, Kokkos::HostSpace> host_view("host view", 10);

  test_create_mirror_properties(device_view, host_view);
}

TEST(TEST_CATEGORY, create_mirror_offsetview) {
  Kokkos::Experimental::OffsetView<int*, TEST_EXECSPACE> device_view(
      "device view", {0, 10});
  Kokkos::Experimental::OffsetView<int*, Kokkos::HostSpace> host_view(
      "host view", {0, 10});

  test_create_mirror_properties(device_view, host_view);
}

TEST(TEST_CATEGORY, create_mirror_dynamicview) {
  Kokkos::Experimental::DynamicView<int*, TEST_EXECSPACE> device_view(
      "device view", 2, 10);
  Kokkos::Experimental::DynamicView<int*, Kokkos::HostSpace> host_view(
      "host view", 2, 10);

  test_create_mirror_properties(device_view, host_view);
}
