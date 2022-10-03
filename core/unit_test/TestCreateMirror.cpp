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

template <typename TestView, typename MemorySpace>
void check_memory_space(TestView, MemorySpace) {
  static_assert(std::is_same_v<typename TestView::memory_space, MemorySpace>);
}

template <class View>
auto host_mirror_test_space(View) {
  return std::conditional_t<
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 typename View::memory_space>::accessible,
      typename View::memory_space, Kokkos::HostSpace>{};
}

template <typename DeviceView, typename HostView>
void test_create_mirror_properties(const DeviceView& device_view,
                                   const HostView& host_view) {
  using DeviceMemorySpace    = typename DeviceView::memory_space;
  using HostMemorySpace      = typename HostView::memory_space;
  using DeviceExecutionSpace = typename DeviceView::execution_space;
  using HostExecutionSpace   = typename HostView::execution_space;
  using namespace Kokkos;

  // clang-format off
  
  // create_mirror
  check_memory_space(create_mirror(WithoutInitializing,                         device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(                                             device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(WithoutInitializing,                         host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror(                                             host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror(WithoutInitializing, DeviceExecutionSpace{}, device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(                     DeviceExecutionSpace{}, device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(WithoutInitializing, DeviceExecutionSpace{}, host_view),   DeviceMemorySpace{});
  check_memory_space(create_mirror(                     DeviceExecutionSpace{}, host_view),   DeviceMemorySpace{});

  // create_mirror_view
  check_memory_space(create_mirror_view(WithoutInitializing,                         device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(                                             device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(WithoutInitializing,                         host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror_view(                                             host_view),   host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(WithoutInitializing, DeviceExecutionSpace{}, device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(                     DeviceExecutionSpace{}, device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(WithoutInitializing, DeviceExecutionSpace{}, host_view),   DeviceMemorySpace{});
  check_memory_space(create_mirror_view(                     DeviceExecutionSpace{}, host_view),   DeviceMemorySpace{});

  // create_mirror view_alloc
  check_memory_space(create_mirror(view_alloc(WithoutInitializing),                      device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(view_alloc(),                                         device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(view_alloc(WithoutInitializing),                      host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror(view_alloc(),                                         host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror(view_alloc(WithoutInitializing, DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(                     DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(WithoutInitializing, DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(                     DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});

  // create_mirror_view view_alloc
  check_memory_space(create_mirror_view(view_alloc(WithoutInitializing),                      device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(view_alloc(),                                         device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(view_alloc(WithoutInitializing),                      host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror_view(view_alloc(),                                         host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror_view(view_alloc(WithoutInitializing, DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(                     DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(WithoutInitializing, DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(                     DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});

  // create_mirror view_alloc + execution space
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{}, WithoutInitializing),                      device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(view_alloc(HostExecutionSpace{}),                                             device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{}, WithoutInitializing),                      host_view), host_mirror_test_space(host_view));
  check_memory_space(create_mirror(view_alloc(HostExecutionSpace{}),                                             host_view), host_mirror_test_space(host_view));
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{},                      DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), host_view), DeviceMemorySpace{});
  check_memory_space(create_mirror(view_alloc(DeviceExecutionSpace{},                      DeviceMemorySpace{}), host_view), DeviceMemorySpace{});

  // create_mirror_view view_alloc + execution space
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{}, WithoutInitializing),                      device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(view_alloc(HostExecutionSpace{}),                                             device_view), host_mirror_test_space(device_view));
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{}, WithoutInitializing),                      host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror_view(view_alloc(HostExecutionSpace{}),                                             host_view),   host_mirror_test_space(host_view));
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{},                      DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{}, WithoutInitializing, DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});
  check_memory_space(create_mirror_view(view_alloc(DeviceExecutionSpace{},                      DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});

  // create_mirror_view_and_copy
  check_memory_space(create_mirror_view_and_copy(HostMemorySpace{},   device_view), HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(HostMemorySpace{},   host_view),   HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(DeviceMemorySpace{}, device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(DeviceMemorySpace{}, host_view),   DeviceMemorySpace{});

  // create_mirror_view_and_copy view_alloc
  check_memory_space(create_mirror_view_and_copy(view_alloc(HostMemorySpace{}),   device_view), HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(HostMemorySpace{}),   host_view),   HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}), host_view),   DeviceMemorySpace{});

  // create_mirror_view_and_copy view_alloc + execution space
  check_memory_space(create_mirror_view_and_copy(view_alloc(HostMemorySpace{},   HostExecutionSpace{}),   device_view), HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(HostMemorySpace{},   HostExecutionSpace{}),   host_view),   HostMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}, DeviceExecutionSpace{}), device_view), DeviceMemorySpace{});
  check_memory_space(create_mirror_view_and_copy(view_alloc(DeviceMemorySpace{}, DeviceExecutionSpace{}), host_view),   DeviceMemorySpace{});

  // clang-format on
}

void test() {
  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> device_view("device view",
                                                                10);
  Kokkos::View<int*, Kokkos::HostSpace> host_view("host view", 10);

  test_create_mirror_properties(device_view, host_view);
}
