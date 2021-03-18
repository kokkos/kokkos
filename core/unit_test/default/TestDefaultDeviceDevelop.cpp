
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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <default/TestDefaultDeviceType_Category.hpp>
#include <Kokkos_DualView.hpp>

namespace Test {

// modify if we have other UVM enabled backends
#ifdef KOKKOS_ENABLE_CUDA  // OR other UVM builds
#ifdef KOKKOS_ENABLE_CUDA_UVM
#define UVM_ENABLED_BUILD
#endif
#endif

#ifdef UVM_ENABLED_BUILD
template <typename ExecSpace>
struct UVMSpaceFor;
#endif

#ifdef KOKKOS_ENABLE_CUDA  // specific to CUDA
#ifdef KOKKOS_ENABLE_CUDA_UVM
template <>
struct UVMSpaceFor<Kokkos::Cuda> {
  using type = Kokkos::CudaUVMSpace;
};
#endif
#endif

#ifdef UVM_ENABLED_BUILD
template <>
struct UVMSpaceFor<Kokkos::DefaultHostExecutionSpace> {
  using type = typename UVMSpaceFor<Kokkos::DefaultExecutionSpace>::type;
};
#else
template <typename ExecSpace>
struct UVMSpaceFor {
  using type = typename ExecSpace::memory_space;
};
#endif

TEST(defaultdevicetype, development_test) {
  using ExecSpace  = Kokkos::DefaultExecutionSpace;
  using MemSpace   = typename UVMSpaceFor<Kokkos::DefaultExecutionSpace>::type;
  using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

  using DualViewType =
      Kokkos::DualView<double*, Kokkos::LayoutLeft, DeviceType>;

  DualViewType dv("myView", 100);
  dv.clear_sync_state();
  using d_device = DeviceType;
  using h_device = Kokkos::Device<
      Kokkos::DefaultHostExecutionSpace,
      typename UVMSpaceFor<Kokkos::DefaultHostExecutionSpace>::type>;
  {
    auto v_d = dv.view_device();
    std::cout << "Device view type: " << typeid(v_d).name() << '\n';
    std::cout << "      (its device_type): "
              << typeid(typename decltype(v_d)::device_type).name() << '\n';
  }
  {
    auto v_h = dv.view_host();
    std::cout << "Host view type: " << typeid(v_h).name() << '\n';
    std::cout << "      (its device_type): "
              << typeid(typename decltype(v_h)::device_type).name() << '\n';
  }
  std::cout << "\n";
  std::cout << "Marking host modified with templated version.\n";
  dv.template modify<h_device>();
  std::cout << "Need sync host/device: " << dv.need_sync_host() << "/"
            << dv.need_sync_device() << '\n';
  std::cout << "Clearing sync state.\n";
  dv.clear_sync_state();
  std::cout << "Marking device modified with templated version.\n";
  dv.template modify<d_device>();
  std::cout << "Need sync host/device: " << dv.need_sync_host() << "/"
            << dv.need_sync_device() << '\n';
  std::cout << "Clearing sync state.\n";
  dv.clear_sync_state();
  std::cout << "\n";
  std::cout << "Getting device view using templated version.\n";
  {
    auto v_d      = dv.template view<d_device>();
    using vdt     = decltype(v_d);
    using vdt_d   = vdt::device_type;
    using vdt_d_e = vdt_d::execution_space;

    ASSERT_TRUE(vdt_d_e::name() == Kokkos::DefaultExecutionSpace::name());
    std::cout << "Its device_type: "
              << typeid(typename decltype(v_d)::device_type).name() << '\n';
  }
  std::cout << "Getting host view using templated version.\n";
  {
    auto v_h = dv.template view<h_device>();
    std::cout << "Its device_type: "
              << typeid(typename decltype(v_h)::device_type).name() << '\n';
    using vht     = decltype(v_h);
    using vht_d   = vht::device_type;
    using vht_d_e = vht_d::execution_space;
    ASSERT_TRUE(vht_d_e::name() == Kokkos::DefaultHostExecutionSpace::name());
  }
  std::cout << "\n";
  std::cout
      << "Marking host modified, then doing sync<Device<DefaultExecutionSpace, "
         "UVMSpace>>\n";
  {
    dv.modify_host();
    dv.template sync<d_device>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  std::cout
      << "Marking host modified, then doing sync<DefaultExecutionSpace>\n";
  {
    dv.modify_host();
    dv.template sync<typename d_device::execution_space>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  std::cout << "Marking device modified, then doing sync<Device<Serial, "
               "UVMSpace>>\n";
  {
    dv.modify_device();
    dv.template sync<h_device>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  std::cout << "Marking device modified, then doing "
               "sync<DefaultHostExecutionSpace>\n";
  {
    dv.modify_device();
    dv.template sync<typename h_device::execution_space>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  std::cout << "Marking host modified, then doing sync<UVMSpace>\n";
  {
    dv.modify_host();
    dv.template sync<
        typename UVMSpaceFor<Kokkos::DefaultExecutionSpace>::type>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  std::cout << "Marking device modified, then doing sync<UVMSpace>\n";
  {
    dv.modify_device();
    dv.template sync<
        typename UVMSpaceFor<Kokkos::DefaultExecutionSpace>::type>();
    std::cout << "Now, need sync host/device: " << dv.need_sync_host() << "/"
              << dv.need_sync_device() << '\n';
    dv.clear_sync_state();
  }
  {
    using hvt = decltype(dv.view<typename Kokkos::DefaultHostExecutionSpace>());
    using dvt = decltype(dv.view<typename Kokkos::DefaultExecutionSpace>());
    ASSERT_TRUE(Kokkos::DefaultExecutionSpace::name() ==
                dvt::device_type::execution_space::name());
    ASSERT_TRUE(Kokkos::DefaultHostExecutionSpace::name() ==
                hvt::device_type::execution_space::name());
  }
}

}  // namespace Test
