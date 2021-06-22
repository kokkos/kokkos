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
#include <iostream>
#include <gtest/gtest.h>
#include "Kokkos_Core.hpp"

#include <impl/Kokkos_Stacktrace.hpp>
#include <vector>
#include <algorithm>
namespace Kokkos {
class Serial;
class OpenMP;
class Cuda;
class Threads;
namespace Experimental {
class SYCL;
class HIP;
class OpenMPTarget;
class HPX;
}  // namespace Experimental
}  // namespace Kokkos
namespace Test {
/**
void expect_allocation_event(const std::string evn, const std::string esn,
                             const std::string em) {
  expected_view_name  = evn;
  expected_space_name = esn;
  error_message       = em;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle hand, const char* name, const void*,
         const uint64_t) {
        ASSERT_EQ(std::string(hand.name), expected_space_name)
            << error_message << " (bad handle)";
        ASSERT_EQ(std::string(name), expected_view_name)
            << error_message << " (bad view name)";
        expect_no_events();
      });
}
*/
struct FencePayload {
  std::string name;
  enum distinguishable_devices { yes, no };
  distinguishable_devices distinguishable;
  uint32_t dev_id;
};

std::vector<FencePayload> found_payloads;
template <typename Lambda>
void expect_fence_events(std::vector<FencePayload>& expected, Lambda lam) {
  found_payloads = {};
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      [](const char* name, const uint32_t dev_id, uint64_t* kID) {
        found_payloads.push_back(
            FencePayload{std::string(name),
                         FencePayload::distinguishable_devices::no, dev_id});
      });
  lam();
  for (auto& entry : expected) {
    auto search = std::find_if(
        found_payloads.begin(), found_payloads.end(),
        [&](const auto& found_entry) {
          // std::cout << entry.dev_id << ": "<<found_entry.dev_id << std::endl;
          // std::cout << entry.name << ": "<<found_entry.name << std::endl;
          return ((found_entry.name.find(entry.name) != std::string::npos) &&
                  (entry.dev_id == found_entry.dev_id));
        });
    auto found = (search != found_payloads.end());
    ASSERT_TRUE(found);
  }
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      [](const char* name, const uint32_t dev_id, uint64_t* kID) {});
}

template <class>
struct increment;
template <>
struct increment<Kokkos::Serial> {
  constexpr static const int size = 0;
};
template <>
struct increment<Kokkos::OpenMP> {
  constexpr static const int size = 0;
};
template <>
struct increment<Kokkos::Threads> {
  constexpr static const int size = 0;
};
template <>
struct increment<Kokkos::Cuda> {
  constexpr static const int size = 1;
};
template <>
struct increment<Kokkos::Experimental::HIP> {
  constexpr static const int size = 1;
};
template <>
struct increment<Kokkos::Experimental::OpenMPTarget> {
  constexpr static const int size = 1;
};
template <>
struct increment<Kokkos::Experimental::SYCL> {
  constexpr static const int size = 1;
};

template <>
struct increment<Kokkos::Experimental::HPX> {
  constexpr static const int size = 0;
};
int num_instances = 1;

TEST(defaultdevicetype, test_named_instance_fence) {
  auto root = Kokkos::Tools::Experimental::device_id_root<
      Kokkos::DefaultExecutionSpace>();
  std::vector<FencePayload> expected{

      {"named_instance", FencePayload::distinguishable_devices::no,
       root + num_instances}};
  expect_fence_events(expected, [=]() {
    Kokkos::DefaultExecutionSpace ex;
    ex.fence("named_instance");
  });
  num_instances += increment<Kokkos::DefaultExecutionSpace>::size;
}
TEST(defaultdevicetype, test_unnamed_instance_fence) {
  auto root = Kokkos::Tools::Experimental::device_id_root<
      Kokkos::DefaultExecutionSpace>();
  std::vector<FencePayload> expected{

      {"Unnamed Instance Fence", FencePayload::distinguishable_devices::no,
       root + num_instances}};
  expect_fence_events(expected, [=]() {
    Kokkos::DefaultExecutionSpace ex;
    ex.fence();
  });
  num_instances += increment<Kokkos::DefaultExecutionSpace>::size;
}

TEST(defaultdevicetype, test_named_global_fence) {
  auto root = Kokkos::Tools::Experimental::device_id_root<
      Kokkos::DefaultExecutionSpace>();

  std::vector<FencePayload> expected{

      {"test global fence", FencePayload::distinguishable_devices::no, root}};
  expect_fence_events(expected, [=]() { Kokkos::fence("test global fence"); });
}

TEST(defaultdevicetype, test_unnamed_global_fence) {
  auto root = Kokkos::Tools::Experimental::device_id_root<
      Kokkos::DefaultExecutionSpace>();

  std::vector<FencePayload> expected{

      {"Unnamed Global Fence", FencePayload::distinguishable_devices::no,
       root}};
  expect_fence_events(expected, [=]() { Kokkos::fence(); });
  num_instances += increment<Kokkos::DefaultExecutionSpace>::size;
}
}  // namespace Test
