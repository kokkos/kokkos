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

#include <gtest/gtest.h>

//<editor-fold desc="my_get(my_array) workaround">
// Because using std::array on the device with CUDA would require compiling with
// the '--expt-relaxed-constexpr' flag
// clang-format off
template <class T, std::size_t N>
struct my_array {
  T data[N];
  KOKKOS_FUNCTION constexpr T const& operator[](std::size_t i) const { return data[i]; }
};

template <std::size_t I, class T, std::size_t N>
KOKKOS_FUNCTION constexpr T const& my_get(my_array<T, N> const& a) { return a[I]; }
// clang-format on
//</editor-fold>

template <class View, class ExecutionSpace>
struct TestViewMemoryAccessViolation {
  View v;
  static constexpr auto rank = View::rank;

  template <std::size_t... Is>
  KOKKOS_FUNCTION decltype(auto) bad_access(std::index_sequence<Is...>) const {
    return v(my_get<Is>(my_array<int, rank>{})...);
  }

  KOKKOS_FUNCTION void operator()(int) const {
    ++bad_access(std::make_index_sequence<rank>{});
  }

  TestViewMemoryAccessViolation(View w, ExecutionSpace const& s,
                                std::string const& matcher)
      : v(std::move(w)) {
    constexpr bool view_accessible_from_execution_space =
        Kokkos::SpaceAccessibility<
            /*AccessSpace=*/ExecutionSpace,
            /*MemorySpace=*/typename View::memory_space>::accessible;
    EXPECT_FALSE(view_accessible_from_execution_space);
    EXPECT_DEATH(
        {
          Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(s, 0, 1),
                               *this);
          Kokkos::fence();
        },
        matcher);
  }
};

template <class View, class ExecutionSpace>
void test_view_memory_access_violation(View v, ExecutionSpace const& s,
                                       std::string const& m) {
  TestViewMemoryAccessViolation<View, ExecutionSpace>(std::move(v), s, m);
}

template <class ExecutionSpace>
void test_view_memory_access_violations_from_host() {
  Kokkos::DefaultHostExecutionSpace const host_exec_space{};
  // clang-format off
  test_view_memory_access_violation(Kokkos::View<int,         ExecutionSpace>("my_label_0"),                 host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_0");
  test_view_memory_access_violation(Kokkos::View<int*,        ExecutionSpace>("my_label_1",1),               host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_1");
  test_view_memory_access_violation(Kokkos::View<int**,       ExecutionSpace>("my_label_2",1,2),             host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_2");
  test_view_memory_access_violation(Kokkos::View<int***,      ExecutionSpace>("my_label_3",1,2,3),           host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_3");
  test_view_memory_access_violation(Kokkos::View<int****,     ExecutionSpace>("my_label_4",1,2,3,4),         host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_4");
  test_view_memory_access_violation(Kokkos::View<int*****,    ExecutionSpace>("my_label_5",1,2,3,4,5),       host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_5");
  test_view_memory_access_violation(Kokkos::View<int******,   ExecutionSpace>("my_label_6",1,2,3,4,5,6),     host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_6");
  test_view_memory_access_violation(Kokkos::View<int*******,  ExecutionSpace>("my_label_7",1,2,3,4,5,6,7),   host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_7");
  test_view_memory_access_violation(Kokkos::View<int********, ExecutionSpace>("my_label_8",1,2,3,4,5,6,7,8), host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*my_label_8");
  auto* const ptr = reinterpret_cast<int*>(0xABADBABE);
  test_view_memory_access_violation(Kokkos::View<int,         ExecutionSpace>(ptr),                 host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int*,        ExecutionSpace>(ptr,1),               host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int**,       ExecutionSpace>(ptr,1,2),             host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int***,      ExecutionSpace>(ptr,1,2,3),           host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int****,     ExecutionSpace>(ptr,1,2,3,4),         host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int*****,    ExecutionSpace>(ptr,1,2,3,4,5),       host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int******,   ExecutionSpace>(ptr,1,2,3,4,5,6),     host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int*******,  ExecutionSpace>(ptr,1,2,3,4,5,6,7),   host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  test_view_memory_access_violation(Kokkos::View<int********, ExecutionSpace>(ptr,1,2,3,4,5,6,7,8), host_exec_space, "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNMANAGED");
  // clang-format on
}

template <class ExecutionSpace>
void test_view_memory_access_violations_from_device() {
  ExecutionSpace const exec_space{};
  // clang-format off
  std::string const matcher = "Kokkos::View ERROR: attempt to access inaccessible memory space.*UNAVAILABLE";
  test_view_memory_access_violation(Kokkos::View<int,         Kokkos::HostSpace>("my_label_0"),                 exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*,        Kokkos::HostSpace>("my_label_1",1),               exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int**,       Kokkos::HostSpace>("my_label_2",1,2),             exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int***,      Kokkos::HostSpace>("my_label_3",1,2,3),           exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int****,     Kokkos::HostSpace>("my_label_4",1,2,3,4),         exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*****,    Kokkos::HostSpace>("my_label_5",1,2,3,4,5),       exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int******,   Kokkos::HostSpace>("my_label_6",1,2,3,4,5,6),     exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*******,  Kokkos::HostSpace>("my_label_7",1,2,3,4,5,6,7),   exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int********, Kokkos::HostSpace>("my_label_8",1,2,3,4,5,6,7,8), exec_space, matcher);

  auto* const ptr = reinterpret_cast<int*>(0xABADBABE);
  test_view_memory_access_violation(Kokkos::View<int,         Kokkos::HostSpace>(ptr),                 exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*,        Kokkos::HostSpace>(ptr,1),               exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int**,       Kokkos::HostSpace>(ptr,1,2),             exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int***,      Kokkos::HostSpace>(ptr,1,2,3),           exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int****,     Kokkos::HostSpace>(ptr,1,2,3,4),         exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*****,    Kokkos::HostSpace>(ptr,1,2,3,4,5),       exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int******,   Kokkos::HostSpace>(ptr,1,2,3,4,5,6),     exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int*******,  Kokkos::HostSpace>(ptr,1,2,3,4,5,6,7),   exec_space, matcher);
  test_view_memory_access_violation(Kokkos::View<int********, Kokkos::HostSpace>(ptr,1,2,3,4,5,6,7,8), exec_space, matcher);
  // clang-format on
}

TEST(TEST_CATEGORY_DEATH, view_memory_access_violations_from_host) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using ExecutionSpace = TEST_EXECSPACE;

  if (Kokkos::SpaceAccessibility<
          /*AccessSpace=*/Kokkos::HostSpace,
          /*MemorySpace=*/typename ExecutionSpace::memory_space>::accessible) {
    GTEST_SKIP() << "skipping since no memory access violation would occur";
  }
  test_view_memory_access_violations_from_host<ExecutionSpace>();
}

TEST(TEST_CATEGORY_DEATH, view_memory_access_violations_from_device) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";

  using ExecutionSpace = TEST_EXECSPACE;

  if (Kokkos::SpaceAccessibility<
          /*AccessSpace=*/ExecutionSpace,
          /*MemorySpace=*/Kokkos::HostSpace>::accessible) {
    GTEST_SKIP() << "skipping since no memory access violation would occur";
  }
  test_view_memory_access_violations_from_device<ExecutionSpace>();
}
