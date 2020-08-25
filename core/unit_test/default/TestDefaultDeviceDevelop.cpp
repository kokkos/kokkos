
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

#include <Kokkos_Graph.hpp>

namespace Test {

TEST(defaultdevicetype, development_test) {
  constexpr int repeats = 5;
  Kokkos::View<int, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Atomic>> count{
      "graph_kernel_count"};
  Kokkos::View<int*, TEST_EXECSPACE> graph_reduction_output{
      "graph_reduction_output", 2};
  Kokkos::View<int, TEST_EXECSPACE> graph_reduction_test_out{
      "graph_reduction_test_out"};
  Kokkos::View<int, TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Atomic>> bugs{
      "graph_kernel_bugs"};
  auto graph = Kokkos::Experimental::create_graph([=](auto builder) {
    auto root = builder.get_root();

    auto f1 = root.then_parallel_for(
        Kokkos::TeamPolicy<TEST_EXECSPACE>(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(auto const& member) {
          Kokkos::single(Kokkos::PerTeam(member), [&] {
            bugs() += int(count() != 0);
            count()++;
          });
        });
    auto f2 = f1.then_parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, TEST_EXECSPACE>({0, 0}, {1, 1}),
        KOKKOS_LAMBDA(auto, auto) {
          bugs() += int(count() < 1 || count() > 2);
          count()++;
        });
    char useless_huge_thing_to_trigger_global_memory_kernel[1 << 16] = {};
    useless_huge_thing_to_trigger_global_memory_kernel[42]           = 1;
    Kokkos::Experimental::GraphNodeRef<TEST_EXECSPACE> f3 =
        f1.then_parallel_for(
            Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(long) {
              bugs() += int(count() < 1 || count() > 2);
              count() +=
                  int{useless_huge_thing_to_trigger_global_memory_kernel[42]};
            });
    // Intended to use Constant Memory with the traditional launch mechanism
    auto f5 = builder.when_all(f2, f3).then_parallel_for(
        Kokkos::Experimental::require(
            Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
            Kokkos::Experimental::WorkItemProperty::HintHeavyWeight),
        KOKKOS_LAMBDA(long) {
          bugs() += int(count() != 3);
          count()++;
        });
    // int val = 0;
    f5.then_parallel_reduce(
          "", Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
          KOKKOS_LAMBDA(long, int& out) {
            bugs() += int(count() != 4);
            count()++;
            out += 1;
          },
          Kokkos::subview(graph_reduction_output, 0))
        .then_parallel_reduce(
            Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
            KOKKOS_LAMBDA(long, int& out) {
              bugs() += int(count() != 5);
              count()++;
              out += 1;
            },
            Kokkos::Sum<int, TEST_EXECSPACE>{
                Kokkos::subview(graph_reduction_output, 1)})
        .then_parallel_reduce(
            Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
            KOKKOS_LAMBDA(long, int& out) {
              bugs() += int(count() != 6);
              count()++;
              out += 1;
            },
            graph_reduction_test_out)
        // Fails via static_assert():
        //   .then_parallel_reduce(
        //     Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
        //     KOKKOS_LAMBDA(long, int& out) {
        //       bugs() += int(count() != 6);
        //       count()++;
        //       out += 1; },
        //     val)
        //
        // Fails at runtime with an exception:
        //   .then_parallel_reduce(
        //       Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1),
        //       KOKKOS_LAMBDA(long, int& out) {
        //         bugs() += int(count() != 5);
        //         count()++;
        //         out += 1;
        //       },
        //       Kokkos::Sum<int, TEST_EXECSPACE>{val})
        ;
    Kokkos::Experimental::GraphNodeRef<TEST_EXECSPACE> ftest = f5;
    Kokkos::Experimental::GraphNodeRef<TEST_EXECSPACE> ftest2;
    ftest2 = f5;
    static_assert(
        std::is_convertible<decltype(f1), Kokkos::Experimental::GraphNodeRef<
                                              TEST_EXECSPACE>>::value,
        "Type erasure didn't work");
    static_assert(
        !std::is_convertible<Kokkos::Experimental::GraphNodeRef<
            TEST_EXECSPACE>, decltype(f1)>::value,
        "Type erasure didn't work");
  });

  for (int i = 0; i < repeats; ++i) {
    Kokkos::deep_copy(graph.get_execution_space(), graph_reduction_output, 0);
    Kokkos::deep_copy(graph.get_execution_space(), count, 0);
    Kokkos::deep_copy(graph.get_execution_space(), bugs, 0);
    graph.submit();
    graph.get_execution_space().fence();
    auto count_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);
    auto bugs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bugs);
    auto graph_reduction_output_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, graph_reduction_output);

    EXPECT_EQ(7, count_host());
    EXPECT_EQ(0, bugs_host());
    EXPECT_EQ(1, graph_reduction_output_host(0));
    EXPECT_EQ(1, graph_reduction_output_host(1));
  }

  Kokkos::deep_copy(graph.get_execution_space(), graph_reduction_output, 0);
  Kokkos::deep_copy(graph.get_execution_space(), count, 0);
  Kokkos::deep_copy(graph.get_execution_space(), bugs, 0);
  auto ex = graph.get_execution_space();
  std::move(graph).submit();
  ex.fence();
  auto count_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, count);
  auto bugs_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bugs);
  auto graph_reduction_output_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, graph_reduction_output);

  EXPECT_EQ(7, count_host());
  EXPECT_EQ(0, bugs_host());
  EXPECT_EQ(1, graph_reduction_output_host(0));
  EXPECT_EQ(1, graph_reduction_output_host(1));
}

}  // namespace Test
