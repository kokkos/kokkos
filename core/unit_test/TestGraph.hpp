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
#include <Kokkos_Graph.hpp>

#include <gtest/gtest.h>

namespace Test {

template <class ExecSpace>
struct CountTestFunctor {
  using value_type = int;
  template <class T>
  using atomic_view =
      Kokkos::View<T, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  atomic_view<int> count;
  atomic_view<int> bugs;
  int expected_count_min;
  int expected_count_max;

  template <class... Ts>
  KOKKOS_FUNCTION void operator()(Ts&&...) const noexcept {
    bugs() += int(count() > expected_count_max || count() < expected_count_min);
    count()++;
  }
};

template <class ExecSpace, class T>
struct SetViewToValueFunctor {
  using value_type = T;
  using view_type =
      Kokkos::View<T, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  view_type v;
  T value;

  template <class... Ts>
  KOKKOS_FUNCTION void operator()(Ts&&...) const noexcept {
    v() = value;
  }
};

template <class ExecSpace, class T>
struct SetResultToViewFunctor {
  using value_type = T;
  using view_type =
      Kokkos::View<T, ExecSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>;
  view_type v;

  template <class U>
  KOKKOS_FUNCTION void operator()(U&&, value_type& val) const noexcept {
    val += v();
  }
};

struct TEST_CATEGORY_FIXTURE(count_bugs) : public ::testing::Test {
 public:
  using count_functor      = CountTestFunctor<TEST_EXECSPACE>;
  using set_functor        = SetViewToValueFunctor<TEST_EXECSPACE, int>;
  using set_result_functor = SetResultToViewFunctor<TEST_EXECSPACE, int>;
  using view_type          = typename count_functor::template atomic_view<int>;
  using view_host          = Kokkos::View<int, Kokkos::HostSpace>;
  view_type count{"count"};
  view_type bugs{"bugs"};
  view_host count_host{"count_host"};
  view_host bugs_host{"bugs_host"};
  TEST_EXECSPACE ex{};

 protected:
  void SetUp() override {
    Kokkos::deep_copy(ex, count, 0);
    Kokkos::deep_copy(ex, bugs, 0);
    ex.fence();
  }
};

TEST_F(TEST_CATEGORY_FIXTURE(count_bugs), launch_one) {
  auto graph = Kokkos::Experimental::create_graph([=](auto builder) {
    builder.parallel_for(1, count_functor{count, bugs, 0, 0});
  });
  graph.submit();
  Kokkos::deep_copy(ex, count_host, count);
  Kokkos::deep_copy(ex, bugs_host, bugs);
  ex.fence();
  ASSERT_EQ(1, count_host());
  ASSERT_EQ(0, bugs_host());
}

TEST_F(TEST_CATEGORY_FIXTURE(count_bugs), launch_one_rvalue) {
  Kokkos::Experimental::create_graph(ex, [=](auto builder) {
    builder.get_root().then_parallel_for(1, count_functor{count, bugs, 0, 0});
  }).submit();
  Kokkos::deep_copy(ex, count_host, count);
  Kokkos::deep_copy(ex, bugs_host, bugs);
  ex.fence();
  ASSERT_EQ(1, count_host());
  ASSERT_EQ(0, bugs_host());
}

TEST_F(TEST_CATEGORY_FIXTURE(count_bugs), launch_six) {
  auto graph = Kokkos::Experimental::create_graph(ex, [=](auto builder) {
    auto f_setup_count = builder.parallel_for(1, set_functor{count, 0});
    auto f_setup_bugs  = builder.parallel_for(1, set_functor{bugs, 0});

    //----------------------------------------
    auto ready = builder.when_all(f_setup_count, f_setup_bugs);

    //----------------------------------------
    ready.then_parallel_for(1, count_functor{count, bugs, 0, 6});
    //----------------------------------------
    ready.then_parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>{0, 1},
                            count_functor{count, bugs, 0, 6});
    //----------------------------------------
    ready.then_parallel_for(
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{0, 0}, {1, 1}},
        count_functor{count, bugs, 0, 6});
    //----------------------------------------
    ready.then_parallel_for(Kokkos::TeamPolicy<TEST_EXECSPACE>{1, 1},
                            count_functor{count, bugs, 0, 6});
    //----------------------------------------
    ready.then_parallel_for(2, count_functor{count, bugs, 0, 6});
    //----------------------------------------
  });
  graph.submit();
  Kokkos::deep_copy(ex, count_host, count);
  Kokkos::deep_copy(ex, bugs_host, bugs);
  ex.fence();

  ASSERT_EQ(6, count_host());
  ASSERT_EQ(0, bugs_host());
}

TEST_F(TEST_CATEGORY_FIXTURE(count_bugs), repeat_chain) {
  auto graph = Kokkos::Experimental::create_graph(ex, [=](auto builder) {
    //----------------------------------------
    builder.parallel_for(1, set_functor{count, 0})
        .then_parallel_for(1, count_functor{count, bugs, 0, 0})
        .then_parallel_for(1, count_functor{count, bugs, 1, 1})
        .then_parallel_reduce(1, set_result_functor{count}, count_host)
        .then_parallel_reduce(1, set_result_functor{bugs},
                              Kokkos::Sum<int, TEST_EXECSPACE>{bugs_host});
    //----------------------------------------
  });

  //----------------------------------------
  constexpr int repeats = 10;

  for (int i = 0; i < repeats; ++i) {
    graph.submit();
    ex.fence();
    ASSERT_EQ(2, count_host());
    ASSERT_EQ(0, bugs_host());
  }
  //----------------------------------------
}

TEST_F(TEST_CATEGORY_FIXTURE(count_bugs), when_all_cycle) {
  Kokkos::Experimental::create_graph(ex, [=](auto builder) {
    //----------------------------------------
    // Test when_all when redundant dependencies are given
    auto f1 = builder.parallel_for(1, set_functor{count, 0});
    auto f2 = f1.then_parallel_for(1, count_functor{count, bugs, 0, 0});
    auto f3 = f2.then_parallel_for(5, count_functor{count, bugs, 1, 5});
    auto f4 = builder.when_all(f2, f3).then_parallel_for(
        1, count_functor{count, bugs, 6, 6});
    builder.when_all(f1, f4, f3)
        .then_parallel_reduce(5, set_result_functor{bugs}, bugs_host)
        .then_parallel_reduce(1, set_result_functor{count}, count_host);
    //----------------------------------------
  }).submit();
  ASSERT_EQ(0, bugs_host());
  ASSERT_EQ(7, count_host());
  //----------------------------------------
}

}  // end namespace Test
