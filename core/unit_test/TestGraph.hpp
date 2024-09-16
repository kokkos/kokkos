//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>

#include <gtest/gtest.h>

namespace Test {

template <class ExecSpace, class ValueType>
struct NoOpReduceFunctor {
  KOKKOS_FUNCTION void operator()(int, ValueType&) const {
    Kokkos::abort("Should never be called!");
  }
  KOKKOS_FUNCTION void operator()(int, int, ValueType&) const {
    Kokkos::abort("Should never be called!");
  }
  KOKKOS_FUNCTION void operator()(
      const typename Kokkos::TeamPolicy<ExecSpace>::member_type&,
      ValueType&) const {
    Kokkos::abort("Should never be called!");
  }
};

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

struct TEST_CATEGORY_FIXTURE(graph) : public ::testing::Test {
 public:
  using count_functor      = CountTestFunctor<TEST_EXECSPACE>;
  using set_functor        = SetViewToValueFunctor<TEST_EXECSPACE, int>;
  using set_result_functor = SetResultToViewFunctor<TEST_EXECSPACE, int>;
  using view_type          = Kokkos::View<int, TEST_EXECSPACE>;
  using atomic_view_type   = typename count_functor::template atomic_view<int>;
  using view_host          = Kokkos::View<int, Kokkos::HostSpace>;
  atomic_view_type count{"count"};
  atomic_view_type bugs{"bugs"};
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

// Check if a rank-0 view contains a given value.
template <typename Exec, typename ViewType>
::testing::AssertionResult contains(
    const Exec& exec, const ViewType& view,
    const typename ViewType::value_type& expected) {
  static_assert(ViewType::rank() == 0);
  typename ViewType::non_const_value_type value;
  Kokkos::deep_copy(exec, value, view);
  exec.fence();
  if (value != expected)
    return ::testing::AssertionFailure()
           << expected << " is not in " << view.label() << ", got " << value;
  else
    return ::testing::AssertionSuccess();
}

TEST_F(TEST_CATEGORY_FIXTURE(graph), submit_once) {
  auto graph =
      Kokkos::Experimental::create_graph<TEST_EXECSPACE>([&](auto root) {
        root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
      });
  graph.submit();

  ASSERT_TRUE(contains(graph.get_execution_space(), count, 1));
  ASSERT_TRUE(contains(graph.get_execution_space(), bugs, 0));
}

TEST_F(TEST_CATEGORY_FIXTURE(graph), submit_once_rvalue) {
  Kokkos::Experimental::create_graph(ex, [&](auto root) {
    root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
  }).submit();

  ASSERT_TRUE(contains(ex, count, 1));
  ASSERT_TRUE(contains(ex, bugs, 0));
}

// Ensure that Kokkos::Graph::instantiate works.
// For now, Kokkos::Graph::submit will instantiate if needed,
// so this test is not very strong.
TEST_F(TEST_CATEGORY_FIXTURE(graph), instantiate_and_submit_once) {
  auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
    root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
  });
  graph.instantiate();
  graph.submit();

  ASSERT_TRUE(contains(ex, count, 1));
  ASSERT_TRUE(contains(ex, bugs, 0));
}

// FIXME death tests and fixtures
#define TEST_CATEGORY_FIXTURE_DEATH_HELPER(category, name) \
  category##_##name##_DeathTest
#define TEST_CATEGORY_FIXTURE_DEATH_HELPER_EXPAND(category, name) \
  TEST_CATEGORY_FIXTURE_DEATH_HELPER(category, name)
#define TEST_CATEGORY_FIXTURE_DEATH(name) \
  TEST_CATEGORY_FIXTURE_DEATH_HELPER_EXPAND(TEST_CATEGORY, name)

struct TEST_CATEGORY_FIXTURE_DEATH(graph)
    : public TEST_CATEGORY_FIXTURE(graph) {};

// Ensure that Kokkos::Graph::instantiate can be called only once.
// This test checks 2 cases:
//   1. Instantiating after submission is invalid (this also implicitly
//      checks that submission instantiates if need be).
//   2. Instantiating twice in a row is invalid.
TEST_F(TEST_CATEGORY_FIXTURE_DEATH(graph), can_instantiate_only_once) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  {
    bool checked_assertions = false;
    KOKKOS_ASSERT(checked_assertions = true);
    if (!checked_assertions) {
      GTEST_SKIP() << "Preconditions are not checked.";
    }
  }
  {
    auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
      root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
    });
    graph.submit();
    ASSERT_DEATH(graph.instantiate(),
                 "Expected precondition `.*` evaluated false.");
  }
  {
    auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
      root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
    });
    graph.instantiate();
    ASSERT_DEATH(graph.instantiate(),
                 "Expected precondition `.*` evaluated false.");
  }
}

// This test submits on an execution space instance different from the
// one passed to the Kokkos::Graph constructor.
TEST_F(TEST_CATEGORY_FIXTURE(graph),
       submit_onto_another_execution_space_instance) {
  const auto execution_space_instances =
      Kokkos::Experimental::partition_space(ex, 1, 1);

  auto graph = Kokkos::Experimental::create_graph(
      execution_space_instances.at(0), [&](auto root) {
        root.then_parallel_for(1, count_functor{count, bugs, 0, 0});
      });
  graph.instantiate();

  execution_space_instances.at(0).fence(
      "The graph might make async copies to device.");

  graph.submit(execution_space_instances.at(1));

  ASSERT_TRUE(contains(execution_space_instances.at(1), count, 1));
  ASSERT_TRUE(contains(execution_space_instances.at(1), bugs, 0));
}

// This test ensures that it's possible to build a Kokkos::Graph using
// Kokkos::Experimental::create_graph without providing a closure, but giving an
// execution space instance.
TEST_F(TEST_CATEGORY_FIXTURE(graph), create_graph_no_closure_with_exec) {
  auto graph = Kokkos::Experimental::create_graph(ex);

  auto root = Kokkos::Impl::GraphAccess::create_root_ref(graph);

  auto node = root.then_parallel_for(1, count_functor{count, bugs, 0, 0});

  graph.submit(ex);

  ASSERT_TRUE(contains(ex, count, 1));
  ASSERT_TRUE(contains(ex, bugs, 0));
}

// This test ensures that it's possible to build a Kokkos::Graph using
// Kokkos::Experimental::create_graph without any argument.
// The test has to be skipped if the test fixture is
// not instantiated for the default execution space.
TEST_F(TEST_CATEGORY_FIXTURE(graph), create_graph_no_arg) {
  if constexpr (!std::is_same_v<TEST_EXECSPACE,
                                Kokkos::DefaultExecutionSpace>) {
    GTEST_SKIP() << "Skipping since useless if the test fixture is not on the "
                    "default execution space.";
  }

  auto graph = Kokkos::Experimental::create_graph();

  static_assert(std::is_same_v<typename decltype(graph)::execution_space,
                               Kokkos::DefaultExecutionSpace>);

  auto root = Kokkos::Impl::GraphAccess::create_root_ref(graph);

  auto node = root.then_parallel_for(1, count_functor{count, bugs, 0, 0});

  graph.submit(graph.get_execution_space());

  ASSERT_TRUE(contains(graph.get_execution_space(), count, 1));
  ASSERT_TRUE(contains(graph.get_execution_space(), bugs, 0));
}

TEST_F(TEST_CATEGORY_FIXTURE(graph), submit_six) {
#ifdef KOKKOS_ENABLE_OPENMPTARGET  // FIXME_OPENMPTARGET team_size incompatible
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenMPTarget>)
    GTEST_SKIP() << "skipping since OpenMPTarget can't use team_size 1";
#endif
#if defined(KOKKOS_ENABLE_SYCL) && \
    !defined(SYCL_EXT_ONEAPI_GRAPH)  // FIXME_SYCL
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::SYCL>)
    GTEST_SKIP() << "skipping since test case is known to fail with SYCL";
#endif

  auto graph = Kokkos::Experimental::create_graph(ex, [&](auto root) {
    auto f_setup_count = root.then_parallel_for(1, set_functor{count, 0});
    auto f_setup_bugs  = root.then_parallel_for(1, set_functor{bugs, 0});

    //----------------------------------------
    auto ready = Kokkos::Experimental::when_all(f_setup_count, f_setup_bugs);

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

  ASSERT_TRUE(contains(ex, count, 6));
  ASSERT_TRUE(contains(ex, bugs, 0));
}

TEST_F(TEST_CATEGORY_FIXTURE(graph), when_all_cycle) {
  view_type reduction_out{"reduction_out"};
  view_host reduction_host{"reduction_host"};
  Kokkos::Experimental::create_graph(ex, [&](auto root) {
    //----------------------------------------
    // Test when_all when redundant dependencies are given
    auto f1 = root.then_parallel_for(1, set_functor{count, 0});
    auto f2 = f1.then_parallel_for(1, count_functor{count, bugs, 0, 0});
    auto f3 = f2.then_parallel_for(5, count_functor{count, bugs, 1, 5});
    auto f4 = Kokkos::Experimental::when_all(f2, f3).then_parallel_for(
        1, count_functor{count, bugs, 6, 6});
    Kokkos::Experimental::when_all(f1, f4, f3)
        .then_parallel_reduce(6, set_result_functor{count}, reduction_out);
    //----------------------------------------
  }).submit();

  ASSERT_TRUE(contains(ex, bugs, 0));
  ASSERT_TRUE(contains(ex, count, 7));
  ASSERT_TRUE(contains(ex, reduction_out, 42));
  //----------------------------------------
}

// This test is disabled because we don't currently support copying to host,
// even asynchronously. We _may_ want to do that eventually?
TEST_F(TEST_CATEGORY_FIXTURE(graph), DISABLED_repeat_chain) {
  auto graph = Kokkos::Experimental::create_graph(
      ex, [&, count_host = count_host](auto root) {
        //----------------------------------------
        root.then_parallel_for(1, set_functor{count, 0})
            .then_parallel_for(1, count_functor{count, bugs, 0, 0})
            .then_parallel_for(1, count_functor{count, bugs, 1, 1})
            .then_parallel_reduce(1, set_result_functor{count}, count_host)
            .then_parallel_reduce(
                1, set_result_functor{bugs},
                Kokkos::Sum<int, Kokkos::HostSpace>{bugs_host});
        //----------------------------------------
      });

  //----------------------------------------
  constexpr int repeats = 10;

  for (int i = 0; i < repeats; ++i) {
    graph.submit();
    ex.fence();
    EXPECT_EQ(2, count_host());
    EXPECT_EQ(0, bugs_host());
  }
  //----------------------------------------
}

TEST_F(TEST_CATEGORY_FIXTURE(graph), zero_work_reduce) {
  auto graph = Kokkos::Experimental::create_graph(
      ex, [&](Kokkos::Experimental::GraphNodeRef<TEST_EXECSPACE> root) {
        NoOpReduceFunctor<TEST_EXECSPACE, int> no_op_functor;
        root.then_parallel_reduce(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 0),
                                  no_op_functor, count)
#if !defined(KOKKOS_ENABLE_SYCL) || \
    defined(SYCL_EXT_ONEAPI_GRAPH)  // FIXME_SYCL
#if !defined(KOKKOS_ENABLE_CUDA) && \
    !defined(KOKKOS_ENABLE_HIP)  // FIXME_CUDA FIXME_HIP
            .then_parallel_reduce(
                Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<2>>{{0, 0},
                                                                       {0, 0}},
                no_op_functor, count)
#endif
            .then_parallel_reduce(
                Kokkos::TeamPolicy<TEST_EXECSPACE>{0, Kokkos::AUTO},
                no_op_functor, count)
#endif
            ;
      });
// These fences are only necessary because of the weirdness of how CUDA
// UVM works on pre pascal cards.
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_UVM) && \
    (defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL))
  Kokkos::fence();
#endif
  graph.submit();
  Kokkos::deep_copy(ex, count, 1);
// These fences are only necessary because of the weirdness of how CUDA
// UVM works on pre pascal cards.
#if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_CUDA_UVM) && \
    (defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL))
  if constexpr (std::is_same_v<TEST_EXECSPACE, Kokkos::Cuda>) Kokkos::fence();
#endif
  graph.submit();

  ASSERT_TRUE(contains(ex, count, 0));
}

// Ensure that an empty graph can be submitted.
TEST_F(TEST_CATEGORY_FIXTURE(graph), empty_graph) {
  auto graph = Kokkos::Experimental::create_graph(ex, [](auto) {});
  graph.instantiate();
  graph.submit(ex);
  ex.fence();
}

template <typename ViewType, size_t TargetIndex, size_t NumIndices = 0>
struct FetchValuesAndContribute {
  static_assert(std::is_same_v<typename ViewType::value_type,
                               typename ViewType::non_const_value_type>);

  ViewType data;
  typename ViewType::value_type value;
  Kokkos::Array<size_t, NumIndices> indices{};

  FetchValuesAndContribute(ViewType data_,
                           std::integral_constant<size_t, TargetIndex>,
                           typename ViewType::value_type value_)
      : data(std::move(data_)), value(value_) {}

  FetchValuesAndContribute(ViewType data_,
                           Kokkos::Array<size_t, NumIndices> indices_,
                           std::integral_constant<size_t, TargetIndex>,
                           typename ViewType::value_type value_)
      : data(std::move(data_)), value(value_), indices(std::move(indices_)) {}

  template <typename T>
  KOKKOS_FUNCTION void operator()(const T) const {
    for (const auto index : indices) data(TargetIndex) += data(index);
    data(TargetIndex) += value;
  }
};

template <typename ViewType, size_t TargetIndex, size_t NumIndices>
FetchValuesAndContribute(ViewType, const size_t (&)[NumIndices],
                         std::integral_constant<size_t, TargetIndex>,
                         typename ViewType::non_const_value_type)
    -> FetchValuesAndContribute<ViewType, TargetIndex, NumIndices>;

// Ensure that we can handle the simple diamond use case.
//
// topology     stream-based approach       graph-based
//
//   A          A(exec_0)                   Using the API to add nodes, no
//  / \         fence(exec_0)               user-facing fence anymore because
// B   C        B(exec_0)   C(exec_1)       we'd like to rely on the graph to
//  \ /         fence(exec_1)               enforce dependencies.
//   D          D(exec_0)
TEST_F(TEST_CATEGORY_FIXTURE(graph), diamond) {
  const auto execution_space_instances =
      Kokkos::Experimental::partition_space(ex, 1, 1, 1, 1);

  const auto exec_0 = execution_space_instances.at(0);
  const auto exec_1 = execution_space_instances.at(1);
  const auto exec_2 = execution_space_instances.at(2);
  const auto exec_3 = execution_space_instances.at(3);

  using policy_t = Kokkos::RangePolicy<TEST_EXECSPACE>;
  using view_t   = Kokkos::View<int*, TEST_EXECSPACE>;
  using view_h_t = Kokkos::View<int*, Kokkos::HostSpace>;

  view_t data(Kokkos::view_alloc(ex, "diamond - data"), 4);

  constexpr int value_A = 42, value_B = 27, value_C = 13, value_D = 147;
  std::integral_constant<size_t, 0> index_A;
  std::integral_constant<size_t, 1> index_B;
  std::integral_constant<size_t, 2> index_C;
  std::integral_constant<size_t, 3> index_D;

  auto graph = Kokkos::Experimental::create_graph(exec_2, [&](auto root) {
    auto node_A = root.then_parallel_for(
        policy_t(exec_0, 0, 1),
        FetchValuesAndContribute(data, index_A, value_A));

    auto node_B = node_A.then_parallel_for(
        policy_t(exec_0, 0, 1),
        FetchValuesAndContribute(data, {index_A()}, index_B, value_B));
    auto node_C = node_A.then_parallel_for(
        policy_t(exec_1, 0, 1),
        FetchValuesAndContribute(data, {index_A()}, index_C, value_C));

    auto node_D = Kokkos::Experimental::when_all(node_B, node_C)
                      .then_parallel_for(
                          policy_t(exec_0, 0, 1),
                          FetchValuesAndContribute(data, {index_B(), index_C()},
                                                   index_D, value_D));
  });
  graph.instantiate();

  // TODO Check that kernels are running on the execution space instance of
  //      their policy if the defaulted graph implementation is used.
  graph.submit(exec_3);

  view_h_t data_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "diamond - data - host"),
      4);
  Kokkos::deep_copy(exec_3, data_host, data);

  exec_3.fence();

  ASSERT_EQ(data_host(index_A()), value_A);
  ASSERT_EQ(data_host(index_B()), value_A + value_B);
  ASSERT_EQ(data_host(index_C()), value_A + value_C);
  ASSERT_EQ(data_host(index_D()), 2 * value_A + value_B + value_C + value_D);
}

// Test a configuration that has more than one end node. Ensure that we wait for
// them all by adding a manual kernel after the graph.
// This test mainly is there to ensure that the defaulted graph implementation
// enforces a semantically consistent control flow.
//
// topology         stream-based approach
//
//    A       B     A(exec_0)   B(exec_1)
//      \   / |     fence(exec_1)
//        C   |     C(exec_0)
//      /     E                 E(exec_1)
//    D             D(exec_0)
//                  fence(exec_1)
//    F             F(exec_0)
TEST_F(TEST_CATEGORY_FIXTURE(graph), end_of_submit_control_flow) {
  const auto execution_space_instances =
      Kokkos::Experimental::partition_space(ex, 1, 1, 1, 1);

  const auto exec_0 = execution_space_instances.at(0);
  const auto exec_1 = execution_space_instances.at(1);
  const auto exec_2 = execution_space_instances.at(2);
  const auto exec_3 = execution_space_instances.at(3);

  using policy_t = Kokkos::RangePolicy<TEST_EXECSPACE>;
  using view_t   = Kokkos::View<int*, TEST_EXECSPACE>;
  using view_h_t = Kokkos::View<int*, Kokkos::HostSpace>;

  view_t data(Kokkos::view_alloc(ex, "data"), 6);

  constexpr int value_A = 42, value_B = 27, value_C = 13, value_D = 147,
                value_E = 496, value_F = 123;
  std::integral_constant<size_t, 0> index_A;
  std::integral_constant<size_t, 1> index_B;
  std::integral_constant<size_t, 2> index_C;
  std::integral_constant<size_t, 3> index_D;
  std::integral_constant<size_t, 4> index_E;
  std::integral_constant<size_t, 5> index_F;

  auto graph = Kokkos::Experimental::create_graph(exec_2, [&](auto root) {
    auto node_A = root.then_parallel_for(
        policy_t(exec_0, 0, 1),
        FetchValuesAndContribute(data, index_A, value_A));
    auto node_B = root.then_parallel_for(
        policy_t(exec_1, 0, 1),
        FetchValuesAndContribute(data, index_B, value_B));

    auto node_C = Kokkos::Experimental::when_all(node_A, node_B)
                      .then_parallel_for(
                          policy_t(exec_0, 0, 1),
                          FetchValuesAndContribute(data, {index_A(), index_B()},
                                                   index_C, value_C));

    auto node_D = node_C.then_parallel_for(
        policy_t(exec_0, 0, 1),
        FetchValuesAndContribute(data, {index_C()}, index_D, value_D));
    auto node_E = node_B.then_parallel_for(
        policy_t(exec_1, 0, 1),
        FetchValuesAndContribute(data, {index_B()}, index_E, value_E));
  });
  graph.instantiate();

  // TODO Check that kernels are running on the execution space instance of
  //      their policy if the defaulted graph implementation is used.
  graph.submit(exec_3);

  // clang-format off
  Kokkos::parallel_for(
      policy_t(exec_3, 0, 1),
#if defined(KOKKOS_COMPILER_GNU) && (1010 == KOKKOS_COMPILER_GNU)
      // Workaround CTAD bug, see 7316.
      FetchValuesAndContribute<decltype(data), index_F, 2>(data, {index_D(), index_E()}, index_F, value_F));
#else
      FetchValuesAndContribute(data, {index_D(), index_E()}, index_F, value_F));
#endif
  // clang-format on
  view_h_t data_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "data - host"), 6);

  Kokkos::deep_copy(exec_3, data_host, data);

  exec_3.fence();

  ASSERT_EQ(data_host(index_A()), value_A);
  ASSERT_EQ(data_host(index_B()), value_B);
  ASSERT_EQ(data_host(index_C()), value_A + value_B + value_C);
  ASSERT_EQ(data_host(index_D()), value_A + value_B + value_C + value_D);
  ASSERT_EQ(data_host(index_E()), value_B + value_E);
  ASSERT_EQ(data_host(index_F()),
            value_A + 2 * value_B + value_C + value_D + value_E + value_F);
}

}  // end namespace Test
