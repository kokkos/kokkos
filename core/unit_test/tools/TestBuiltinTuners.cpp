// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Tools_Generic.hpp>

#include <gtest/gtest.h>

namespace {

using ExecSpace  = Kokkos::DefaultHostExecutionSpace;
using TeamMember = Kokkos::TeamPolicy<ExecSpace>::member_type;

struct TestTeamFunctor {
  KOKKOS_FUNCTION void operator()(TeamMember) const {}
};

struct TestMDFunctor {
  KOKKOS_FUNCTION void operator()(const int, const int) const {}
};

struct TestRangeFunctor {
  KOKKOS_FUNCTION void operator()(const int) const {}
};

TEST(kokkosp, builtin_tuner) {
  Kokkos::TeamPolicy<ExecSpace> teamp(1, Kokkos::AUTO, Kokkos::AUTO);
  Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdp({0, 0}, {1, 1});
  Kokkos::Tools::Experimental::TeamSizeTuner team_tune_this(
      "team_tuner", teamp, TestTeamFunctor{}, Kokkos::ParallelForTag{},
      Kokkos::Tools::Experimental::Impl::Impl::SimpleTeamSizeCalculator{});

  Kokkos::Tools::Experimental::MDRangeTuner<2> md_tune_this(
      "md_tuner", mdp, TestMDFunctor{}, Kokkos::ParallelForTag{},
      Kokkos::Tools::Experimental::Impl::Impl::SimpleTeamSizeCalculator{});

  std::vector<int> options{1, 2, 3, 4, 5};

  auto new_team_tuner = team_tune_this.combine("options", options);
  auto new_md_tuner   = md_tune_this.combine("options", options);
  using namespace Kokkos::Tools::Experimental;
  VariableInfo info;
  info.category      = StatisticalCategory::kokkos_value_categorical;
  info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
  info.type          = ValueType::kokkos_value_string;
  size_t input       = declare_input_type("kernel", info);
  VariableValue team_kernel_value = make_variable_value(input, "abs");
  VariableValue md_kernel_value   = make_variable_value(input, "abs");
  size_t kernel_context           = get_new_context_id();
  begin_context(kernel_context);
  set_input_values(kernel_context, 1, &team_kernel_value);
  for (int x = 0; x < 10000; ++x) {
    auto config = new_md_tuner.begin();
    int option  = std::get<0>(config);
    (void)option;
    int tile_x = std::get<1>(config);
    int tile_y = std::get<2>(config);
    Kokkos::parallel_for("mdrange",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {1, 1}, {tile_x, tile_y}),
                         TestMDFunctor{});
    new_md_tuner.end();
  }
  end_context(kernel_context);
  begin_context(kernel_context);
  set_input_values(kernel_context, 1, &md_kernel_value);

  /**
   * Note that 0.0 is basically a floating point index into
   * the outermost index in this, which is the options vector
   * above. The At 0.0, this will be the first element (1).
   * At 0.9 this will be the last element (5)
   */
  auto begin_point = new_team_tuner.get_point(0.0, 0.0, 0.0);
  EXPECT_EQ(std::get<0>(begin_point), 1);
  auto end_point = new_team_tuner.get_point(0.9, 0.0, 0.0);
  EXPECT_EQ(std::get<0>(end_point), 5);
  for (int x = 0; x < 10000; ++x) {
    auto config                 = new_team_tuner.begin();
    [[maybe_unused]] int option = std::get<0>(config);
    int team                    = std::get<1>(config);
    int vector                  = std::get<2>(config);
    Kokkos::parallel_for("mdrange",
                         Kokkos::TeamPolicy<ExecSpace>(1, team, vector),
                         TestTeamFunctor{});
    new_team_tuner.end();
  }
  end_context(kernel_context);
}

template <typename T>
void validate_tile_sizes(const std::vector<T>& cont,
                         Kokkos::Array<int, 6>& current_tile,
                         const Kokkos::Array<int, 3>& hw_tile_limits,
                         int current_rank, const int policy_rank) {
  namespace Impl = Kokkos::Tools::Experimental::Impl;
  for (const auto& size : cont) {
    current_tile[current_rank] = size;
    EXPECT_TRUE(Impl::valid_tile(hw_tile_limits, current_tile, policy_rank));
  }
}

template <typename KeyType, typename Mapped>
void validate_tile_sizes(std::map<KeyType, Mapped>& cont,
                         Kokkos::Array<int, 6>& current_tile,
                         const Kokkos::Array<int, 3>& hw_tile_limits,
                         int current_rank, const int policy_rank) {
  for (auto it = cont.begin(); it != cont.end(); it++) {
    KeyType key                     = it->first;
    current_tile[current_rank]      = key;
    Kokkos::Array<int, 6> next_tile = current_tile;
    validate_tile_sizes(it->second, next_tile, hw_tile_limits, current_rank + 1,
                        policy_rank);
  }
}

template <typename KeyType, typename Mapped>
void validate_tile_sizes(std::map<KeyType, Mapped>& cont,
                         const Kokkos::Array<int, 3>& hw_tile_limits,
                         int policy_rank) {
  Kokkos::Array<int, 6> current_tile{1, 1, 1, 1, 1, 1};
  validate_tile_sizes(cont, current_tile, hw_tile_limits, 0, policy_rank);
}

template <int test_policy_rank>
void test_tile_constraints() {
  using namespace Kokkos::Tools::Experimental::Impl;
  using SpaceDescription =
      typename n_dimensional_sparse_structure<int, test_policy_rank>::type;

  Kokkos::Array<int, 3> hw_const{128, 128, 128};
  SpaceDescription tile_configuration_space;
  const int max_total_tile_size = 512;

  fill_tile(tile_configuration_space, max_total_tile_size);
  apply_tiles_constraints(tile_configuration_space, hw_const, test_policy_rank);
  validate_tile_sizes(tile_configuration_space, hw_const, test_policy_rank);
}

TEST(kokkosp, constraint_tiles) {
  namespace Impl = Kokkos::Tools::Experimental::Impl;

  Kokkos::Array<int, 3> hw_const{128, 128, 128};
  Kokkos::Array<int, 6> valid_tile{4, 2, 8, 4, 2, 8};
  Kokkos::Array<int, 6> invalid_tile{256, 256, 256, 1, 1, 1};

  EXPECT_TRUE(Impl::valid_tile(hw_const, valid_tile, 6));
  EXPECT_FALSE(Impl::valid_tile(hw_const, invalid_tile, 3));

  test_tile_constraints<2>();
  test_tile_constraints<3>();
  test_tile_constraints<4>();
  test_tile_constraints<5>();
  test_tile_constraints<6>();
}

}  // namespace
