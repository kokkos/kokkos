// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <type_traits>

namespace {

struct TestTeamThreadMDRangeCTAD {
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using TeamHandle = TeamPolicy::member_type;

  KOKKOS_FUNCTION void operator()(TeamHandle const& team_handle) const {
    // Rank 2 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange md_range(team_handle, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 3 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange md_range(team_handle, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 4 TeamThreadMDRange
    {
      Kokkos::Array<int64_t, 4> lower{0, 0, 0, 0};
      Kokkos::Array<int, 4> upper{0, 0, 0, 0};
      Kokkos::TeamThreadMDRange md_range(team_handle, lower, upper);
      static_assert(
          std::is_same_v<
              Kokkos::TeamThreadMDRange<Kokkos::Rank<4>, TeamHandle, int64_t>,
              decltype(md_range)>);
    }

    // Rank 5 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange<Kokkos::Rank<5>, TeamHandle, int64_t> md_range(
          team_handle, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<
              Kokkos::TeamThreadMDRange<Kokkos::Rank<5>, TeamHandle, int64_t>,
              decltype(md_range)>);
    }

    // Rank 6 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange md_range(team_handle, {0, 0, 0, 0, 0, 0},
                                         {0, 0, 0, 0, 0, 0});
      static_assert(
          std::is_same_v<Kokkos::TeamThreadMDRange<Kokkos::Rank<6>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 7 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamThreadMDRange<Kokkos::Rank<7>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 8 TeamThreadMDRange
    {
      Kokkos::TeamThreadMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamThreadMDRange<Kokkos::Rank<8>, TeamHandle>,
                         decltype(md_range)>);
    }
  }

  TestTeamThreadMDRangeCTAD() {
    Kokkos::parallel_for(TeamPolicy(0, Kokkos::AUTO), *this);
  }
};

struct TestTeamVectorMDRangeCTAD {
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using TeamHandle = TeamPolicy::member_type;

  KOKKOS_FUNCTION void operator()(TeamHandle const& team_handle) const {
    // Rank 2 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange md_range(team_handle, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamVectorMDRange<Kokkos::Rank<2>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 3 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange md_range(team_handle, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamVectorMDRange<Kokkos::Rank<3>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 4 TeamVectorMDRange
    {
      Kokkos::Array<int64_t, 4> lower{0, 0, 0, 0};
      Kokkos::Array<int, 4> upper{0, 0, 0, 0};
      Kokkos::TeamVectorMDRange md_range(team_handle, lower, upper);
      static_assert(
          std::is_same_v<
              Kokkos::TeamVectorMDRange<Kokkos::Rank<4>, TeamHandle, int64_t>,
              decltype(md_range)>);
    }

    // Rank 5 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange<Kokkos::Rank<5>, TeamHandle, int64_t> md_range(
          team_handle, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<
              Kokkos::TeamVectorMDRange<Kokkos::Rank<5>, TeamHandle, int64_t>,
              decltype(md_range)>);
    }

    // Rank 6 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange md_range(team_handle, {0, 0, 0, 0, 0, 0},
                                         {0, 0, 0, 0, 0, 0});
      static_assert(
          std::is_same_v<Kokkos::TeamVectorMDRange<Kokkos::Rank<6>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 7 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamVectorMDRange<Kokkos::Rank<7>, TeamHandle>,
                         decltype(md_range)>);
    }

    // Rank 8 TeamVectorMDRange
    {
      Kokkos::TeamVectorMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0, 0);
      static_assert(
          std::is_same_v<Kokkos::TeamVectorMDRange<Kokkos::Rank<8>, TeamHandle>,
                         decltype(md_range)>);
    }
  }

  TestTeamVectorMDRangeCTAD() {
    Kokkos::parallel_for(TeamPolicy(0, Kokkos::AUTO), *this);
  }
};

struct TestThreadVectorMDRangeCTAD {
  using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  using TeamHandle = TeamPolicy::member_type;

  template <class PolicyTypeExpected, class PolicyTypeToCheck>
  KOKKOS_FUNCTION static void check_types(
      [[maybe_unused]] PolicyTypeToCheck const& team_handle) {
    static_assert(std::is_same_v<PolicyTypeExpected, PolicyTypeToCheck>);
  }

  KOKKOS_FUNCTION void operator()(TeamHandle const& team_handle) const {
    // Rank 2 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange md_range(team_handle, 0, 0);
      check_types<Kokkos::ThreadVectorMDRange<Kokkos::Rank<2>, TeamHandle>>(
          md_range);
    }

    // Rank 3 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange md_range(team_handle, 0, 0, 0);
      check_types<Kokkos::ThreadVectorMDRange<Kokkos::Rank<3>, TeamHandle>>(
          md_range);
    }

    // Rank 4 ThreadVectorMDRange
    {
      Kokkos::Array<int64_t, 4> lower{0, 0, 0, 0};
      Kokkos::Array<int, 4> upper{0, 0, 0, 0};
      Kokkos::ThreadVectorMDRange md_range(team_handle, lower, upper);
      check_types<
          Kokkos::ThreadVectorMDRange<Kokkos::Rank<4>, TeamHandle, int64_t>>(
          md_range);
    }

    // Rank 5 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange<Kokkos::Rank<5>, TeamHandle, int64_t>
          md_range(team_handle, 0, 0, 0, 0, 0);
      check_types<
          Kokkos::ThreadVectorMDRange<Kokkos::Rank<5>, TeamHandle, int64_t>>(
          md_range);
    }

    // Rank 6 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange md_range(team_handle, {0, 0, 0, 0, 0, 0},
                                           {0, 0, 0, 0, 0, 0});
      check_types<Kokkos::ThreadVectorMDRange<Kokkos::Rank<6>, TeamHandle>>(
          md_range);
    }

    // Rank 7 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0);
      check_types<Kokkos::ThreadVectorMDRange<Kokkos::Rank<7>, TeamHandle>>(
          md_range);
    }

    // Rank 8 ThreadVectorMDRange
    {
      Kokkos::ThreadVectorMDRange md_range(team_handle, 0, 0, 0, 0, 0, 0, 0, 0);
      check_types<Kokkos::ThreadVectorMDRange<Kokkos::Rank<8>, TeamHandle>>(
          md_range);
    }
  }

  TestThreadVectorMDRangeCTAD() {
    Kokkos::parallel_for(TeamPolicy(0, Kokkos::AUTO), *this);
  }
};

}  // namespace
