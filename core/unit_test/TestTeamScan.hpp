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
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <type_traits>

namespace Test {

template <class ExecutionSpace, class DataType>
struct TestTeamScan {
  using execution_space = Device;
  using value_type      = DataType;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;
  using view_type       = Kokkos::View<value_type**, execution_space>;

  view_type a_d;
  view_type a_r;
  int32_t M = 0;
  int32_t N = 0;

  KOKKOS_FUNCTION
  void operator()(const member_type& team) const {
    auto leagueRank = team.league_rank();

    auto beg = 0;
    auto end = N;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, beg, end),
        [&](const int i) { a_d(leagueRank, i) += leagueRank * N + i; });

    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, beg, end),
                          [&](int i, int32_t& val, const bool final) {
                            val += a_d(leagueRank, i);
                            if (final) a_r(leagueRank, i) = val;
                          });
  }

  auto operator()(int32_t _M, int32_t _N) {
    M   = _M;
    N   = _N;
    a_d = view_type("a_d", M, N);
    a_r = view_type("a_r", M, N);

    Kokkos::parallel_for(policy_type(M, Kokkos::AUTO()), *this);

    auto a_i = Kokkos::create_mirror_view(a_d);
    auto a_o = Kokkos::create_mirror_view(a_r);
    Kokkos::deep_copy(a_i, a_d);
    Kokkos::deep_copy(a_o, a_r);

    EXPECT_EQ(a_i.extent(0), M);
    EXPECT_EQ(a_o.extent(0), M);
    EXPECT_EQ(a_i.extent(1), N);
    EXPECT_EQ(a_o.extent(1), N);

    for (int32_t i = 0; i < M; ++i) {
      value_type _scan_real = 0;
      value_type _scan_calc = 0;
      value_type _epsilon   = std::numeric_limits<value_type>::round_error();
      for (int32_t j = 0; j < N; ++j) {
        _scan_real += a_i(i, j);
        _scan_calc     = a_o(i, j);
        auto _get_mesg = [=]() {
          std::stringstream ss, idx;
          idx << "(" << i << ", " << j << ") = ";
          ss << "a_d" << idx.str() << a_i(i, j);
          ss << ", a_r" << idx.str() << a_o(i, j);
          return ss.str();
        };
        if (std::is_integral<value_type>::value) {
          ASSERT_EQ(_scan_real, _scan_calc) << _get_mesg();
        } else {
          _epsilon += std::numeric_limits<value_type>::round_error();
          ASSERT_NEAR(_scan_real, _scan_calc, _epsilon) << _get_mesg();
        }
      }
    }
  }
};

TEST(TEST_CATEGORY, team_scan) {
  TestTeamScan<TEST_EXECSPACE, int16_t>{}(99, 32);
  TestTeamScan<TEST_EXECSPACE, uint16_t>{}(139, 64);
  TestTeamScan<TEST_EXECSPACE, int32_t>{}(163, 128);
  TestTeamScan<TEST_EXECSPACE, int64_t>{}(2156, 512);
  TestTeamScan<TEST_EXECSPACE, uint64_t>{}(1234, 1024);
  TestTeamScan<TEST_EXECSPACE, float>{}(108, 16);
  TestTeamScan<TEST_EXECSPACE, float>{}(152, 80);
  TestTeamScan<TEST_EXECSPACE, double>{}(34, 32);
  TestTeamScan<TEST_EXECSPACE, double>{}(956, 128);
  TestTeamScan<TEST_EXECSPACE, double>{}(2596, 512);
  TEST_EXECSPACE().fence();
}

}  // namespace Test
