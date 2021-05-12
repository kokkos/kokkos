
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

#include <TestDefaultDeviceType_Category.hpp>
struct TF {
  KOKKOS_FUNCTION void operator()(const int) const {}
};
namespace Test {

TEST(defaultdevicetype, development_test) {
  Kokkos::RangePolicy<> pol(0, 10000);
  auto next_pol =
      Kokkos::Experimental::prefer(pol, Kokkos::Experimental::TuneOccupancy{});
  TF f;
  using namespace Kokkos::Tools::Experimental;
  VariableInfo info;
  info.category      = StatisticalCategory::kokkos_value_categorical;
  info.type          = ValueType::kokkos_value_int64;
  info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
  size_t id          = declare_input_type("dogggo", info);
  auto ctx           = get_new_context_id();
  auto v             = make_variable_value(id, int64_t(1));
  begin_context(ctx);
  set_input_values(ctx, 1, &v);
  Kokkos::Tools::Experimental::RangePolicyOccupancyTuner tuner(
      "dogs", next_pol, f, Kokkos::ParallelForTag{},
      Kokkos::Tools::Impl::Impl::SimpleTeamSizeCalculator{});

  for (int x = 0; x < 10000; ++x) {
    // auto nexter_pol = tuner.tune(next_pol);
    // usleep(10 * abs(33 - nexter_pol.impl_get_desired_occupancy().value()));
    // tuner.end();
    Kokkos::parallel_for(
        "puppies",
        Kokkos::Experimental::prefer(Kokkos::RangePolicy<>(0, 1),
                                     Kokkos::Experimental::TuneOccupancy{}),
        KOKKOS_LAMBDA(int i){});
  }
  end_context(ctx);
}

}  // namespace Test
