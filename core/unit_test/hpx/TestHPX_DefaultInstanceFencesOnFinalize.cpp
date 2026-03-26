// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHPX_Category.hpp>

#include "tools/include/ToolTestingUtilities.hpp"

namespace Test {

// Test whether the HPX default instance fences on finalize.
TEST(hpx, default_instance_fences_on_finalize) {
  Kokkos::Test::Tools::listen_tool_events(
      Kokkos::Test::Tools::Config::DisableAll(),
      Kokkos::Test::Tools::Config::EnableFences());

  auto success = Kokkos::Test::Tools::validate_existence(
      [&]() {
        Kokkos::initialize();
        {
          const Kokkos::Experimental::HPX exec{};

          Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 100),
                               KOKKOS_LAMBDA(const auto){});
        }
        Kokkos::finalize();
      },
      [&](Kokkos::Test::Tools::BeginFenceEvent event) {
        return Kokkos::Test::Tools::MatchDiagnostic{
            event.descriptor().find("fence on destruction") !=
            std::string::npos};
      });

  ASSERT_TRUE(success);

  Kokkos::Test::Tools::listen_tool_events(
      Kokkos::Test::Tools::Config::DisableAll());
}

}  // namespace Test
