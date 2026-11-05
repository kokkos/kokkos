# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

# Probe for external GoogleTest with GMock support
#
# This subdirectory probes for an external GTest installation without
# polluting the parent scope. Imported targets from find_package are
# directory-scoped by default, so if the external installation is
# missing GMock, we can discard it here without causing target
# conflicts when falling back to the bundled version.

set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL OFF)

find_package(GTest 1.14.0 CONFIG QUIET)

if(GTest_FOUND AND TARGET GTest::gmock)
  set(KOKKOS_EXTERNAL_GTEST_USABLE TRUE PARENT_SCOPE)
else()
  if(GTest_FOUND)
    if(CMAKE_REQUIRE_FIND_PACKAGE_GTest)
      set(MESSAGE_MODE FATAL_ERROR)
      set(MESSAGE_ACTION "CMAKE_REQUIRE_FIND_PACKAGE_GTest is set, so not falling back to the bundled GoogleTest.")
    else()
      set(MESSAGE_MODE WARNING)
      set(MESSAGE_ACTION
          "Falling back to the bundled GoogleTest. Use -DCMAKE_DISABLE_FIND_PACKAGE_GTest=ON to silence this warning."
      )
    endif()
    message(${MESSAGE_MODE}
            "External GoogleTest found at ${GTest_DIR} (version \"${GTest_VERSION}\") was built without GMock, "
            "which Kokkos tests require. ${MESSAGE_ACTION}"
    )
  endif()
  set(KOKKOS_EXTERNAL_GTEST_USABLE FALSE PARENT_SCOPE)
endif()
