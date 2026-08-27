// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_NEXTSILICON_HPP
#define KOKKOS_TEST_NEXTSILICON_HPP

#include <gtest/gtest.h>
#include <NextSilicon/Kokkos_NextSilicon_DeathTest.hpp>

#define TEST_CATEGORY nextsilicon
#define TEST_CATEGORY_NUMBER 9
#define TEST_CATEGORY_DEATH nextsilicon_DeathTest
#define TEST_EXECSPACE \
  Kokkos::Experimental::NextSilicon  // NextSiliconSharedSpace
#define TEST_CATEGORY_FIXTURE(name) nextsilicon_##name

#endif
