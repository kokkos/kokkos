// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

namespace {

#if defined(_WIN32) && defined(KOKKOS_ENABLE_CUDA)
constexpr bool is_windows_cuda = std::is_same_v<TEST_EXECSPACE, Kokkos::Cuda>;
#else
constexpr bool is_windows_cuda = false;
#endif

template <class MemorySpace>
void test_kokkos_malloc_bad_alloc() {
  constexpr auto too_large = std::numeric_limits<size_t>::max() - 4096;
  const std::string label  = "kokkos_malloc_bad_alloc";

  try {
    [[maybe_unused]] auto allocation =
        Kokkos::kokkos_malloc<MemorySpace>(label, too_large);
    FAIL() << "It should have thrown.";
  } catch (Kokkos::Experimental::BadAlloc const& error) {
    ASSERT_EQ(error.memory_space_name(), MemorySpace::name());
    ASSERT_GE(error.allocation_size(), too_large);
    ASSERT_EQ(error.label(), label);
  }
}

TEST(TEST_CATEGORY, kokkos_malloc_bad_alloc_default) {
  using MemorySpace = TEST_EXECSPACE::memory_space;

  // These backend workarounds are copied from TestViewBadAlloc.hpp.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  if (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
    GTEST_SKIP() << "AddressSanitizer detects allocating too much memory "
                    "preventing our checks to run";
  }
#endif
#endif
#if defined(KOKKOS_ENABLE_OPENACC)  // FIXME_OPENACC
  if (std::is_same_v<TEST_EXECSPACE, Kokkos::Experimental::OpenACC>) {
    GTEST_SKIP() << "acc_malloc() not properly returning nullptr";
  }
#endif

  if constexpr (is_windows_cuda) {
    GTEST_SKIP() << "MSVC/CUDA segfaults when allocating too much memory";
  }

  test_kokkos_malloc_bad_alloc<MemorySpace>();
}

#ifdef KOKKOS_HAS_SHARED_SPACE
TEST(TEST_CATEGORY, kokkos_malloc_bad_alloc_shared) {
  if constexpr (!std::is_same_v<TEST_EXECSPACE,
                                Kokkos::DefaultExecutionSpace> ||
                std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    GTEST_SKIP()
        << "SharedSpace can only be tested with the default device execution "
           "space";
  }

  if constexpr (is_windows_cuda) {
    GTEST_SKIP() << "MSVC/CUDA segfaults when allocating too much memory";
  }

  test_kokkos_malloc_bad_alloc<Kokkos::SharedSpace>();
}
#endif

#ifdef KOKKOS_HAS_SHARED_HOST_PINNED_SPACE
TEST(TEST_CATEGORY, kokkos_malloc_bad_alloc_shared_host_pinned) {
#ifdef KOKKOS_HAS_SHARED_SPACE
  if constexpr (std::is_same_v<Kokkos::SharedHostPinnedSpace,
                               Kokkos::SharedSpace>) {
    GTEST_SKIP() << "SharedHostPinnedSpace is the same as SharedSpace";
  }
#endif

  if constexpr (!std::is_same_v<TEST_EXECSPACE,
                                Kokkos::DefaultExecutionSpace> ||
                std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    GTEST_SKIP() << "SharedHostPinnedSpace can only be tested with the "
                    "default device execution space";
  }

  if constexpr (is_windows_cuda) {
    GTEST_SKIP() << "MSVC/CUDA segfaults when allocating too much memory";
  }

  test_kokkos_malloc_bad_alloc<Kokkos::SharedHostPinnedSpace>();
}
#endif

}  // namespace
