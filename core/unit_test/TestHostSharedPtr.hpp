// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <impl/Kokkos_HostSharedPtr.hpp>

#include <gtest/gtest.h>

#include <array>
// FIXME gcc-10 doesn't have latch
#if __has_include(<latch>)
#include <latch>
#endif
#include <thread>

using Kokkos::Impl::HostSharedPtr;

TEST(TEST_CATEGORY, host_shared_ptr_use_count) {
  using T = int;
  {
    HostSharedPtr<T> p1;
    EXPECT_EQ(p1.use_count(), 0);
  }
  {
    HostSharedPtr<T> p1(nullptr);
    EXPECT_EQ(p1.use_count(), 0);
  }
  {
    HostSharedPtr<T> p1(new T());
    EXPECT_EQ(p1.use_count(), 1);
  }
  {
    HostSharedPtr<T> p1(new T(), [](T* p) { delete p; });
    EXPECT_EQ(p1.use_count(), 1);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    EXPECT_EQ(p1.use_count(), 1);
  }
  {
    HostSharedPtr<T> p1(new T());
    HostSharedPtr<T> p2(p1);  // copy construction
    EXPECT_EQ(p1.use_count(), 2);
    EXPECT_EQ(p2.use_count(), 2);
  }
  {
    HostSharedPtr<T> p1(new T());
    HostSharedPtr<T> p2(std::move(p1));  // move construction
    EXPECT_EQ(p2.use_count(), 1);
  }
  {
    HostSharedPtr<T> p1(new T());
    HostSharedPtr<T> p2;
    p2 = p1;  // copy assignment
    EXPECT_EQ(p1.use_count(), 2);
    EXPECT_EQ(p2.use_count(), 2);
  }
  {
    HostSharedPtr<T> p1(new T());
    HostSharedPtr<T> p2;
    p2 = std::move(p1);  // move assignment
    EXPECT_EQ(p2.use_count(), 1);
  }
}

TEST(TEST_CATEGORY, host_shared_ptr_get) {
  using T = int;
  {
    HostSharedPtr<T> p1;
    EXPECT_EQ(p1.get(), nullptr);
  }
  {
    HostSharedPtr<T> p1(nullptr);
    EXPECT_EQ(p1.get(), nullptr);
  }
  {
    T* p_i = new T();
    HostSharedPtr<T> p1(p_i);
    EXPECT_EQ(p1.get(), p_i);
  }
  {
    T* p_i = new T();
    HostSharedPtr<T> p1(p_i, [](T* p) { delete p; });
    EXPECT_EQ(p1.get(), p_i);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    EXPECT_EQ(p1.get(), &i);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    HostSharedPtr<T> p2(p1);  // copy construction
    EXPECT_EQ(p1.get(), &i);
    EXPECT_EQ(p1.get(), &i);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    HostSharedPtr<T> p2(std::move(p1));  // move construction
    EXPECT_EQ(p1.get(), nullptr);        // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(p2.get(), &i);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    HostSharedPtr<T> p2;
    p2 = p1;  // copy assignment
    EXPECT_EQ(p1.get(), &i);
    EXPECT_EQ(p2.get(), &i);
  }
  {
    T i;
    HostSharedPtr<T> p1(&i, [](T*) {});
    HostSharedPtr<T> p2;
    p2 = std::move(p1);            // move assignment
    EXPECT_EQ(p1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(p2.get(), &i);
  }
}

// FIXME gcc-10 doesn't have latch
#if __has_include(<latch>)
TEST(TEST_CATEGORY, host_shared_ptr_use_count_under_contention) {
  const HostSharedPtr<int> hsp{new int(42)};

  constexpr std::size_t num_threads = 2;

  std::latch all_copied{num_threads};
  std::latch release{1};

// FIXME_APPLECC
#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
  using thread_t = std::jthread;
#else
#define KOKKOS_TEST_WORKAROUND_JTHREAD_NOT_AVAILABLE
  using thread_t = std::thread;
#endif
  std::array<thread_t, num_threads> threads;
  for (auto& thread : threads)
    thread = thread_t([&] {
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      const auto l_c = hsp;
      all_copied.count_down();
      EXPECT_GE(l_c.use_count(), 2);
      release.wait();
    });
  all_copied.wait();

  EXPECT_EQ(hsp.use_count(), 1 + num_threads);

  release.count_down();

// FIXME_APPLECC
#if defined(KOKKOS_TEST_WORKAROUND_JTHREAD_NOT_AVAILABLE)
  for (auto& thread : threads) thread.join();
#endif
}
#endif
