// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_InitializeFinalize.hpp>
#endif

#include <gtest/gtest.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

#include "KokkosExecutionEnvironmentNeverInitializedFixture.hpp"

namespace {

using PushFinalizeHook_DeathTest = KokkosExecutionEnvironmentNeverInitialized;

// Output for the finalize hooks.  Use this to make sure that all the hooks
// ran, and that they ran in the correct order.
std::ostringstream hookOutput;

const char hook1str[] = "Behold, I am Hook 1; first pushed, last to be called.";
const char hook2str[] = "Yea verily, I am Hook 2.";
const char hook3str[] = "Indeed, I am Hook 3.";
const char hook4str[] = "Last but not least, I am Hook 4.";

// Don't just have all the hooks print the same thing except for a number.
// Have them print different things, so we can detect interleaving.  The hooks
// need to run sequentially, in LIFO order.  Also, make sure that the function
// accepts at least the following kinds of hooks:
//
// 1. A plain old function that takes no arguments and returns nothing.
// 2. Lambda, that can be assigned to std::function<void()>
// 3. An actual std::function<void()>
// 4. A named object with operator().  This is what C++ programmers
//    unfortunately like to call "functor," even though this word means
//    something different in other languages.

void hook1() { hookOutput << hook1str << '\n'; }

struct Hook4 {
  void operator()() const { hookOutput << hook4str << '\n'; }
};

TEST_F(PushFinalizeHook_DeathTest, called_in_reverse_order) {
  std::string const expectedOutput([] {
    std::ostringstream os;
    os << hook4str << '\n'
       << hook3str << '\n'
       << hook2str << '\n'
       << hook1str << '\n';
    return os.str();
  }());

  EXPECT_EXIT(
      {
        Kokkos::push_finalize_hook(hook1);  // plain old function
        Kokkos::push_finalize_hook(
            [] { hookOutput << hook2str << '\n'; });  // lambda
        Kokkos::initialize();
        std::function<void()> hook3 = [] { hookOutput << hook3str << '\n'; };
        Kokkos::push_finalize_hook(hook3);  // actual std::function
        Hook4 hook4;
        Kokkos::push_finalize_hook(hook4);  // function object instance

        Kokkos::finalize();
        std::exit(hookOutput.str() == expectedOutput ? EXIT_SUCCESS
                                                     : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

char const my_terminate_handler_msg[] = "my terminate handler was called\n";
[[noreturn]] void my_terminate_handler() {
  std::cerr << my_terminate_handler_msg;
  std::abort();
}

TEST_F(PushFinalizeHook_DeathTest, terminate_on_throw) {
  auto terminate_handler = std::get_terminate();

  std::set_terminate(my_terminate_handler);

  EXPECT_DEATH(
      {
        Kokkos::push_finalize_hook(
            [] { throw std::runtime_error("uncaught exception"); });
        Kokkos::initialize();
        Kokkos::finalize();
      },
      my_terminate_handler_msg);

  std::set_terminate(terminate_handler);
}

TEST_F(PushFinalizeHook_DeathTest, ignore_late_registration) {
  EXPECT_EXIT(
      {
        Kokkos::initialize();
        Kokkos::finalize();
        // legal to register after finalize but will never be called
        Kokkos::push_finalize_hook(
            [] { throw std::runtime_error("never actually thrown"); });
        std::exit(EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST_F(PushFinalizeHook_DeathTest, thread_safe) {
  EXPECT_EXIT(
      ([] {
        constexpr int num_pushes_1 = 8;
        constexpr int num_pushes_2 = 4;
        constexpr int num_pushes_3 = 2;
        int count                  = 0;
        // generates a nullary callable that pushes n times a callback to
        // increment the counter by one
        auto push_increment_n = [&count](int n) {
          return [&count, n] {
            for (int i = 0; i < n; ++i)
              Kokkos::push_finalize_hook([&count] { ++count; });
          };
        };
        Kokkos::initialize(
            Kokkos::InitializationSettings().set_disable_warnings(true));
        std::thread t1(push_increment_n(num_pushes_1));
        std::thread t2(push_increment_n(num_pushes_2));
        std::thread t3(push_increment_n(num_pushes_3));
        t1.join();
        t2.join();
        t3.join();
        Kokkos::finalize();
        std::exit(count == num_pushes_1 + num_pushes_2 + num_pushes_3
                      ? EXIT_SUCCESS
                      : EXIT_FAILURE);
      }()),
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

// Registering a hook from within a running finalize hook must not deadlock,
// since finalize_hooks_mutex is not held while a hook is being called.
TEST_F(PushFinalizeHook_DeathTest, recursive) {
  EXPECT_EXIT(
      {
        bool hook_from_hook_ran = false;
        Kokkos::push_finalize_hook([&hook_from_hook_ran] {
          Kokkos::push_finalize_hook(
              [&hook_from_hook_ran] { hook_from_hook_ran = true; });
        });
        Kokkos::initialize(
            Kokkos::InitializationSettings().set_disable_warnings(true));
        Kokkos::finalize();
        std::exit(hook_from_hook_ran ? EXIT_SUCCESS : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

}  // namespace
