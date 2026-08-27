// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_DEATH_TEST_HPP
#define KOKKOS_NEXTSILICON_DEATH_TEST_HPP

#include <csetjmp>
#include <csignal>
#include <mutex>
#include <type_traits>
#include <functional>

/// Include GoogleTest now, so it never gets included after macro redefinitions
#include <gtest/gtest.h>

namespace Kokkos::Impl {

/// Process-shared jump buffer used to siglongjmp out of the SIGABRT handler.
///
/// Thread-safety invariant: SIGABRT must be raised on the same thread that
/// called `sigsetjmp`. On Linux, `abort()` emits the signal via
/// `pthread_kill(pthread_self(), SIGABRT)`, targeting the calling thread.
/// If the handler were to call `siglongjmp` from a different thread, it would
/// restore the wrong thread's stack pointer and CPU state — undefined behavior.
/// This means `fn()` may spawn worker threads, but must only abort from the
/// thread that originally called `NextSilicon_DeathTest_RaisesSigabrt`.
inline sigjmp_buf NextSilicon_DeathTest_JumpBuf;

inline void NextSilicon_DeathTest_AbortHandler(const int /*signum*/) {
  // Pass 1 as the value sigsetjmp will return on the jump; any non-zero value
  // works since the call site only tests for == 0.
  siglongjmp(NextSilicon_DeathTest_JumpBuf, 1);
}

/// Serializes concurrent `NextSilicon_DeathTest_RaisesSigabrt` calls across
/// threads. `std::signal` is process-wide, so two threads installing/restoring
/// SIGABRT handlers concurrently could leave one of them unprotected.
/// Non-recursive on purpose: nesting `NextSilicon_DeathTest_RaisesSigabrt`
/// from within a `fn()` will deadlock immediately, which surfaces the design
/// problem instead of silently corrupting the jump buffer.
inline std::mutex NextSilicon_DeathTest_Sigaction_Mutex;

/// RAII: installs `new_handler` via std::signal on construction, restores the
/// previous handler on destruction — including during C++ exception unwinding.
/// Uses std::signal rather than POSIX sigaction; signal mask management is
/// handled separately by sigsetjmp/siglongjmp (see
/// NextSilicon_DeathTest_RaisesSigabrt).
class [[nodiscard]] NextSilicon_DeathTest_HandlerGuard {
 public:
  using SignalHandler = void (*)(int);

  NextSilicon_DeathTest_HandlerGuard(const int signum,
                                     SignalHandler new_handler) noexcept
      : signum_(signum), old_handler_(std::signal(signum, new_handler)) {}

  ~NextSilicon_DeathTest_HandlerGuard() { std::signal(signum_, old_handler_); }

  NextSilicon_DeathTest_HandlerGuard(
      const NextSilicon_DeathTest_HandlerGuard&) = delete;
  NextSilicon_DeathTest_HandlerGuard& operator=(
      const NextSilicon_DeathTest_HandlerGuard&) = delete;
  NextSilicon_DeathTest_HandlerGuard(NextSilicon_DeathTest_HandlerGuard&&) =
      delete;
  NextSilicon_DeathTest_HandlerGuard& operator=(
      NextSilicon_DeathTest_HandlerGuard&&) = delete;

 private:
  const int signum_;
  SignalHandler old_handler_;
};

/// Returns true if `fn()` raised SIGABRT (and was caught by our handler);
/// false if it returned normally. Not reentrant — concurrent calls serialize on
/// a mutex; nesting from within fn() deadlocks.
template <typename Fn>
  requires std::is_nothrow_invocable_v<Fn>
[[nodiscard]] bool NextSilicon_DeathTest_RaisesSigabrt(Fn&& fn) noexcept {
  std::lock_guard<std::mutex> lock(NextSilicon_DeathTest_Sigaction_Mutex);

  NextSilicon_DeathTest_HandlerGuard guard(SIGABRT,
                                           NextSilicon_DeathTest_AbortHandler);

  // Perform the death test.
  volatile bool died = true;
  // On SIGABRT, the handler calls siglongjmp, which restores the CPU state
  // saved by sigsetjmp. Execution resumes here as if sigsetjmp returned a
  // second time, this time with value 1, so the body is skipped and `died`
  // keeps its initial `true`.
  if (sigsetjmp(NextSilicon_DeathTest_JumpBuf, 1) == 0) {
    std::invoke(std::forward<Fn>(fn));
    died = false;
  }

  return died;
}

}  // namespace Kokkos::Impl

// Catches SIGABRT raised by `statement` using a signal handler + siglongjmp,
// instead of fork/exec'ing a subprocess like GoogleTest. The `regex` argument
// is accepted for source compatibility but ignored — there is no captured
// stderr to match against. The `_EXIT` variants additionally accept (and
// ignore) the GoogleTest exit predicate — since the statement never actually
// exits the process, no predicate can be evaluated against it.
#define NEXT_DEATH_TEST_IMPL(gtest_assert, statement, regex)           \
  gtest_assert(::Kokkos::Impl::NextSilicon_DeathTest_RaisesSigabrt(    \
      [&]() noexcept { statement; }))                                  \
      << "Expected `" #statement "` to raise SIGABRT, but it returned" \
      << " (regex `" << (regex) << "` was ignored — no captured stderr)"

#define NEXT_EXIT_TEST_IMPL(gtest_assert, statement, predicate, regex) \
  gtest_assert(::Kokkos::Impl::NextSilicon_DeathTest_RaisesSigabrt(    \
      [&]() noexcept { statement; }))                                  \
      << "Expected `" #statement "` to raise SIGABRT, but it returned" \
      << " (predicate `" #predicate "` and regex `" << (regex)         \
      << "` were ignored — no captured stderr or exit status)"

#define NEXT_ASSERT_DEATH(statement, regex) \
  NEXT_DEATH_TEST_IMPL(ASSERT_TRUE, statement, regex)
#define NEXT_EXPECT_DEATH(statement, regex) \
  NEXT_DEATH_TEST_IMPL(EXPECT_TRUE, statement, regex)
#define NEXT_ASSERT_EXIT(statement, predicate, regex) \
  NEXT_EXIT_TEST_IMPL(ASSERT_TRUE, statement, predicate, regex)
#define NEXT_EXPECT_EXIT(statement, predicate, regex) \
  NEXT_EXIT_TEST_IMPL(EXPECT_TRUE, statement, predicate, regex)

#ifdef ASSERT_DEATH
#undef ASSERT_DEATH
#endif
#ifdef EXPECT_DEATH
#undef EXPECT_DEATH
#endif
#ifdef ASSERT_EXIT
#undef ASSERT_EXIT
#endif
#ifdef EXPECT_EXIT
#undef EXPECT_EXIT
#endif
#define ASSERT_DEATH(statement, regex) NEXT_ASSERT_DEATH(statement, regex)
#define EXPECT_DEATH(statement, regex) NEXT_EXPECT_DEATH(statement, regex)
#define ASSERT_EXIT(statement, predicate, regex) \
  NEXT_ASSERT_EXIT(statement, predicate, regex)
#define EXPECT_EXIT(statement, predicate, regex) \
  NEXT_EXPECT_EXIT(statement, predicate, regex)

#endif
