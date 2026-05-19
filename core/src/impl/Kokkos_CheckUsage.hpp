// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_CHECK_USAGE_HPP
#define KOKKOS_CHECK_USAGE_HPP

#include <Kokkos_Abort.hpp>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_InitializeFinalize.hpp>
#include <impl/Kokkos_Utilities.hpp>
#include <impl/Kokkos_ErrorMessage.hpp>

#include <string>
#include <string_view>
#include <type_traits>

// FIXME: Obtain file and line number information via std::source_location
// (since C++20) which requires GCC 11 etc.

// Idea: CheckUsage is templated on a type representing
// a condition and implements the condition logic via
// the check method

// Since CheckUsage can be called from different places
// the error output message is templated on a message
// type which contains the actual message.

namespace Kokkos {

namespace Impl {

template <typename... T>
struct CheckUsage;

struct UsageRequires {
  struct isInitialized {};
  struct isNotFinalized {};
  struct insideExecEnv {};
};

// NOLINTBEGIN(bugprone-exception-escape)
template <>
struct CheckUsage<UsageRequires::isInitialized> {
  template <typename T>
  static void check(T msg) noexcept {
    if (!Kokkos::is_initialized()) {
      Kokkos::abort(msg.get().c_str());
    }
  }
};

template <>
struct CheckUsage<UsageRequires::isNotFinalized> {
  template <typename T>
  static void check(T msg) noexcept {
    if (Kokkos::is_finalized()) {
      Kokkos::abort(msg.get().c_str());
    }
  }
};

// NOLINTEND(bugprone-exception-escape)

// Compound condition requires two messages for sub conditions.
// Check isNotFinalized before isInitialized so that after finalize() we report
// "after finalize" (is_finalized() is true) rather than "before init"
// (is_initialized() is false after finalize).
template <>
struct CheckUsage<UsageRequires::insideExecEnv> {
  template <typename T, typename U>
  static void check(T msg1, U msg2) noexcept {
    Kokkos::Impl::CheckUsage<
        Kokkos::Impl::UsageRequires::isNotFinalized>::check(msg2);
    Kokkos::Impl::CheckUsage<Kokkos::Impl::UsageRequires::isInitialized>::check(
        msg1);
  }
};

// Helpers
template <typename T>
void check_execution_space_constructor_precondition(T msg) noexcept {
  Impl::CheckUsage<Impl::UsageRequires::insideExecEnv>::check(
      Impl::Message::Message<
          Impl::Message::Type::InstanceConstructionBeforeInit>(
          std::string(msg)),
      Impl::Message::Message<
          Impl::Message::Type::InstanceConstructionAfterFini>(
          std::string(msg)));
}

template <typename T>
inline void check_execution_space_destructor_precondition(T msg) noexcept {
  Impl::CheckUsage<Impl::UsageRequires::isNotFinalized>::check(
      Impl::Message::Message<Impl::Message::Type::InstanceDestructionAfterFini>(
          std::string(msg)));
}

template <Message::ExecutionPolicyOrInt T>
void check_parallel_precondition(
    const char* func_name, const T& policy,
    const std::string& label = "no-label") noexcept {
  CheckUsage<UsageRequires::insideExecEnv>::check(
      Message::Message<Message::Type::CalledBeforeInit>(
          func_name, Message::toString(policy), label),
      Message::Message<Message::Type::CalledAfterFini>(
          func_name, Message::toString(policy), label));
}

inline void check_view_allocation_precondition(
    std::string_view label) noexcept {
  CheckUsage<UsageRequires::insideExecEnv>::check(
      Message::Message<Message::Type::ViewAllocationBeforeInit>(label),
      Message::Message<Message::Type::ViewAllocationAfterFini>(label));
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_CHECK_USAGE_HPP
