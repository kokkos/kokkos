// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_ERROR_MESSAGES_HPP
#define KOKKOS_ERROR_MESSAGES_HPP

#include <Kokkos_Concepts.hpp>

#include <sstream>
#include <string>
#include <type_traits>

namespace Kokkos {

template <typename... Args>
class RangePolicy;

template <typename... Args>
struct MDRangePolicy;

template <typename... Args>
class TeamPolicy;

namespace Impl {
namespace Message {

template <typename T>
concept ExecutionPolicyOrInt =
    Kokkos::is_execution_policy_v<T> || std::is_same_v<T, int> ||
    std::is_same_v<T, std::size_t>;

template <ExecutionPolicyOrInt T>
const std::string toString(const T&) {
  if constexpr ((std::is_same_v<T, std::size_t>) ||
                (Kokkos::Impl::is_specialization_of_v<T, Kokkos::RangePolicy>))
    return "exec policy RangePolicy";

  if constexpr (Kokkos::Impl::is_specialization_of_v<T, Kokkos::MDRangePolicy>)
    return "exec policy MDRangePolicy";

  if constexpr (Kokkos::Impl::is_specialization_of_v<T, Kokkos::TeamPolicy>)
    return "exec policy TeamPolicy";

  return "probably nested execution policy";
}

// add more toString variants for type specific string conversions

// Enumeration of message types
enum class Type {
  CalledBeforeInit,
  CalledAfterFini,
  InstanceConstructionBeforeInit,
  InstanceConstructionAfterFini,
  InstanceDestructionAfterFini,
  ViewAllocationBeforeInit,
  ViewAllocationAfterFini
};

template <Type>
struct Message;

template <>
struct Message<Type::CalledBeforeInit> {
  std::string arg1, arg2, arg3;
  Message(const std::string& arg1_, const std::string& arg2_,
          const std::string& arg3_ = "no-label")
      : arg1(arg1_), arg2(arg2_), arg3(arg3_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: attempting to call " << arg1
       << "() **before** Kokkos::initialize() was called."
       << " Concerns " << arg3 << " with " << arg2 << ".";
    return ss.str();
  }
};

template <>
struct Message<Type::CalledAfterFini> {
  std::string arg1, arg2, arg3;
  Message(const std::string& arg1_, const std::string& arg2_,
          const std::string& arg3_ = "no-label")
      : arg1(arg1_), arg2(arg2_), arg3(arg3_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: attempting to call " << arg1
       << "() **after** Kokkos::finalize() was called."
       << " Concerns " << arg3 << " with " << arg2 << ".";
    return ss.str();
  }
};

template <>
struct Message<Type::InstanceConstructionBeforeInit> {
  std::string arg;
  Message(const std::string& arg_) : arg(arg_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: " << arg
       << " execution space is being constructed before initialize() has been "
          "called";
    return ss.str();
  }
};

template <>
struct Message<Type::InstanceConstructionAfterFini> {
  std::string arg;
  Message(const std::string& arg_) : arg(arg_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: " << arg
       << " execution space is being constructed after finalize() has been "
          "called";
    return ss.str();
  }
};

template <>
struct Message<Type::InstanceDestructionAfterFini> {
  std::string arg;
  Message(const std::string& arg_) : arg(arg_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: " << arg
       << " execution space is being destructed after finalize() has been "
          "called";
    return ss.str();
  }
};

template <>
struct Message<Type::ViewAllocationBeforeInit> {
  std::string label;
  Message(std::string_view label_) : label(label_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: View ";
    if (!label.empty()) ss << "(label=\"" << label << "\") ";
    ss << "is being constructed before initialize() has been called";
    return ss.str();
  }
};

template <>
struct Message<Type::ViewAllocationAfterFini> {
  std::string label;
  Message(std::string_view label_) : label(label_) {}

  std::string get() const {
    std::ostringstream ss;
    ss << "Kokkos ERROR: View ";
    if (!label.empty()) ss << "(label=\"" << label << "\") ";
    ss << "is being constructed after finalize() has been called";
    return ss.str();
  }
};

}  // namespace Message
}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ERROR_MESSAGES_HPP
