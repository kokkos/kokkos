// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <impl/Kokkos_Command_Line_Parsing.hpp>
#include <impl/Kokkos_Error.hpp>

#include <cctype>
#include <cstring>
#include <iostream>
#include <regex>
#include <string>
#include <sstream>
#include <vector>

namespace {

bool string_equal_case_insensitive(char const* lhs, char const* rhs) {
  while (*lhs != '\0' && *rhs != '\0') {
    auto const lhs_char = static_cast<unsigned char>(*lhs);
    auto const rhs_char = static_cast<unsigned char>(*rhs);
    if (std::tolower(lhs_char) != std::tolower(rhs_char)) {
      return false;
    }
    ++lhs;
    ++rhs;
  }
  return *lhs == '\0' && *rhs == '\0';
}

bool parse_bool(char const* str, bool& val) {
  if (string_equal_case_insensitive(str, "yes") ||
      string_equal_case_insensitive(str, "true") ||
      string_equal_case_insensitive(str, "1")) {
    val = true;
    return true;
  }

  if (string_equal_case_insensitive(str, "no") ||
      string_equal_case_insensitive(str, "false") ||
      string_equal_case_insensitive(str, "0")) {
    val = false;
    return true;
  }

  return false;
}

std::vector<std::regex>& do_not_warn_regular_expressions() {
  static std::vector<std::regex> expressions;
  return expressions;
}

}  // namespace

bool Kokkos::Impl::is_unsigned_int(const char* str) {
  const size_t len = strlen(str);
  for (size_t i = 0; i < len; ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

bool Kokkos::Impl::check_arg(char const* arg, char const* expected) {
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  if (arg_len < exp_len) return false;
  if (std::strncmp(arg, expected, exp_len) != 0) return false;
  if (arg_len == exp_len) return true;

  if (std::isalnum(arg[exp_len]) || arg[exp_len] == '-' ||
      arg[exp_len] == '_') {
    return false;
  }
  return true;
}

bool Kokkos::Impl::check_arg_starts_with(char const* arg,
                                         char const* expected) {
  std::size_t arg_len = std::strlen(arg);
  std::size_t exp_len = std::strlen(expected);
  if (arg_len < exp_len) return false;
  return std::strncmp(arg, expected, exp_len) == 0;
}

bool Kokkos::Impl::check_arg_starts_with_optional_leading_dash(
    char const* arg, char const* expected) {
  if (check_arg_starts_with(arg, expected)) return true;
  if (expected[0] != '-' || arg[0] != '-') return false;
  return check_arg_starts_with(arg + 1, expected);
}

bool Kokkos::Impl::check_env_bool(char const* name, bool& val) {
  char const* var = std::getenv(name);

  if (!var) {
    return false;
  }

  if (parse_bool(var, val)) {
    return true;
  }

  std::stringstream ss;
  ss << "Error: cannot convert environment variable '" << name << "=" << var
     << "' to a boolean."
     << " Raised by Kokkos::initialize().\n";
  Kokkos::abort(ss.str().c_str());
}

bool Kokkos::Impl::check_env_int(char const* name, int& val) {
  char const* var = std::getenv(name);

  if (!var) {
    return false;
  }

  errno = 0;
  char* var_end;
  val = std::strtol(var, &var_end, 10);

  if (var == var_end) {
    std::stringstream ss;
    ss << "Error: cannot convert environment variable '" << name << '=' << var
       << "' to an integer."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  if (errno == ERANGE) {
    std::stringstream ss;
    ss << "Error: converted value for environment variable '" << name << '='
       << var << "' falls out of range."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  return true;
}

bool Kokkos::Impl::check_arg_bool(char const* arg, char const* name,
                                  bool& val) {
  auto const len = std::strlen(name);
  if (std::strncmp(arg, name, len) != 0) {
    return false;
  }
  auto const arg_len = strlen(arg);
  if (arg_len == len) {
    val = true;  // --kokkos-foo without =BOOL interpreted as fool=true
    return true;
  }
  if (arg_len <= len + 1 || arg[len] != '=') {
    std::stringstream ss;
    ss << "Error: command line argument '" << arg
       << "' is not recognized as a valid boolean."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  std::advance(arg, len + 1);
  if (parse_bool(arg, val)) {
    return true;
  }

  std::stringstream ss;
  ss << "Error: cannot convert command line argument '" << name << "=" << arg
     << "' to a boolean."
     << " Raised by Kokkos::initialize().\n";
  Kokkos::abort(ss.str().c_str());
}

bool Kokkos::Impl::check_arg_int(char const* arg, char const* name, int& val) {
  auto const len = std::strlen(name);
  if (std::strncmp(arg, name, len) != 0) {
    return false;
  }
  auto const arg_len = strlen(arg);
  if (arg_len <= len + 1 || arg[len] != '=') {
    std::stringstream ss;
    ss << "Error: command line argument '" << arg
       << "' is not recognized as a valid integer."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  std::advance(arg, len + 1);

  errno = 0;
  char* arg_end;
  val = std::strtol(arg, &arg_end, 10);

  if (arg == arg_end) {
    std::stringstream ss;
    ss << "Error: cannot convert command line argument '" << name << '=' << arg
       << "' to an integer."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  if (errno == ERANGE) {
    std::stringstream ss;
    ss << "Error: converted value for command line argument '" << name << '='
       << arg << "' falls out of range."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  return true;
}

bool Kokkos::Impl::check_arg_str(char const* arg, char const* name,
                                 std::string& val) {
  auto const len = std::strlen(name);
  if (std::strncmp(arg, name, len) != 0) {
    return false;
  }
  auto const arg_len = strlen(arg);
  if (arg_len <= len + 1 || arg[len] != '=') {
    std::stringstream ss;
    ss << "Error: command line argument '" << arg
       << "' is not recognized as a valid string."
       << " Raised by Kokkos::initialize().\n";
    Kokkos::abort(ss.str().c_str());
  }

  std::advance(arg, len + 1);

  val = arg;
  return true;
}

void Kokkos::Impl::warn_deprecated_environment_variable(
    std::string deprecated) {
  std::cerr << "Warning: environment variable '" << deprecated
            << "' is deprecated."
            << " Raised by Kokkos::initialize()." << std::endl;
}

void Kokkos::Impl::warn_deprecated_environment_variable(
    std::string deprecated, std::string use_instead) {
  std::cerr << "Warning: environment variable '" << deprecated
            << "' is deprecated."
            << " Use '" << use_instead << "' instead."
            << " Raised by Kokkos::initialize()." << std::endl;
}

void Kokkos::Impl::warn_deprecated_command_line_argument(
    std::string deprecated) {
  std::cerr << "Warning: command line argument '" << deprecated
            << "' is deprecated."
            << " Raised by Kokkos::initialize()." << std::endl;
}

void Kokkos::Impl::warn_deprecated_command_line_argument(
    std::string deprecated, std::string use_instead) {
  std::cerr << "Warning: command line argument '" << deprecated
            << "' is deprecated."
            << " Use '" << use_instead << "' instead."
            << " Raised by Kokkos::initialize()." << std::endl;
}

void Kokkos::Impl::do_not_warn_not_recognized_command_line_argument(
    std::regex ignore) {
  do_not_warn_regular_expressions().push_back(std::move(ignore));
}

void Kokkos::Impl::warn_not_recognized_command_line_argument(
    std::string not_recognized) {
  if (check_arg_starts_with(not_recognized.c_str(), "--kokkos-tool")) {
    return;
  }

  for (auto const& ignore : do_not_warn_regular_expressions()) {
    if (std::regex_match(not_recognized, ignore)) {
      return;
    }
  }
  std::cerr << "Warning: command line argument '" << not_recognized
            << "' is not recognized."
            << " Raised by Kokkos::initialize()." << std::endl;
}
