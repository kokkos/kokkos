//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_TYPE_INFO_HPP
#define KOKKOS_TYPE_INFO_HPP

#include <array>
#include <string_view>
#include <utility>

#include <Kokkos_Macros.hpp>

#ifndef KOKKOS_COMPILER_INTEL

namespace Kokkos {
namespace Impl {

template <std::size_t... Is>
constexpr auto string_view_to_char_array(std::string_view s,
                                         std::index_sequence<Is...>) {
  return std::array{s[Is]...};
}

template <class T>
constexpr auto type_name() {
#if defined(__clang__)
  constexpr std::string_view func = __PRETTY_FUNCTION__;
  constexpr std::string_view prefix{"[T = "};
  constexpr std::string_view suffix{"]"};
#elif defined(__GNUC__)
  constexpr std::string_view func = __PRETTY_FUNCTION__;
  constexpr std::string_view prefix{"[with T = "};
  constexpr std::string_view suffix{"]"};
#elif defined(_MSC_VER)
  constexpr std::string_view func = __FUNCSIG__;
  constexpr std::string_view prefix{"type_name<"};
  constexpr std::string_view suffix{">(void)"};
#else
#error bug
#endif
  constexpr auto beg = func.find(prefix) + prefix.size();
  constexpr auto end = func.rfind(suffix);
  static_assert(beg != std::string_view::npos);
  static_assert(end != std::string_view::npos);
  static_assert(beg < end);
  constexpr auto name = func.substr(beg, end - beg);
  return string_view_to_char_array(name,
                                   std::make_index_sequence<name.size()>());
}

}  // namespace Impl

template <class T>
class TypeInfo {
  static constexpr auto value_ = Impl::type_name<T>();

 public:
  static constexpr std::string_view name() noexcept {
    return {value_.data(), value_.size()};
  }
};

}  // namespace Kokkos

#else  // out of luck, using Intel C++ Compiler Classic

namespace Kokkos {

template <class T>
class TypeInfo {
 public:
  static constexpr std::string_view name() noexcept { return "not supported"; }
};

}  // namespace Kokkos

#endif

#endif
