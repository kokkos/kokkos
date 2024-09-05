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

template <size_t N>
constexpr std::array<char, N> to_array(std::string_view src) {
  std::array<char, N> dst{};
  for (size_t i = 0; i < N; ++i) {
    dst[i] = src[i];
  }
  return dst;
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
  return to_array<end - beg>(func.substr(beg, end));
}

}  // namespace Impl

template <class T>
class TypeInfo {
  static constexpr auto value_ = Impl::type_name<T>();

 public:
  static constexpr std::string_view name() noexcept {
    return {value_.data(), value_.size()};
  }
  TypeInfo() = delete;
};

}  // namespace Kokkos

#else  // out of luck, using Intel C++ Compiler Classic

namespace Kokkos {

template <class T>
class TypeInfo {
 public:
  static constexpr std::string_view name() noexcept { return "not supported"; }
  TypeInfo() = delete;
};

}  // namespace Kokkos

#endif

#endif