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

#ifndef KOKKOS_CHECK_USAGE_HPP
#define KOKKOS_CHECK_USAGE_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos {

[[nodiscard]] bool is_initialized() noexcept;
[[nodiscard]] bool is_finalized() noexcept;

namespace Impl {

struct UsageRequires {
  struct isInitialized {};
  // Another examples:
  // isFinalized
  // isConstV{}, etc.
};

template <typename... T>
class CheckUsage;

template <>
class CheckUsage<UsageRequires::isInitialized> {
 public:
  static void check() { KOKKOS_EXPECTS_CRITICAL(Kokkos::is_initialized()); }
};

// Another examples
//  template<typename T>
//  class CheckUsage<UsageRequires::isConstV, T>{
//    std::string msg = "A const reduction result type is only allowed for a
//    View, pointer or "
//       "reducer return type!";
//    public:
//    static void check(){
//      KOKKOS_EXPECTS_CRITICAL(!std::is_const_v<int>);
//    }
//  };

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_CHECK_USAGE_HPP