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

#ifndef KOKKOS_IMPL_ERROR_HPP
#define KOKKOS_IMPL_ERROR_HPP

#include <new>  // bad_alloc
#include <string>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Assert.hpp>

namespace Kokkos::Impl {

[[noreturn]] void throw_runtime_exception(const std::string &msg);
[[noreturn]] void throw_bad_alloc(std::string_view memory_space_name,
                                  std::size_t size, std::string label);
void log_warning(const std::string &msg);

}  // namespace Kokkos::Impl

namespace Kokkos::Experimental {

class RawMemoryAllocationFailure : public std::bad_alloc {
  std::string m_msg;

 public:
  explicit RawMemoryAllocationFailure(std::string msg) noexcept
      : m_msg(std::move(msg)) {}

  const char *what() const noexcept override { return m_msg.c_str(); }
};

}  // namespace Kokkos::Experimental

#endif
