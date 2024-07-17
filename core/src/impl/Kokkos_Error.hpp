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

#include <new>  // align_val_t
#include <string>
#include <iosfwd>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Assert.hpp>

namespace Kokkos {
namespace Impl {

[[noreturn]] void throw_runtime_exception(const std::string &msg);

void log_warning(const std::string &msg);

std::string human_memory_size(size_t arg_bytes);

}  // namespace Impl

namespace Experimental {

class RawMemoryAllocationFailure : public std::bad_alloc {
 public:
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
  enum class KOKKOS_DEPRECATED FailureMode {
    OutOfMemoryError,
    AllocationNotAligned,
    InvalidAllocationSize,
    Unknown
  };
#endif

 private:
  std::string m_msg;
  size_t m_attempted_size;
  size_t m_attempted_alignment;

 public:
  RawMemoryAllocationFailure(size_t size, size_t alignment,
                             std::string what) noexcept
      : m_msg(std::move(what)),
        m_attempted_size(size),
        m_attempted_alignment(alignment) {}

  RawMemoryAllocationFailure() noexcept = delete;

  RawMemoryAllocationFailure(RawMemoryAllocationFailure const &) noexcept =
      default;
  RawMemoryAllocationFailure(RawMemoryAllocationFailure &&) noexcept = default;

  RawMemoryAllocationFailure &operator             =(
      RawMemoryAllocationFailure const &) noexcept = default;
  RawMemoryAllocationFailure &operator             =(
      RawMemoryAllocationFailure &&) noexcept = default;

  ~RawMemoryAllocationFailure() noexcept override = default;

  [[nodiscard]] const char *what() const noexcept override {
    return "Memory allocation error";
  }

  [[nodiscard]] size_t attempted_size() const noexcept {
    return m_attempted_size;
  }

  [[nodiscard]] size_t attempted_alignment() const noexcept {
    return m_attempted_alignment;
  }

  void print_error_message(std::ostream &o) const;
  [[nodiscard]] std::string get_error_message() const;
};

}  // namespace Experimental

namespace Impl {

[[noreturn]] void throw_bad_alloc(std::size_t size, std::align_val_t alignment,
                                  std::string what);

}  // namespace Impl
}  // namespace Kokkos

#endif /* #ifndef KOKKOS_IMPL_ERROR_HPP */
