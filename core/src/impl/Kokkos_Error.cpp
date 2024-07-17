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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <cstring>
#include <cstdlib>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <Kokkos_Core.hpp>  // show_warnings
#include <impl/Kokkos_Error.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

void Kokkos::Impl::throw_bad_alloc(
    std::size_t size, std::align_val_t alignment,
    Kokkos::Experimental::RawMemoryAllocationFailure::FailureMode failure_mode,
    std::string what) {
  throw Kokkos::Experimental::RawMemoryAllocationFailure(
      size, static_cast<size_t>(alignment), failure_mode, std::move(what));
}

namespace Kokkos {
namespace Impl {

void throw_runtime_exception(const std::string &msg) {
  throw std::runtime_error(msg);
}

void log_warning(const std::string &msg) {
  if (show_warnings()) {
    std::cerr << msg << std::flush;
  }
}

std::string human_memory_size(size_t arg_bytes) {
  double bytes   = arg_bytes;
  const double K = 1024;
  const double M = K * 1024;
  const double G = M * 1024;

  std::ostringstream out;
  if (bytes < K) {
    out << std::setprecision(4) << bytes << " B";
  } else if (bytes < M) {
    bytes /= K;
    out << std::setprecision(4) << bytes << " K";
  } else if (bytes < G) {
    bytes /= M;
    out << std::setprecision(4) << bytes << " M";
  } else {
    bytes /= G;
    out << std::setprecision(4) << bytes << " G";
  }
  return out.str();
}

}  // namespace Impl

void Experimental::RawMemoryAllocationFailure::print_error_message(
    std::ostream &o) const {
  o << "Allocation of size "
    << ::Kokkos::Impl::human_memory_size(m_attempted_size);
  o << " failed";
  switch (m_failure_mode) {
    case FailureMode::OutOfMemoryError:
      o << ", likely due to insufficient memory.";
      break;
    case FailureMode::AllocationNotAligned:
      o << " because the allocation was improperly aligned.";
      break;
    case FailureMode::InvalidAllocationSize:
      o << " because the requested allocation size is not a valid size for the"
           " requested allocation mechanism (it's probably too large).";
      break;
    case FailureMode::Unknown: o << " because of an unknown error."; break;
  }
  o << "  (The allocation mechanism was " << m_msg << ")\n";
}

std::string Experimental::RawMemoryAllocationFailure::get_error_message()
    const {
  std::ostringstream out;
  print_error_message(out);
  return out.str();
}

}  // namespace Kokkos
