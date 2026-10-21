// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "macros.hpp"

#if defined(__cpp_lib_span)
#include <span>
#endif

#include <cstddef>  // size_t
#include <limits>   // numeric_limits

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
#if defined(__cpp_lib_span)
using std::dynamic_extent;
#else
MDSPAN_IMPL_INLINE_VARIABLE constexpr auto dynamic_extent = std::numeric_limits<size_t>::max();
#endif
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

//==============================================================================================================
