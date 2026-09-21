// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

#include "macros.hpp"

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

struct full_extent_t { explicit full_extent_t() = default; };

MDSPAN_IMPL_INLINE_VARIABLE constexpr auto full_extent = full_extent_t{ };

} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
