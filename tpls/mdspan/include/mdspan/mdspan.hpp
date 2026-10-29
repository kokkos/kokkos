// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef MDSPAN_HPP_
#define MDSPAN_HPP_

#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#define MDSPAN_IMPL_STANDARD_NAMESPACE Kokkos
#endif

#ifndef MDSPAN_IMPL_PROPOSED_NAMESPACE
#define MDSPAN_IMPL_PROPOSED_NAMESPACE Experimental
#endif

#include "impl/default_accessor.hpp"
#include "impl/full_extent_t.hpp"
#include "impl/mdspan.hpp"
#include "impl/dynamic_extent.hpp"
#include "impl/extents.hpp"
#include "impl/layout_stride.hpp"
#include "impl/layout_left.hpp"
#include "impl/layout_right.hpp"
#include "impl/macros.hpp"
#if MDSPAN_HAS_CXX_17
#include "impl/layout_padded.hpp"
#include "impl/submdspan.hpp"
#endif
#include "impl/dims.hpp"

#endif  // MDSPAN_HPP_
