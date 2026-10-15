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

#include "../experimental/__p0009_bits/default_accessor.hpp"
#include "../experimental/__p0009_bits/full_extent_t.hpp"
#include "../experimental/__p0009_bits/mdspan.hpp"
#include "../experimental/__p0009_bits/dynamic_extent.hpp"
#include "../experimental/__p0009_bits/extents.hpp"
#include "../experimental/__p0009_bits/layout_stride.hpp"
#include "../experimental/__p0009_bits/layout_left.hpp"
#include "../experimental/__p0009_bits/layout_right.hpp"
#include "../experimental/__p0009_bits/macros.hpp"
#if MDSPAN_HAS_CXX_17
#include "../experimental/__p2642_bits/layout_padded.hpp"
#include "../experimental/__p2630_bits/submdspan.hpp"
#endif
#include "../experimental/__p2389_bits/dims.hpp"

#endif // MDSPAN_HPP_
