// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#pragma once

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

template< ::std::size_t Rank, class IndexType = std::size_t>
using dims =
  :: MDSPAN_IMPL_STANDARD_NAMESPACE :: dextents<IndexType, Rank>;

} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
