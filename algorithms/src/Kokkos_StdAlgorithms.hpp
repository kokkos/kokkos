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

#ifndef KOKKOS_STD_ALGORITHMS_HPP
#define KOKKOS_STD_ALGORITHMS_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_STD_ALGORITHMS
#endif

/// \file Kokkos_StdAlgorithms.hpp
/// \brief Kokkos counterparts for Standard C++ Library algorithms

#include "std_algorithms/impl/Kokkos_Constraints.hpp"  // IWYU pragma: export
#include "std_algorithms/impl/Kokkos_RandomAccessIterator.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_BeginEnd.hpp"  // IWYU pragma: export

// distance
#include "std_algorithms/Kokkos_Distance.hpp"  // IWYU pragma: export

// note that we categorize below the headers
// following the std classification.

// modifying ops
#include "std_algorithms/Kokkos_IterSwap.hpp"  // IWYU pragma: export

// non-modifying sequence
#include "std_algorithms/Kokkos_AdjacentFind.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_Count.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_CountIf.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_AllOf.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_AnyOf.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_NoneOf.hpp"        // IWYU pragma: export
#include "std_algorithms/Kokkos_Equal.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_Find.hpp"          // IWYU pragma: export
#include "std_algorithms/Kokkos_FindIf.hpp"        // IWYU pragma: export
#include "std_algorithms/Kokkos_FindIfNot.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_FindEnd.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_FindFirstOf.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_ForEach.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_ForEachN.hpp"      // IWYU pragma: export
#include "std_algorithms/Kokkos_LexicographicalCompare.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_Mismatch.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_Search.hpp"    // IWYU pragma: export
#include "std_algorithms/Kokkos_SearchN.hpp"   // IWYU pragma: export

// modifying sequence
#include "std_algorithms/Kokkos_Fill.hpp"           // IWYU pragma: export
#include "std_algorithms/Kokkos_FillN.hpp"          // IWYU pragma: export
#include "std_algorithms/Kokkos_Replace.hpp"        // IWYU pragma: export
#include "std_algorithms/Kokkos_ReplaceIf.hpp"      // IWYU pragma: export
#include "std_algorithms/Kokkos_ReplaceCopyIf.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_ReplaceCopy.hpp"    // IWYU pragma: export
#include "std_algorithms/Kokkos_Copy.hpp"           // IWYU pragma: export
#include "std_algorithms/Kokkos_CopyN.hpp"          // IWYU pragma: export
#include "std_algorithms/Kokkos_CopyBackward.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_CopyIf.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_Transform.hpp"      // IWYU pragma: export
#include "std_algorithms/Kokkos_Generate.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_GenerateN.hpp"      // IWYU pragma: export
#include "std_algorithms/Kokkos_Reverse.hpp"        // IWYU pragma: export
#include "std_algorithms/Kokkos_ReverseCopy.hpp"    // IWYU pragma: export
#include "std_algorithms/Kokkos_Move.hpp"           // IWYU pragma: export
#include "std_algorithms/Kokkos_MoveBackward.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_SwapRanges.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_Unique.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_UniqueCopy.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_Rotate.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_RotateCopy.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_Remove.hpp"         // IWYU pragma: export
#include "std_algorithms/Kokkos_RemoveIf.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_RemoveCopy.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_RemoveCopyIf.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_ShiftLeft.hpp"      // IWYU pragma: export
#include "std_algorithms/Kokkos_ShiftRight.hpp"     // IWYU pragma: export

// sorting
#include "std_algorithms/Kokkos_IsSortedUntil.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_IsSorted.hpp"       // IWYU pragma: export

// min/max element
#include "std_algorithms/Kokkos_MinElement.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_MaxElement.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_MinMaxElement.hpp"  // IWYU pragma: export

// partitioning
#include "std_algorithms/Kokkos_IsPartitioned.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_PartitionCopy.hpp"   // IWYU pragma: export
#include "std_algorithms/Kokkos_PartitionPoint.hpp"  // IWYU pragma: export

// numeric
#include "std_algorithms/Kokkos_AdjacentDifference.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_Reduce.hpp"              // IWYU pragma: export
#include "std_algorithms/Kokkos_TransformReduce.hpp"     // IWYU pragma: export
#include "std_algorithms/Kokkos_ExclusiveScan.hpp"       // IWYU pragma: export
#include "std_algorithms/Kokkos_TransformExclusiveScan.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_InclusiveScan.hpp"  // IWYU pragma: export
#include "std_algorithms/Kokkos_TransformInclusiveScan.hpp"  // IWYU pragma: export

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_STD_ALGORITHMS
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_STD_ALGORITHMS
#endif
#endif
