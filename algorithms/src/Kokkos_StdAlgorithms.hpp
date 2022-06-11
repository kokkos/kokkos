/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_STD_ALGORITHMS_HPP
#define KOKKOS_STD_ALGORITHMS_HPP

/// \file Kokkos_StdAlgorithms.hpp
/// \brief Kokkos counterparts for Standard C++ Library algorithms

#include <std_algorithms/impl/Kokkos_Constraints.hpp>
#include <std_algorithms/impl/Kokkos_RandomAccessIterator.hpp>
#include <std_algorithms/Kokkos_BeginEnd.hpp>

// distance
#include <std_algorithms/Kokkos_Distance.hpp>

// note that we categorize below the headers
// following the std classification.

// modifying ops
#include "std_algorithms/Kokkos_swap.hpp"
#include "std_algorithms/Kokkos_iter_swap.hpp"

// non-modifying sequence
#include "std_algorithms/Kokkos_adjacent_find.hpp"
#include "std_algorithms/Kokkos_count.hpp"
#include "std_algorithms/Kokkos_count_if.hpp"
#include "std_algorithms/Kokkos_all_of.hpp"
#include "std_algorithms/Kokkos_any_of.hpp"
#include "std_algorithms/Kokkos_none_of.hpp"
#include "std_algorithms/Kokkos_equal.hpp"
#include "std_algorithms/Kokkos_find.hpp"
#include "std_algorithms/Kokkos_find_if.hpp"
#include "std_algorithms/Kokkos_find_if_not.hpp"
#include "std_algorithms/Kokkos_find_end.hpp"
#include "std_algorithms/Kokkos_find_first_of.hpp"
#include "std_algorithms/Kokkos_for_each.hpp"
#include "std_algorithms/Kokkos_for_each_n.hpp"
#include "std_algorithms/Kokkos_lexicographical_compare.hpp"
#include "std_algorithms/Kokkos_mismatch.hpp"
#include "std_algorithms/Kokkos_search.hpp"
#include "std_algorithms/Kokkos_search_n.hpp"

// modifying sequence
#include "std_algorithms/Kokkos_fill.hpp"
#include "std_algorithms/Kokkos_fill_n.hpp"
#include "std_algorithms/Kokkos_replace.hpp"
#include "std_algorithms/Kokkos_replace_if.hpp"
#include "std_algorithms/Kokkos_replace_copy_if.hpp"
#include "std_algorithms/Kokkos_replace_copy.hpp"
#include "std_algorithms/Kokkos_copy.hpp"
#include "std_algorithms/Kokkos_copy_n.hpp"
#include "std_algorithms/Kokkos_copy_backward.hpp"
#include "std_algorithms/Kokkos_copy_if.hpp"
#include "std_algorithms/Kokkos_transform.hpp"
#include "std_algorithms/Kokkos_generate.hpp"
#include "std_algorithms/Kokkos_generate_n.hpp"
#include "std_algorithms/Kokkos_reverse.hpp"
#include "std_algorithms/Kokkos_reverse_copy.hpp"
#include "std_algorithms/Kokkos_move.hpp"
#include "std_algorithms/Kokkos_move_backward.hpp"
#include "std_algorithms/Kokkos_swap_ranges.hpp"
#include "std_algorithms/Kokkos_unique.hpp"
#include "std_algorithms/Kokkos_unique_copy.hpp"
#include "std_algorithms/Kokkos_rotate.hpp"
#include "std_algorithms/Kokkos_rotate_copy.hpp"
#include "std_algorithms/Kokkos_remove.hpp"
#include "std_algorithms/Kokkos_remove_if.hpp"
#include "std_algorithms/Kokkos_remove_copy.hpp"
#include "std_algorithms/Kokkos_remove_copy_if.hpp"
#include "std_algorithms/Kokkos_shift_left.hpp"
#include "std_algorithms/Kokkos_shift_right.hpp"

// sorting ops
#include "std_algorithms/Kokkos_is_sorted_until.hpp"
#include "std_algorithms/Kokkos_is_sorted.hpp"

// min/max element
#include "std_algorithms/Kokkos_min_element.hpp"
#include "std_algorithms/Kokkos_max_element.hpp"
#include "std_algorithms/Kokkos_minmax_element.hpp"

// partitioning ops
#include "std_algorithms/Kokkos_is_partitioned.hpp"
#include "std_algorithms/Kokkos_partition_copy.hpp"
#include "std_algorithms/Kokkos_partition_point.hpp"

// numeric
#include "std_algorithms/Kokkos_adjacent_difference.hpp"
#include "std_algorithms/Kokkos_reduce.hpp"
#include "std_algorithms/Kokkos_transform_reduce.hpp"
#include "std_algorithms/Kokkos_exclusive_scan.hpp"
#include "std_algorithms/Kokkos_transform_exclusive_scan.hpp"
#include "std_algorithms/Kokkos_inclusive_scan.hpp"
#include "std_algorithms/Kokkos_transform_inclusive_scan.hpp"

#endif
