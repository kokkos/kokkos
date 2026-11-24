Std Algorithms
##############

This is a list of basic algorithms provided by Kokkos as parallel replacements for
C++ STL algorithms. Each one has multiple overloads: some take Views directly, and
others take Kokkos iterators (see ./std-algorithms/Iterators). The View versions are
always preferred in application code.
Each algorithm name listed here is both an STL function in ``std::``, and the Kokkos replacement in ``Kokkos::Experimental``.


.. toctree::
   :maxdepth: 1

   `min_element`: ./std-algorithms/all/StdMinElement
   `max_element`: ./std-algorithms/all/StdMaxElement
   `minmax_element`: ./std-algorithms/all/StdMinMaxElement
   `copy`: ./std-algorithms/all/StdCopy
   `copy_if`: ./std-algorithms/all/StdCopyIf
   `copy_n`: ./std-algorithms/all/StdCopy_n
   `copy_backward`: ./std-algorithms/all/StdCopyBackward
   `move`: ./std-algorithms/all/StdMove
   `move_backward`: ./std-algorithms/all/StdMoveBackward
   `fill`: ./std-algorithms/all/StdFill
   `fill_n`: ./std-algorithms/all/StdFill_n
   `transform`: ./std-algorithms/all/StdTransform
   `generate`: ./std-algorithms/all/StdGenerate
   `generate_n`: ./std-algorithms/all/StdGenerate_n
   `remove`: ./std-algorithms/all/StdRemove
   `remove_if`: ./std-algorithms/all/StdRemoveIf
   `remove_copy`: ./std-algorithms/all/StdRemoveCopy
   `remove_copy_if`: ./std-algorithms/all/StdRemoveCopyIf
   `replace`: ./std-algorithms/all/StdReplace
   `replace_if`: ./std-algorithms/all/StdReplaceIf
   `replace_copy`: ./std-algorithms/all/StdReplaceCopy
   `replace_copy_if`: ./std-algorithms/all/StdReplaceCopyIf
   `swap_ranges`: ./std-algorithms/all/StdSwapRanges
   `reverse`: ./std-algorithms/all/StdReverse
   `reverse_copy`: ./std-algorithms/all/StdReverseCopy
   `rotate`: ./std-algorithms/all/StdRotate
   `rotate_copy`: ./std-algorithms/all/StdRotateCopy
   `shift_left`: ./std-algorithms/all/StdShiftLeft
   `shift_right`: ./std-algorithms/all/StdShiftRight
   `unique`: ./std-algorithms/all/StdUnique
   `unique_copy`: ./std-algorithms/all/StdUniqueCopy
   `adjacent_find`: ./std-algorithms/all/StdAdjacentFind
   `count`: ./std-algorithms/all/StdCount
   `count_if`: ./std-algorithms/all/StdCountIf
   `equal`: ./std-algorithms/all/StdEqual
   `all_of`: ./std-algorithms/all/StdAllOf
   `any_of`: ./std-algorithms/all/StdAnyOf
   `none_of`: ./std-algorithms/all/StdNoneOf
   `find`: ./std-algorithms/all/StdFind
   `find_if`: ./std-algorithms/all/StdFindIf
   `find_if_not`: ./std-algorithms/all/StdFindIfNot
   `find_end`: ./std-algorithms/all/StdFindEnd
   `find_first_of`: ./std-algorithms/all/StdFindFirstOf
   `for_each`: ./std-algorithms/all/StdForEach
   `for_each_n`: ./std-algorithms/all/StdForEachN
   `lexicographical_compare`: ./std-algorithms/all/StdLexicographicalCompare
   `mismatch`: ./std-algorithms/all/StdMismatch
   `search`: ./std-algorithms/all/StdSearch
   `search_n`: ./std-algorithms/all/StdSearchN
   `adjacent_difference`: ./std-algorithms/all/StdAdjacentDifference
   `reduce`: ./std-algorithms/all/StdReduce
   `exclusive_scan`: ./std-algorithms/all/StdExclusiveScan
   `inclusive_scan`: ./std-algorithms/all/StdInclusiveScan
   `transform_reduce`: ./std-algorithms/all/StdTransformReduce
   `transform_exclusive_scan`: ./std-algorithms/all/StdTransformExclusiveScan
   `transform_inclusive_scan`: ./std-algorithms/all/StdTransformInclusiveScan
   `is_partitioned`: ./std-algorithms/all/StdIsPartitioned
   `partition_point`: ./std-algorithms/all/StdPartitionPoint
   `partition_copy`: ./std-algorithms/all/StdPartitionCopy
   `is_sorted`: ./std-algorithms/all/StdIsSorted
   `is_sorted_until`: ./std-algorithms/all/StdIsSortedUntil
