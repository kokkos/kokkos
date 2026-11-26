
# `inclusive_scan`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

//
// overload set A
//
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (1)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType inclusive_scan(const std::string& label,                       (2)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto inclusive_scan(const ExecutionSpace& exespace,                               (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (4)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest);

//
// overload set B
//
template <
 class ExecutionSpace, class InputIteratorType,
 class OutputIteratorType, class BinaryOpType
 >
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (5)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType
  >
OutputIteratorType inclusive_scan(const std::string& label,                       (6)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType>
auto inclusive_scan(const ExecutionSpace& exespace,                               (7)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (8)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op);

//
// overload set C
//
template <
 class ExecutionSpace,
 class InputIteratorType, class OutputIteratorType,
 class BinaryOpType, class ValueType
 >
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (9)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op,
                                  ValueType init_value);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class ValueType
  >
OutputIteratorType inclusive_scan(const std::string& label,                       (10)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op,
                                  ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class ValueType>
auto inclusive_scan(const ExecutionSpace& exespace,                               (11)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op,
                    ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class ValueType>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (12)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op,
                    ValueType init_value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- 1,2,3,4: computes an inclusive prefix scan over the range `[first_from, last_from)` (1,2)
or for `view_from` (3,4) using the binary op `bin_op` to combine two elements,
and writes the results to the range beginning at `first_dest` (1,2) or to `view_dest` (3,4).

- 5,6,7,8: computes an inclusive prefix scan over the range `[first_from, last_from)` (5,6)
or `view_from` (7,8) using the binary op `bin_op` to combine two elements, and writes
the results to the range beginning at `first_dest` (5,6) or to `view_dest` (7,8).

- 9,10,11,12: computes an inclusive prefix scan over the range `[first_from, last_from)` (9,10)
or `view_from` (11,12) using the binary functor `bin_op` to combine two elements
and `init` as the initial value, and writes
the results to the range beginning at `first_dest` (9,10) or to `view_dest` (11,12).

Inclusive means that the i-th input element is included in the i-th sum.

## Parameters and Requirements

- `exespace`, `first_from`, `first_last`, `first_dest`, `view_from`, `view_dest`, `bin_op`:
  - same as in [`exclusive_scan`](./StdExclusiveScan)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::inclusive_scan_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::inclusive_scan_view_api_default"

## Return

Iterator to the element *after* the last element written.