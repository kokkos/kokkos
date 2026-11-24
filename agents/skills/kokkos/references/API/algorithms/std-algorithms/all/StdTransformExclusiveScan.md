# `transform_exclusive_scan`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType,
  class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_exclusive_scan(const ExecutionSpace& exespace,     (1)
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            ValueType init_value,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType,
  class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_exclusive_scan(const std::string& label,           (2)
                                            const ExecutionSpace& exespace,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            ValueType init_value,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType, class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(const ExecutionSpace& exespace,                   (3)
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              ValueType init_value, BinaryOpType binary_op,
                              UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType, class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(const std::string& label,                         (4)
                              const ExecutionSpace& exespace,
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              ValueType init_value, BinaryOpType binary_op,
                              UnaryOpType unary_op);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- 1,2: transforms each element in the range `[first_from, last_from)`
with `unary_op`, then computes an exclusive prefix scan operation using `binary_op`
over the resulting range, with `init` as the initial value, and writes
the results to the range beginning at `first_dest`.
"exclusive" means that the i-th input element is not included in the i-th sum

- 3,4: same as (1,2) except that the elements are read from `view_from`
and written to `view_dest`

## Parameters and Requirements

- `exespace`, `first_from`, `first_last`, `first_dest`, `view_from`, `view_dest`:
  - same as [`exclusive_scan`](./StdExclusiveScan)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1 the default string is: "Kokkos::transform_exclusive_scan_iterator_api_default"
  - for 3 the default string is: "Kokkos::transform_exclusive_scan_view_api_default"
- `unary_op`:
  - *unary* functor performing the desired transformation operation to an element.
  Must be valid to be called from the execution space passed, and callable with
  an arguments `v` of type (possible const) `value_type`,
  where `value_type` is the value type of `first_from` (for 1,2)
  or the value type of `view_from` (for 3,4), and must not modify `v`.
  - Must conform to:
  ```c++
  struct UnaryOp {
	KOKKOS_FUNCTION
	constexpr value_type operator()(const value_type & v) const {
	  return /* ... */
	}
  };
  ```

## Return

Iterator to the element *after* the last element written.
