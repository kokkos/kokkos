
# `exclusive_scan`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

//
// overload set A
//
template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,                (1)
                                  InputIteratorType first,
                                  InputIteratorType last,
                                  OutputIteratorType first_dest,
                                  ValueType init_value);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan(const std::string& label,                      (2)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first,
                                  InputIteratorType last,
                                  OutputIteratorType first_dest,
                                  ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto exclusive_scan(const ExecutionSpace& exespace,                              (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto exclusive_scan(const std::string& label, const ExecutionSpace& exespace,    (4)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value);

//
// overload set B
//
template <
 class ExecutionSpace, class InputIteratorType,
 class OutputIteratorType, class ValueType, class BinaryOpType
 >
OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,                (5)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  ValueType init_value, BinaryOpType bin_op);


template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType, class BinaryOpType
  >
OutputIteratorType exclusive_scan(const std::string& label,                      (6)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  ValueType init_value, BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class ValueType,
  class BinaryOpType>
auto exclusive_scan(const ExecutionSpace& exespace,                              (7)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value, BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class ValueType,
  class BinaryOpType>
auto exclusive_scan(const std::string& label, const ExecutionSpace& exespace,    (8)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value, BinaryOpType bin_op);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- 1,2,3,4: computes an exclusive prefix *sum* for the range `[first_from, last_from)` (1,2)
or `view_from` (3,4), using `init` as the initial value, and writes
the results to the range beginning at `first_dest` (1,2) or to `view_dest` (3,4).

- 5,6,7,8: computes an exclusive prefix scan using the binary functor `bin_op`
to combine two elements for the range `[first_from, last_from)` (5,6)
or `view_from` (7,8), using `init` as the initial value, and writes
the results to the range beginning at `first_dest` (5,6) or to `view_dest` (7,8).

Exclusive means that the i-th input element is not included in the i-th sum.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,2 the default string is: "Kokkos::exclusive_scan_iterator_api_default"
  - for 5,6 the default string is: "Kokkos::exclusive_scan_view_api_default"
- `first_from`, `last_from`, `first_dest`:
  - range of elements to read from (`*_from`) and write to (`first_dest`)
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `view_from`, `view_dest`:
  - views to read elements from `view_from` and write to `view_dest`
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `bin_op`:
  - *binary* functor representing the operation to combine pair of elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2,5,6) or the value type
  of `view_from` (for 3,4,7,8), and must not modify `a,b`.
  - must conform to:
  ```c++
  struct BinaryOp
  {
     KOKKOS_INLINE_FUNCTION
     return_type operator()(const value_type & a,
	                        const value_type & b) const {
       return /* ... */;
     }
  };
  ```
  The return type `return_type` must be such that an object of type `OutputIteratorType`
  for (1,2,5,6) or an object of type `value_type` where `value_type` is the
  value type of `view_dest` for (3,4,7,8) can be dereferenced and assigned a value of type `return_type`.

## Return

Iterator to the element *after* the last element written.