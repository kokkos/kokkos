
# `reduce`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

//
// overload set A
//
template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const ExecutionSpace& exespace,        (1)
                                         IteratorType first,
                                         IteratorType last);

template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const std::string& label,              (2)
                                         const ExecutionSpace& exespace,
                                         IteratorType first,
                                         IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const ExecutionSpace& exespace,                                     (3)
            const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const std::string& label, const ExecutionSpace& exespace,           (4)
            const ::Kokkos::View<DataType, Properties...>& view);

//
// overload set B
//
template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const ExecutionSpace& exespace,                                (5)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (6)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class DataType, class... Properties, class ValueType>
ValueType reduce(const ExecutionSpace& exespace,                                (7)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class DataType, class... Properties, class ValueType>
ValueType reduce(const std::string& label,                                      (8)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value);

//
// overload set C
//
template <
  class ExecutionSpace, class IteratorType, class ValueType,
  class BinaryOp>
ValueType reduce(const ExecutionSpace& exespace,                                (9)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class IteratorType, class ValueType,
  class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (10)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class DataType, class... Properties,
  class ValueType, class BinaryOp>
ValueType reduce(const ExecutionSpace& exespace,                                (11)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class DataType, class... Properties,
  class ValueType, class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (12)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overload set A (1,2,3,4): performs a reduction of the elements
  in the range `[first, last)` (1,2) or in `view` (3,4).

- Overload set B (5,6,7,8): performs a reduction of the elements
  in the range `[first, last)` (5,6) or in `view` (7,8) accounting for
  the initial value `init_reduction_value`.

- Overload set C (9,10,11,12): performs a reduction of the elements
  in the range `[first, last)` (9,10)
  or in `view` (11,12) accounting for the initial value `init_reduction_value`
  using the functor  `joiner` to join operands during the reduction operation.


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::reduce_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::reduce_view_api_default"
- `first`, `last`:
  - range of elements to reduce over
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to reduce
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `init_reduction_value`:
  - initial reduction value to use
- `joiner`:
  - *binary* functor performing the desired operation to join two elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `ValueType`, and must not modify `a,b`.
  - Must conform to:
  ```c++
  struct JoinFunctor {
	KOKKOS_FUNCTION
	constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
	  return /* ... */
	}
  };
  ```
  - The behavior is non-deterministic if the `joiner` operation
  is not associative or not commutative.


## Return

The reduction result.
