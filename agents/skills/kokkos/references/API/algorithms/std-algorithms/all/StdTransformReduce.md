
# `transform_reduce`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

//
// overload set A
//
template <
  class ExecutionSpace, class IteratorType1,
  class IteratorType2, class ValueType>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (1)
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace, class IteratorType1,
  class IteratorType2, class ValueType>
ValueType transform_reduce(const std::string& label,                            (2)
                           const ExecutionSpace& exespace,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (3)
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
ValueType transform_reduce(const std::string& label,                            (4)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value);

//
// overload set B
//
template <
  class ExecutionSpace,
  class IteratorType1, class IteratorType2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (5)
                           IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
  class ExecutionSpace,
  class IteratorType1, class IteratorType2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label,                            (6)
                           const ExecutionSpace& exespace,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (7)
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
   class ExecutionSpace,
   class DataType1, class... Properties1,
   class DataType2, class... Properties2,
   class ValueType,
   class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label,                            (8)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

//
// overload set C
//
template <
  class ExecutionSpace,
  class IteratorType, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (9)
                           IteratorType first1, IteratorType last1,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class IteratorType, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label,
                           const ExecutionSpace& exespace,                      (10)
                           IteratorType first1, IteratorType last1,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (11)
                           const ::Kokkos::View<DataType, Properties...>& first_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label,                            (12)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType, Properties...>& first_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2): performs the *product* (via `operator*`) between each pair
  of elements in the range `[first1, last1)` and the range starting at `first2`,
  and reduces the results along with the initial value `init_reduction_value`

- (3,4): performs the *product* (via `operator*`) between each pair
  of elements in `first_view` and `second_view`,
  and reduces the results along with the initial value `init_reduction_value`

- (5,6): applies the functor `binary_transformer` to each pair of elements
  in the range `[first1, last1)` and the range starting at `first2`,
  and reduces the results along with the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (7,8): applies the functor `binary_transformer` to each pair of elements
  in `first_view` and `second_view`,
  and reduces the result along with the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (9,10): applies the functor `unary_transformer` to each element
  in the range `[first, last)`, and reduces the results along with
  the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (11,12): applies the functor `unary_transformer` to each element
  in `view`, and reduces the results along with
  the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::transform_reduce_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::transform_reduce_view_api_default"
- `first1`, `last1`, `first2`:
  - ranges of elements to transform and reduce
  - must be *random access iterators*
  - must represent a valid range, i.e., `last1 >= first1` (checked in debug mode)
  - must be accessible from `exespace`
- `first_view`, `second_view`:
  - views to transform and reduce
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
	constexpr ValueType operator()(const ValueType& a,
                                 const ValueType& b) const {
	  return /* ... */
	}
  };
  ```
  - The behavior is non-deterministic if the `joiner` operation
  is not associative or not commutative.

- `binary_transformer`:
  - *binary* functor applied to each pair of elements *before* doing the reduction.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `value_type_a` and `value_type_b`,
  where `value_type_{a,b}` are the value types of `first1` and `first2` (for 1,2,5,6)
  or the value types of `first_view` and `second_view` (for 3,4,7,8), and must not modify `a,b`.
  - Must conform to:
  ```c++
  struct BinaryTransformer {
	KOKKOS_FUNCTION
	constexpr return_type operator()(const value_type_a & a, const value_type_b & b) const {
	  return /* ... */
	}
  };
  ```
  - the `return_type` is such that it can be accepted by the `joiner`

- `unary_transformer`:
  - *unary* functor performing the desired operation to an element.
  Must be valid to be called from the execution space passed, and callable with
  an arguments `v` of type (possible const) `value_type`,
  where `value_type` is the value type of `first1` (for 9,10)
  or the value type of `first_view` (for 11,12), and must not modify `v`.
  - Must conform to:
  ```c++
  struct UnaryTransformer {
	KOKKOS_FUNCTION
	constexpr value_type operator()(const value_type & v) const {
	  return /* ... */
	}
  };
  ```

## Return

The reduction result.