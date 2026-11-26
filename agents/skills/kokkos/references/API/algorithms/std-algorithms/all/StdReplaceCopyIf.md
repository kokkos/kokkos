
# `replace_copy_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <
  class ExecutionSpace,
  class InputIteratorType, class OutputIteratorType,
  class UnaryPredicateType, class T
>
OutputIteratorType replace_copy_if(const ExecutionSpace& exespace,              (1)
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType first_to,
                                   UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class InputIteratorType,  class OutputIteratorType,
  class UnaryPredicateType, class T
>
OutputIteratorType replace_copy_if(const std::string& label,                    (2)
                                   const ExecutionSpace& exespace,
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType first_to,
                                   UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType, class T
>
auto replace_copy_if(const ExecutionSpace& exespace,                            (3)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType, class T
>
auto replace_copy_if(const std::string& label,                                  (4)
                     const ExecutionSpace& exespace,
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     UnaryPredicateType pred, const T& new_value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from range `[first_from, last_from)` to another range
beginning at `first_to` (overloads 1,2) or from `view_from` to `view_to`
(overloads 3,4) replacing with `new_value` all elements for which `pred` returns `true`.


## Parameters and Requirements

- `exespace`, `first_from`, `last_from`, `first_to`, `view_from`, `view_to`, `new_value`:
  - same as in [`replace_copy`](./StdReplaceCopy)
- `label`:
  - for 1, the default string is: "Kokkos::replace_copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_copy_if_view_api_default"
- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
  and must not modify `v`.
  - should have the same API as that shown for [`replace_if`](./StdReplaceIf)


## Return

Iterator to the element *after* the last element copied.
