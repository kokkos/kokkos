
# `search_n`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType>
IteratorType search_n(const ExecutionSpace& exespace, IteratorType first,
                      IteratorType last, SizeType count,                             (1)
                      const ValueType& value);

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType>
IteratorType search_n(const std::string& label, const ExecutionSpace& exespace,
                      IteratorType first, IteratorType last, SizeType count,         (2)
                      const ValueType& value);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType>
auto search_n(const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (3)
              SizeType count, const ValueType& value);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType>
auto search_n(const std::string& label, const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (4)
              SizeType count, const ValueType& value);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
IteratorType search_n(const ExecutionSpace& exespace, IteratorType first,
                      IteratorType last, SizeType count, const ValueType& value,     (5)
                      const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
IteratorType search_n(const std::string& label, const ExecutionSpace& exespace,
                      IteratorType first, IteratorType last, SizeType count,         (6)
                      const ValueType& value, const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType, class BinaryPredicateType>
auto search_n(const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (7)
              SizeType count, const ValueType& value,
              const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType, class BinaryPredicateType>
auto search_n(const std::string& label, const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (8)
              SizeType count, const ValueType& value,
              const BinaryPredicateType& pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Searches the range [first, last) for a range of `count` elements each comparing equal to `value`  (1,2).
Searches the `view` for `count` elements each comparing equal to `value`  (3,4).
Searches the range [first, last) for a range of `count` elements for which the `pred` returns true for `value` in (5,6).
Searches the `view` for a range of `count` elements for which the `pred` returns true for `value` in (7,8).

## Parameters and Requirements

- `exespace`, `first`, `last`, `view` and `count` similar to [`for_each_n`](./StdForEachN).

- `label`:
    - 1,5: The default string is "Kokkos::search_n_iterator_api_default".
    - 3,7: The default string is "Kokkos::search_n_view_api_default".

- `pred` - similar to [`equal`](./StdEqual)
