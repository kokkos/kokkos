
# `search`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 search(const ExecutionSpace& exespace, IteratorType1 first,
                     IteratorType1 last, IteratorType2 s_first,                      (1)
                     IteratorType2 s_last);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 search(const std::string& label, const ExecutionSpace& exespace,
                     IteratorType1 first, IteratorType1 last,                        (2)
                     IteratorType2 s_first, IteratorType2 s_last);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto search(const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType1, Properties1...>& view,                   (3)
            const ::Kokkos::View<DataType2, Properties2...>& s_view);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto search(const std::string& label, const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType1, Properties1...>& view,                   (4)
            const ::Kokkos::View<DataType2, Properties2...>& s_view);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 search(const ExecutionSpace& exespace, IteratorType1 first,                  (5)
                     IteratorType1 last, IteratorType2 s_first,
                     IteratorType2 s_last, const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 search(const std::string& label, const ExecutionSpace& exespace,
                     IteratorType1 first, IteratorType1 last,                        (6)
                     IteratorType2 s_first, IteratorType2 s_last,
                     const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto search(const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType1, Properties1...>& view,                   (7)
            const ::Kokkos::View<DataType2, Properties2...>& s_view,
            const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto search(const std::string& label, const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType1, Properties1...>& view,                   (8)
            const ::Kokkos::View<DataType2, Properties2...>& s_view,
            const BinaryPredicateType& pred)

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Searches for the first occurrence of the sequence of elements `[s_first, s_last)` in the range `[first, last)` in (1,2,5,6).
Searches for the first occurrence of the sequence of elements `s_view` in `view` in (3,4,7,8).
Elements in (1,2,3,4) are compared using `==` and elements in (5,6,7,8) are compared using `pred`.

## Parameters and Requirements

- `exespace`, `s_first`, `s_last`, `first`, `last`, `s_view` and `view` similar to [`mismatch`](./StdMismatch).

- `label`:
    - 1,5: The default string is "Kokkos::search_iterator_api_default".
    - 3,7: The default string is "Kokkos::search_view_api_default".

- `pred` - similar to [`equal`](./StdEqual)
