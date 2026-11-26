
# `partition_point`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point(const ExecutionSpace& exespace,                   (1)
                             IteratorType first, IteratorType last,
                             PredicateType pred);

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point(const std::string& label,                         (2)
                             const ExecutionSpace& exespace,
                             IteratorType first, IteratorType last,
                             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto partition_point(const ExecutionSpace& exespace,
                     const ::Kokkos::View<DataType, Properties...>& view,      (3)
                     PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto partition_point(const std::string& label,                                 (4)
                     const ExecutionSpace& exespace,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     PredicateType pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Examines the range `[first, last)` or `view` and locates the
first element that does not satisfy `pred`.

Assumes the range (or the view) already to be partitioned.


## Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `pred`: same as in [`is_partitioned`](./StdIsPartitioned)
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::partition_point_iterator_api_default"
  - for 3, the default string is: "Kokkos::partition_point_view_api_default"

## Return

Iterator to the element *after* the last element in the first partition,
or `last` if all elements satisfy `pred`.