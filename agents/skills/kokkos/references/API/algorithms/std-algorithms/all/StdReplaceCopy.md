
# `replace_copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
OutputIteratorType replace_copy(const ExecutionSpace& exespace,               (1)
                                InputIteratorType first_from,
                                InputIteratorType last_from,
                                OutputIteratorType first_to,
                                const T& old_value, const T& new_value);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
OutputIteratorType replace_copy(const std::string& label,                     (2)
                                const ExecutionSpace& exespace,
                                OutputIteratorType first_to,
                                const T& old_value, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class T
>
auto replace_copy(const ExecutionSpace& exespace,                             (3)
                  const Kokkos::View<DataType1, Properties1...>& view_from,
                  const Kokkos::View<DataType2, Properties2...>& view_to,
                  const T& old_value, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class T
>
auto replace_copy(const std::string& label,
                  const ExecutionSpace& exespace,                             (4)
                  const Kokkos::View<DataType1, Properties1...>& view_from,
                  const Kokkos::View<DataType2, Properties2...>& view_to,
                  const T& old_value, const T& new_value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from range `[first_from, last_from)` to another range
beginning at `first_to` (overloads 1,2) or from `view_from` to `view_to`
(overloads 3,4) replacing with `new_value` all elements that equal `old_value`.
Comparison between elements is done using `operator==`.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::replace_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `old_value`, `new_value`:
  - self-explanatory


## Return

Iterator to the element *after* the last element copied.
