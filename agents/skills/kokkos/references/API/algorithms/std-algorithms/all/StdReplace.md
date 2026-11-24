
# `replace`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class T>
void replace(const ExecutionSpace& exespace,                                 (1)
             IteratorType first, IteratorType last,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class IteratorType, class T>
void replace(const std::string& label, const ExecutionSpace& exespace,       (2)
             IteratorType first, IteratorType last,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void replace(const ExecutionSpace& exespace,                                 (3)
             const Kokkos::View<DataType, Properties...>& view,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void replace(const std::string& label, const ExecutionSpace& exespace,       (4)
             const Kokkos::View<DataType, Properties...>& view,
             const T& old_value, const T& new_value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Replaces with `new_value` all elements that are equal to `old_value` in the
range `[first, last)` (overloads 1,2) or in `view` (overloads 3,4).
Equality is checked using `operator==`.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::replace_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_view_api_default"
- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., `Kokkos::Experimental::begin/end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `old_value`, `new_value`:
  - self-explanatory


## Return

None


## Example

```c++
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// do something with a
// ...

const double oldValue{2};
const double newValue{34};
KE::replace(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a), oldValue, newValue);

// explicitly set label and execution space (assuming active)
KE::replace("mylabel", Kokkos::OpenMP(), a, oldValue, newValue);
```
