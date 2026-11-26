
# `for_each_n`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const ExecutionSpace& exespace,
                      InputIterator first, SizeType n,
                      UnaryFunctorType functor);                                     (1)

template <class ExecutionSpace, class InputIterator, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const std::string& label, const ExecutionSpace& exespace,
                      InputIterator first, SizeType n
                      UnaryFunctorType functor);                                     (2)

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view, SizeType n,
             UnaryFunctorType functor);                                              (3)

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view, SizeType n,
             UnaryFunctorType func);                                                 (4)

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Applies the UnaryFunctorType `func` to the result of dereferencing each iterator in `[first,first+n]` for (1,2) and in (3,4) the functor is applied to the first `n` elements of the view.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

## Parameters and Requirements

- `exespace`, `first`, `view`, `func` : same as in [`for_each`](./StdForEach)

- `label`:
  - for 1, the default string is: "Kokkos::for_each_n_iterator_api_default"
  - for 3, the default string is: "Kokkos::for_each_n_view_api_default"

- `n`:
  - number of elements to operate on

## Return

`func`
