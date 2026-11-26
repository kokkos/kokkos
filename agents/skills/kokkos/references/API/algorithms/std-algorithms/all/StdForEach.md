
# `for_each`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
void for_each(const ExecutionSpace& exespace,                            (1)
              InputIterator first, InputIterator last,
              UnaryFunctorType functor);

template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
void for_each(const std::string& label, const ExecutionSpace& exespace,  (2)
              InputIterator first, InputIterator last,
              UnaryFunctorType functor);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryFunctorType>
void for_each(const ExecutionSpace& exespace,
              const Kokkos::View<DataType, Properties...>& view,                      (3)
              UnaryFunctorType functor);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
void for_each(const std::string& label, const ExecutionSpace& exespace,
              const Kokkos::View<DataType, Properties...>& view,                      (4)
              UnaryFunctorType func);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Applies the UnaryFunctorType `func` to the result of dereferencing each iterator in `[first,last)` for (1,2) and to the view elements in (3,4).

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
  - for 1, the default string is: "Kokkos::for_each_iterator_api_default"
  - for 3, the default string is: "Kokkos::for_each_view_api_default"

- `first, last`:
  - range of elements to operate on
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `func`:
  - function object called on the all the elements;
  - The signature of the function should be `func(v)`.
  - Must be valid to be called from the execution space passed, and must accept every argument `v` of type (possible const) `value_type`, where `value_type` is the value type of `InputIterator`, and must not modify `v`.
  - must conform to:
  ```cpp
  struct func
  {
     KOKKOS_INLINE_FUNCTION
     void operator()(const /*type needed */ & operand) const { /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     void operator()(/*type needed */ operand) const { /* ... */; }
  };
  ```

## Return

(nothing)


## Example
```cpp
namespace KE = Kokkos::Experimental;

template<class ValueType>
struct IncrementValsFunctor
{
  const ValueType m_value;
  IncrementValsFunctor(ValueType value) : m_value(value){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ValueType & operand) const {
    operand += m_value;
  }
};

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<exespace, int*>;
view_type a("a", 15);
// fill "a" somehow

// create functor
IncrementValsFunctor<int> p(5);

// Increment each element in "a" by 5.
KE::for_each(exespace, KE::begin(a), KE::end(a), p);

// assuming OpenMP is enabled, then you can also explicitly call
KE::for_each(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
```
