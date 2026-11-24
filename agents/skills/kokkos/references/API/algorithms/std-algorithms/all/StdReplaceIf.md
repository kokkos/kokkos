
# `replace_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
void replace_if(const ExecutionSpace& exespace,                              (1)
                IteratorType first, IteratorType last,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
void replace_if(const std::string& label, const ExecutionSpace& exespace,    (2)
                IteratorType first, IteratorType last,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
void replace_if(const ExecutionSpace& exespace,                              (3)
                const Kokkos::View<DataType, Properties...>& view,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
void replace_if(const std::string& label, const ExecutionSpace& exespace,    (4)
                const Kokkos::View<DataType, Properties...>& view,
                UnaryPredicateType pred, const T& new_value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Replaces with `new_value` all the elements for which `pred` is `true` in
the range `[first, last)` (overloads 1,2) or in `view` (overloads 3,4).

## Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `new_value`: same as in [`replace`](./StdReplace)
- `label`:
  - for 1, the default string is: "Kokkos::replace_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_if_view_api_default"
- `pred`:
  - *unary* predicate returning `true` for the required element to replace; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `IteratorType` (for 1,2) or the value type of `view` (for 3,4),
  and must not modify `v`.
  - must conform to:
  ```c++
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & v) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(value_type v) const { return /* ... */; }
  };
  ```


## Return

None

## Example

```c++
template <class ValueType>
struct IsPositiveFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val > 0); }
};
// ---

namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// do something with a
// ...

const double oldValue{2};
const double newValue{34};
KE::replace_if(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a),
   IsPositiveFunctor<double>(), newValue);

// explicitly set label and execution space (assuming active)
KE::replace_if("mylabel", Kokkos::OpenMP(), a,
   IsPositiveFunctor<double>(), newValue);
```
