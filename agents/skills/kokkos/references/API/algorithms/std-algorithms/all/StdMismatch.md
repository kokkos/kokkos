
# `mismatch`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& exespace,
                  IteratorType1 first1,
                  IteratorType1 last1,                                               (1)
                  IteratorType2 first2,
                  IteratorType2 last2);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(
                const std::string& label,
                const ExecutionSpace& exespace,
                IteratorType1 first1,
                IteratorType1 last1,                                                 (2)
                IteratorType2 first2,
                IteratorType2 last2)

template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& exespace,
                  IteratorType1 first1,
                  IteratorType1 last1,                                               (3)
                  IteratorType2 first2,
                  IteratorType2 last2, BinaryPredicate pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const std::string& label,
                  const ExecutionSpace& exespace,
                  IteratorType1 first1,
                  IteratorType1 last1,                                               (4)
                  IteratorType2 first2,
                  IteratorType2 last2, BinaryPredicate pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const ExecutionSpace& exespace,
              const Kokkos::View<DataType1, Properties1...>& view1,                  (5)
              const Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const std::string& label, const ExecutionSpace& exespace,
              const Kokkos::View<DataType1, Properties1...>& view1,                  (6)
              const Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const ExecutionSpace& exespace,
              const Kokkos::View<DataType1, Properties1...>& view1,                  (7)
              const Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const std::string& label, const ExecutionSpace& exespace,
              const Kokkos::View<DataType1, Properties1...>& view1,                  (8)
              const Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns the first mismatching pair of elements from two ranges: one defined by [first1, last1) and another defined by [first2,last2) for (1,2,3,4).
Returns the first mismatching pair of elements from the two views `view1` and `view2` in (5,6,7,8).
The elements are compared using `operator==` in (1,2,5,6).
The elements in (3,4,7,8) are compared using a BinaryPredicate `pred`.

## Parameters and Requirements 

- `exespace`:
  - execution space instance

- `label`:
  - for 1,3, the default string is: "Kokkos::mismatch_iterator_api_default"
  - for 5,7, the default string is: "Kokkos::mismatch_view_api_default"

- `first1`, `last1`, `first2`, `last2`:
  - range of elements to compare
  - must be *random access iterators*
  - must represent valid ranges, i.e., `last1 >= first1` and `last2 >= first2` 
  - must be accessible from `exespace`

- `view1`, `view2`:
  - views to compare
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `pred`
  ```cpp
  template <class ValueType1, class ValueType2 = ValueType1>
  struct IsEqualFunctor {

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
    return (a == b);
    }
  };
 ```

## Return

- (1,2) - Kokkos::pair, where the `.first` and `.second` are the IteratorType1 and IteratorType2 instances where the `operator==` evaluates to false
- (3,4) - Kokkos::pair, where the `.first` and `.second` are the IteratorType1 and IteratorType2 instances where the `pred` evaluates to false

## Example 

```cpp
namespace KE = Kokkos::Experimental;

template <class ValueType1, class ValueType2 = ValueType1>
struct MismatchFunctor {

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
    if(a != b)
        return (Kokkos::pair<ValueType1, ValueType2> (a,b));
  }
};

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<exespace, int*>;
view_type a("a", 15);
view_type b("b", 15);
// fill a,b somehow

// create functor
MismatchFunctor<int, int> p();

Kokkos::pair<int,int> mismatch_index = KE::mismatch(exespace, KE::begin(a), KE::end(a), KE::begin(b), KE::end(b) p);

// assuming OpenMP is enabled, then you can also explicitly call
Kokkos::pair<int,int> mismatch_index = KE::mismatch(Kokkos::OpenMP(), KE::begin(a), KE::end(a), KE::begin(b), KE::end(b), p);
```
