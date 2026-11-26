# Custom Reducers

Custom arbitrary reductions are implemented using a reduction class and a "reduced" class.  The "reduced" class is much like the custom scalar type used with [Built-In Reducers with Custom Scalar Types](Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types) and the reduction class implements the [ReducerConcept](../API/core/builtinreducers/ReducerConcept)

The following requirements must be fulfilled for the "reduced" class
     
   * Operators required for applied reduction class must be implemented.
   * The class / struct must either use the default copy constructor or have a specific copy constructor 
     implemented. 

The following requirements must be fulfilled for the reduction class
     
   * Typedefs reducer, value_type and result_view_type must be defined.  see the [ReducerConcept](../API/core/builtinreducers/ReducerConcept) for details.
   * The reducer concept methods must be implemented
   * The exposed result_view_type must be defined in the memory space where the object is used 

Note: Even for tagged reductions, i.e., when specifying a tag in the policy, potential `init`/`join`/`final` member functions must not take a `WorkTag` argument.

## Example

This example performs a custom reduction on an array using a custom class and reducer. 

```c++
#include <Kokkos_Core.hpp>

namespace sample {

template <class ScalarType, int N>
struct array_type {
  ScalarType myArray[N];

  KOKKOS_INLINE_FUNCTION
  array_type() { init(); }

  KOKKOS_INLINE_FUNCTION
  array_type(const array_type& rhs) {
    for (int i = 0; i < N; i++) {
      myArray[i] = rhs.myArray[i];
    }
  }

  KOKKOS_INLINE_FUNCTION  // initialize myArray to 0
      void
      init() {
    for (int i = 0; i < N; i++) {
      myArray[i] = 0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  array_type& operator+=(const array_type& src) {
    for (int i = 0; i < N; i++) {
      myArray[i] += src.myArray[i];
    }
    return *this;
  }
};

template <class T, class Space, int N>
struct SumMyArray {
 public:
  // Required
  typedef SumMyArray reducer;
  typedef array_type<T, N> value_type;
  typedef Kokkos::View<value_type*, Space, Kokkos::MemoryUnmanaged>
      result_view_type;

 private:
  value_type& value;

 public:
  KOKKOS_INLINE_FUNCTION
  SumMyArray(value_type& value_) : value(value_) {}

  // Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const { val.init(); }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return value; }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return result_view_type(&value, 1); }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const { return true; }
};
}  // namespace sample

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int E = 1024;

    typedef sample::array_type<int, 4> ValueType;
    typedef sample::SumMyArray<int, Kokkos::HostSpace, 4> ArraySumResult;

    ValueType tr;

    Kokkos::parallel_reduce(
        E,
        KOKKOS_LAMBDA(const int i, ValueType& upd) {
          int ndx = i % 4;  // sum all of the i%4 entries (divide total by 4)
          upd.myArray[ndx] += 1;
        },
        ArraySumResult(tr));

    // Output result.
    printf("  Computed result %d, %d, %d, %d \n", tr.myArray[0], tr.myArray[1],
           tr.myArray[2], tr.myArray[3]);
  }
  Kokkos::finalize();
}
```
 
