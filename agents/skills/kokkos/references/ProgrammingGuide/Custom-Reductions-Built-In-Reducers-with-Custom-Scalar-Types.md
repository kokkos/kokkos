# Built-In Reducers with Custom Scalar Types

In order to use a Custom Scalar Type with Built-in reductions, the following requirements must be fulfilled.

   * An initialization function must be provided via a specialization of the `Kokkos::reduction_identity<T>` class.  
   * Operators required for applied reduction class must be implemented.
   * The class / struct must either use the default copy constructor or have a specific copy constructor 
     implemented. 

## Example

This example performs a custom reduction on an array using the built-in [`Sum`](../API/core/builtinreducers/Sum) reducer. 

```c++
namespace sample {  // namespace helps with name resolution in reduction identity 
   template< class ScalarType, int N >
   struct array_type {
     ScalarType the_array[N];
  
     KOKKOS_INLINE_FUNCTION   // Default constructor - Initialize to 0's
     array_type() { 
       for (int i = 0; i < N; i++ ) { the_array[i] = 0; }
     }
     KOKKOS_INLINE_FUNCTION   // Copy Constructor
     array_type(const array_type & rhs) { 
        for (int i = 0; i < N; i++ ){
           the_array[i] = rhs.the_array[i];
        }
     }
     KOKKOS_INLINE_FUNCTION   // add operator
     array_type& operator += (const array_type& src) {
       for ( int i = 0; i < N; i++ ) {
          the_array[i]+=src.the_array[i];
       }
       return *this;
     }
   };
   typedef array_type<int,4> ValueType;  // used to simplify code below
}
namespace Kokkos { //reduction identity must be defined in Kokkos namespace
   template<>
   struct reduction_identity< sample::ValueType > {
      KOKKOS_FORCEINLINE_FUNCTION static sample::ValueType sum() {
         return sample::ValueType();
      }
   };
}
int main( int argc, char* argv[] )
{
  int E = 1024;
  Kokkos::initialize( argc, argv );
  {
     sample::ValueType tr;         
     Kokkos::parallel_reduce( E, KOKKOS_LAMBDA (const int& i, 
                                                sample::ValueType & upd) {
        int ndx =i%4;  // sum all of the i%4 entries (divide total by 4)
        upd.the_array[ndx] += 1; 
     }, Kokkos::Sum<sample::ValueType>(tr) );
     printf( "  Computed result for %d is %d, %d, %d, %d \n", 
             E, tr.the_array[0], tr.the_array[1], 
                tr.the_array[2], tr.the_array[3] );
  }
  Kokkos::finalize();

  return 0;
}
```
