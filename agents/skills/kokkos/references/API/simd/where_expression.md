# `Experimental::where_expression`

(Deprecated since Kokkos 5.0)

Header File: `Kokkos_SIMD.hpp`

Usage:

`Kokkos::Experimental::where_expression` references a subset of the values in a vector register. Which values are in the subset is described by a mask. `where_expression` thus forms the basis for masked operations on vector values.

## Interface

```c++
namespace Experimental {
template <class M, class T>
class const_where_expression;
template <class M, class T>
class where_expression : public const_where_expression;
}
```

### Template Parameters
The first template parameter `M` is the mask type, and should either be an instance of class template `Kokkos::Experimental::simd_mask` or `bool`.
The second template parameter `T` is the value type, and should either be an instance of class template `Kokkos::Experimental::simd` or a fundamental type such as `double`.

### Where Function
Where expression objects are only constructed by calling the non-member method `Kokkos::Experimental::where`:
 * `template <class T, class Abi> where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(const simd_mask<T, Abi>&, simd<T, Abi>&)`: creates a non-const where expression that references the values in the `simd` argument selected by the `simd_mask` argument.
 * `template <class T, class Abi> const_where_expression<simd_mask<T, Abi>, simd<T, Abi>>
    where(const simd_mask<T, Abi>&, const simd<T, Abi>&)`: creates a const where expression that references the values in the `simd` argument selected by the `simd_mask` argument.

### Load/Store Methods
 * `template<class U, class Flags> void const_where_expression::copy_to(U* mem, Flags) const`: Executes a masked store operation, storing vector value `i` at `mem[i]` only if mask value `i` is true. `Flags` is the `simd_flags` type that is used to describe the alignment at the address `mem`.
 * `template<class U, class Flags> void where_expression::copy_from(const U* mem, Flags)`: Executes a masked load operation, loading vector value `i` from `mem[i]` only if mask value `i` is true. `Flags` is the `simd_flags` type that is used to describe the alignment at the address `mem`.

#### Simd Flags
 * Available `simd_flags` are `simd_flag_default` and `simd_flag_aligned`.
 * For backward compatibility, `Kokkos::Experimental::element_aligned_tag` and `Kokkos::Experimental::vector_aligned_tag` types are available.
 * `Kokkos::Experimental::element_aligned_tag` is a type alias for `decltype(simd_flag_default)` and `Kokkos::Experimental::vector_aligned_tag` is a type alias for `decltype(simd_flag_aligned)`.

### Gather/Scatter Methods
 These methods were added by Kokkos and are not present in the ISO C++ proposal.
 * `void const_where_expression::scatter_to(double* mem, simd<std::int32_t, Abi> const& index) const`: When the value type `T` is `Kokkos::Experimental::simd<double, Abi>`, this function scatters values into `mem[index[i]]` if the mask value `i` is true.
 * `void where_expression::gather_from(const double* mem, simd<std::int32_t, Abi> const& index)`: When the value type `T` is `Kokkos::Experimental::simd<double, Abi>`, this function gathers values from `mem[index[i]]` if the mask value `i` is true.

### Assignment
 * `template<class U> void where_expression::operator=(U&& x)`: Assigns to vector value `i` the value `x[i]` only if mask value `i` is true.

## Example

```c++
#include <Kokkos_SIMD.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    using simd_type = Kokkos::Experimental::simd<double>;
    // the first value in vector a will be negative after this
    simd_type a([] (std::size_t i) { return 1.0 * i - 1.0; });
    // we can use where expressions to set negative values to 0.0
    where(a < 0.0, a) = 0.0;
    // now it might be safer to call a function with domain limitations
    auto b = Kokkos::sqrt(a);
  }
  Kokkos::finalize();
}
```
