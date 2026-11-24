# `Experimental::simd_mask`

Header File: `Kokkos_SIMD.hpp`

Usage: 

`Kokkos::Experimental::simd_mask` is an abstraction over platform-specific vector masks and calls platform-specific vector intrinsics.
It is based on the `simd_mask` type proposed for ISO C++ in [this document](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4808.pdf)

## Interface

```c++
namespace Experimental {
template <class T, class Abi>
class basic_simd_mask;
}
```

### Template Parameters

The first template parameter `T` should be a C++ fundamental type for which the current platform supports vector intrinsics. Kokkos supports the following types for `T`:
 - `float`
 - `double`
 - `std::int32_t`
 - `std::int64_t`
 - `std::uint32_t`
 - `std::uint64_t`

The second template parameter `Abi` is one of the pre-defined ABI types in the namespace `Kokkos::Experimental::simd_abi`. This type determines the size of the vector and what architecture-specific intrinsics will be used. The following types are always available in that namespace:
 - `scalar`: a fallback ABI which always has vector size of 1 and uses no special intrinsics.
 - `native`: the "best" ABI for the architecture for which Kokkos was compiled. (deprecated since Kokkos 4.6)

### Typedefs

 *  `value_type`: Equal to `bool`
 *  `reference`: This type should be convertible to `bool` and `bool` should be assignable to `reference`. It may be a plain reference or it may be a special class that calls vector intrinsics to extract or fill in one mask bit. (removed in Kokkos 4.6)
 *  `simd_type`: Equal to `simd<T, Abi>`
 *  `abi_type`: Equal to `Abi`

### Width

 * `static constexpr std::integral_constant<simd_size_t, N> size()`: `basic_simd_mask<T, Abi>::size()` is a compile-time constant of the width of the vector, i.e. the number of values of type `T` in the vector.

### Constructors

  * `simd_mask()`: Default Constructor. The vector values are not initialized by this constructor.
  * `simd_mask(bool)`: Single-value constructor. All values in the mask will be set to the value of the argument.
  * `template <class G> simd_mask(G&& gen)`: Generator constructor. The generator `gen` should be a callable type (e.g. functor) that can accept `std::integral_constant<std::size_t, i>()` as an argument and return something convertible to `bool`. Vector mask value `i` will be initialized to the value of `gen(std::integral_constant<std::size_t, i>())`.

### Value Access Methods
  * `bool operator[](simd_size_t) const`: returns the mask value `i`.
  * `reference operator[](std::size_t)`: returns a reference to mask value `i` that can be modified. (removed in Kokkos 4.6)

### Boolean Operators
  * `simd_mask simd_mask::operator!() const`
  * `simd_mask operator&&(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator||(const simd_mask& lhs, const simd_mask& rhs)`

### Bitwise Operators
  * `simd_mask simd_mask::operator~() const`
  * `simd_mask operator&(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator|(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator^(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator&=(simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator|=(simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator^=(simd_mask& lhs, const simd_mask& rhs)`

### Comparison Operators
  * `simd_mask operator==(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator!=(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator>=(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator<=(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator>(const simd_mask& lhs, const simd_mask& rhs)`
  * `simd_mask operator<(const simd_mask& lhs, const simd_mask& rhs)`

### Reductions
  * `bool all_of(const simd_mask&)`: returns true iff all of the vector values in the mask are true
  * `bool any_of(const simd_mask&)`: returns true iff any of the vector values in the mask are true
  * `bool none_of(const simd_mask&)`: returns true iff none of the vector values in the mask are true

### Global Typedefs
  * `template <class T> Kokkos::Experimental::native_simd_mask`: Alias for `Kokkos::Experimental::simd_mask<T, Kokkos::Experimental::simd_abi::native<T>>`. (deprecated since Kokkos 4.6)
  * `template <class T, int N> Kokkos::Experimental::simd_mask`: Alias for `Kokkos::Experimental::basic_simd_mask<T, ...>`. (since Kokkos 4.6)

## Examples

```c++
#include <Kokkos_SIMD.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    using mask_type = Kokkos::Experimental::simd_mask<double>;
    mask_type a([] (std::size_t i) { return i == 0; });
    mask_type b([] (std::size_t i) { return i == 1; });
    mask_type c([] (std::size_t i) { return i == 0 || i == 1; });
    if (all_of(c == (a || b))) {
      printf("Kokkos simd_mask works as expected!");
    }
  }
  Kokkos::finalize();
}
```
