# `Experimental::simd`

Header File: `Kokkos_SIMD.hpp`

Usage: 

`Kokkos::Experimental::simd` is an abstraction over platform-specific vector datatypes and calls platform-specific vector intrinsics.
It is based on the `simd` type proposed for ISO C++ in [this document](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4808.pdf)

## Interface

```c++
namespace Experimental {
template <class T, class Abi>
class basic_simd;
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

 *  `value_type`: Equal to `T`
 *  `reference`: This type should be convertible to `value_type` and `value_type` should be assignable to `reference`. It may be a plain reference or it may be an implementation-defined type that calls vector intrinsics to extract or fill in one vector lane. (removed in Kokkos 4.6)
 *  `mask_type`: Equal to `basic_simd_mask<T, Abi>`
 *  `abi_type`: Equal to `Abi`

### Width

 * `static constexpr std::integral_constant<simd_size_t, N> size()`: `basic_simd<T, Abi>::size()` is a compile-time constant of the width of the vector, i.e. the number of values of type `T` in the vector.

### Constructors

  * `simd()`: Default Constructor. The vector values are not initialized by this constructor.
  * `template <class U> simd(U&&)`: Single-value constructor. The argument will be converted to `value_type` and all the vector values will be set equal to that value.
  * `template <class G> simd(G&& gen)`: Generator constructor. The generator `gen` should be a callable type (e.g. functor) that can accept `std::integral_constant<std::size_t, i>()` as an argument and return something convertible to `value_type`. Vector lane `i` will be initialized to the value of `gen(std::integral_constant<std::size_t, i>())`.

#### Simd Flags

  * Available `simd_flags` are `simd_flag_default` and `simd_flag_aligned`.
  * For backward compatibility, `Kokkos::Experimental::element_aligned_tag` and `Kokkos::Experimental::vector_aligned_tag` types are available.
  * `Kokkos::Experimental::element_aligned_tag` is a type alias for `decltype(simd_flag_default)` and `Kokkos::Experimental::vector_aligned_tag` is a type alias for `decltype(simd_flag_aligned)`.

### Value Access Methods
  * `value_type operator[](simd_size_t) const`: returns the vector value `i`.
  * `reference operator[](std::size_t)`: returns a reference to vector value `i` that can be modified. (removed in Kokkos 4.6)

### Arithmetic Operators
  * `simd simd::operator-() const`
  * `simd operator+(const simd& lhs, const simd& rhs)`
  * `simd operator-(const simd& lhs, const simd& rhs)`
  * `simd operator*(const simd& lhs, const simd& rhs)`
  * `simd operator/(const simd& lhs, const simd& rhs)`
  * `simd simd::operator~() const`
  * `simd operator&(const simd& lhs, const simd& rhs)`
  * `simd operator|(const simd& lhs, const simd& rhs)`
  * `simd operator^(const simd& lhs, const simd& rhs)`
  * `simd operator>>(const simd& lhs, const simd& rhs)`
  * `simd operator>>(const simd& lhs, int rhs)`
  * `simd operator<<(const simd& lhs, const simd& rhs)`
  * `simd operator<<(const simd& lhs, int rhs)`

### Compound Assignment Operators
  * `simd operator+=(simd& lhs, const simd& rhs)`
  * `simd operator-=(simd& lhs, const simd& rhs)`
  * `simd operator*=(simd& lhs, const simd& rhs)`
  * `simd operator/=(simd& lhs, const simd& rhs)`
  * `simd operator&=(simd& lhs, const simd& rhs)`
  * `simd operator|=(simd& lhs, const simd& rhs)`
  * `simd operator^=(simd& lhs, const simd& rhs)`
  * `simd operator>>=(simd& lhs, const simd& rhs)`
  * `simd operator<<=(simd& lhs, const simd& rhs)`

### Comparison Operators
  * `mask_type operator==(const simd& lhs, const simd& rhs)`
  * `mask_type operator!=(const simd& lhs, const simd& rhs)`
  * `mask_type operator>=(const simd& lhs, const simd& rhs)`
  * `mask_type operator<=(const simd& lhs, const simd& rhs)`
  * `mask_type operator>(const simd& lhs, const simd& rhs)`
  * `mask_type operator<(const simd& lhs, const simd& rhs)`

### Rounding Functions
  * `simd Kokkos::floor(const simd& lhs)`
  * `simd Kokkos::ceil(const simd& lhs)`
  * `simd Kokkos::round(const simd& lhs)`
  * `simd Kokkos::trunc(const simd& lhs)`

### Min/Max Functions
  * `simd Kokkos::min(const simd& lhs, const simd& rhs)`
  * `simd Kokkos::max(const simd& lhs, const simd& rhs)`

### Reductions 
  * `T Kokkos::Experimental::reduce(const simd& lhs, const simd_mask& mask)`
  * `T Kokkos::Experimental::reduce(const simd& lhs, Op binary_op)`
  * `T Kokkos::Experimental::reduce_min(const simd& lhs, const simd_mask& mask)`
  * `T Kokkos::Experimental::reduce_min(const simd& lhs)`
  * `T Kokkos::Experimental::reduce_max(const simd& lhs, const simd_mask& mask)`
  * `T Kokkos::Experimental::reduce_max(const simd& lhs)`

### `<cmath>` Functions
  * `simd Kokkos::abs(const simd& lhs)`
  * `simd Kokkos::exp(const simd& lhs)`
  * `simd Kokkos::exp2(const simd& lhs)`
  * `simd Kokkos::log(const simd& lhs)`
  * `simd Kokkos::log10(const simd& lhs)`
  * `simd Kokkos::sqrt(const simd& lhs)`
  * `simd Kokkos::cbrt(const simd& lhs)`
  * `simd Kokkos::sin(const simd& lhs)`
  * `simd Kokkos::cos(const simd& lhs)`
  * `simd Kokkos::tan(const simd& lhs)`
  * `simd Kokkos::asin(const simd& lhs)`
  * `simd Kokkos::acos(const simd& lhs)`
  * `simd Kokkos::atan(const simd& lhs)`
  * `simd Kokkos::sinh(const simd& lhs)`
  * `simd Kokkos::cosh(const simd& lhs)`
  * `simd Kokkos::tanh(const simd& lhs)`
  * `simd Kokkos::asinh(const simd& lhs)`
  * `simd Kokkos::acosh(const simd& lhs)`
  * `simd Kokkos::atanh(const simd& lhs)`
  * `simd Kokkos::erf(const simd& lhs)`
  * `simd Kokkos::erfc(const simd& lhs)`
  * `simd Kokkos::tgamma(const simd& lhs)`
  * `simd Kokkos::lgamma(const simd& lhs)`
  * `simd Kokkos::pow(const simd& lhs, const simd& rhs)`
  * `simd Kokkos::hypot(const simd& x, const simd& y)`
  * `simd Kokkos::hypot(const simd& x, const simd& y, const simd& z)`
  * `simd Kokkos::atan2(const simd& x, const simd& y)`
  * `simd Kokkos::copysign(const simd& mag, const simd& sgn)`
  * `simd Kokkos::fma(const simd& x, const simd& y, const simd& z)`

  These functions are only defined in `AVX2` and `AVX512` for `value_type=float` and `value_type=double`
  * `simd Kokkos::cbrt(simd& lhs)`
  * `simd Kokkos::exp(simd& lhs)`
  * `simd Kokkos::log(simd& lhs)`

### Load/Store Functions

  * `template <class U, class Flags> void copy_from(const U* mem, Flags flags)`: Loads the full vector of contiguous values starting at the address `mem`. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (deprecated since Kokkos 5.0)
  * `template <class U, class Flags> void copy_to(U* mem, Flags flags)`: Stores the full vector of contiguous values starting at the address `mem`. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (deprecated since Kokkos 5.0)

  * `template <class U, class Flags> simd simd_unchecked_load(U* mem, Flags flags)`: Loads the full vector of contiguous values starting at the address `mem`and returns a Kokkos simd type. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> simd simd_unchecked_load(U* mem, mask_type mask, Flags flags)`: Executes a masked load operation, loading vector value i at `mem[i]` if mask value mask[i] is true and returns a Kokkos simd type. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> simd simd_partial_load(U* mem, Flags flags)`: Loads the full vector of contiguous values starting at the address `mem` and returns a Kokkos simd type. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> simd simd_partial_load(U* mem, mask_type mask, Flags flags)`: Executes a masked load operation, loading vector value i at `mem[i]` if mask value mask[i] is true and returns a Kokkos simd type. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)

  * `template <class U, class Flags> void simd_unchecked_store(simd& s, U* mem, Flags flags)`: Stores the full vector of contiguous values starting at the address `mem`. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> void simd_unchecked_store(simd& s, U* mem, mask_type mask, Flags flags)`: Executes a masked store operation, storing vector value i at `mem[i]` if mask value mask[i] is true. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> void simd_partial_store(simd& s, U* mem, Flags flags)`: Stores the full vector of contiguous values starting at the address `mem`. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)
  * `template <class U, class Flags> void simd_partial_store(simd& s, U* mem, mask_type mask, Flags flags)`: Executes a masked store operation, storing vector value i at `mem[i]` if mask value mask[i] is true. `Flags` is the `simd_flags` that is used to describe the alignment at the address `mem`. (since Kokkos 5.0)

### Memory Permutes

  * `template <class R, class I, class Flags> simd Kokkos::Experimental::unchecked_gather_from(R&& in, const I& indices, Flags flags)`: Gathers values from `in[indices[i]]` and returns a Kokkos simd type. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> simd Kokkos::Experimental::unchecked_gather_from(R&& in, const mask_type& mask, const I& indices, Flags flags)`: Gathers values from `in[indices[i]]` if the mask value `mask[i]` is true and returns a Kokkos simd type. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> simd Kokkos::Experimental::partial_gather_from(R&& in, const I& indices, Flags flags)`: Gathers values from `in[indices[i]]` and returns a Kokkos simd type. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> simd Kokkos::Experimental::partial_gather_from(R&& in, const mask_type& mask, const I& indices, Flags flags)`: Gathers values from `in[indices[i]]` if the mask value `mask[i]` is true and returns a Kokkos simd type. (since Kokkos 5.1)

  * `template <class R, class I, class Flags> void Kokkos::Experimental::unchecked_scatter_to(const simd& s, R&& out, const I& indices, Flags flags)`: Scatters values from s to `out[indices[i]]`. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> void Kokkos::Experimental::unchecked_scatter_from(const simd& s, R&& in, const mask_type& mask, const I& indices, Flags flags)`: Scatters values from s to `out[indices[i]]` if the mask value `mask[i]` is true. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> void Kokkos::Experimental::partial_scatter_to(const simd& s, R&& out, const I& indices, Flags flags)`: Scatters values from s to `out[indices[i]]`. (since Kokkos 5.1)
  * `template <class R, class I, class Flags> void Kokkos::Experimental::partial_scatter_from(const simd& s, R&& in, const mask_type& mask, const I& indices, Flags flags)`: Scatters values from s to `out[indices[i]]` if the mask value `mask[i]` is true. (since Kokkos 5.1)

### Global Typedefs

  * `template <class T> Kokkos::Experimental::native_simd`: Alias for `Kokkos::Experimental::simd<T, Kokkos::Experimental::simd_abi::native<T>>`. (deprecated since Kokkos 4.6)
  * `template <class T, int N> Kokkos::Experimental::simd`: Alias for `Kokkos::Experimental::basic_simd<T, ...>` (since Kokkos 4.6)
  * `Kokkos::Experimental::element_aligned_tag`: Alias for `Kokkos::Experimental::simd_flags<>`
  * `Kokkos::Experimental::vector_aligned_tag`: Alias for `Kokkos::Experimental::simd_flags<simd_alignment_vector_aligned>`

## Examples

```c++
#include <Kokkos_SIMD.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    using simd_type = Kokkos::Experimental::simd<double>;
    simd_type a([] (std::size_t i) { return 0.1 * i; });
    simd_type b(2.0);
    simd_type c = Kokkos::sqrt(a * a + b * b);
    for (std::size_t i = 0; i < simd_type::size(); ++i) {
      printf("[%zu] = %g\n", i, c[i]);
    }
  }
  Kokkos::finalize();
}
```
