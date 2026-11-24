(KokkosALL)=
# `Kokkos::ALL`

Defined in header `<Kokkos_Core.hpp>`

```c++
namespace Kokkos{
  constexpr UNSPECIFIED-TYPE ALL = IMPLEMENTATION-DETAIL;
}
```

`Kokkos::ALL` is a constant of unspecified type that is used to select all elements in a dimension.


## Examples

```c++
Kokkos::View<double**[5]> a("A",N0,N1);

auto s  = Kokkos::subview(a,
              5,
              Kokkos::ALL,
              Kokkos::ALL);
```
