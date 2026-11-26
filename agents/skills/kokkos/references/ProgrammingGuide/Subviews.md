# Subviews

After reading this chapter, you should understand the following:

*  A _slice_ of a multidimensional array behaves as an array and is a view of a structured subset of that array
*  A _subview_ is a slice of an existing Kokkos View
*  A subview has the same reference count as its parent View
*  Use C++11 type inference (`auto`) to let Kokkos pick the subview's type

## A subview is a slice of a View

In Kokkos, a _subview_ is a slice of a View. A _slice_ of a multidimensional array behaves as an array, and is a view of a
structured subset of the original array. "Behaves as an array" means that the slice has the same syntax as an array does; one can access its entries using array indexing notation. "View" means that the slice and the original array point to the same data, i.e, the slice sees changes to the original array and vice versa. "Structured subset" means a cross product of indices along each dimension, as for example a plane or face of a cube. If the original array has dimensions $\left(N_0, N_1, ..., N_{k-1}\right)$, then a slice views all entries whose indices are $\left(a_0, a_1, ..., a_{k-1}\right)$, where $a_j$ is an ordered subset of $\left\{N_0, N_1, ..., N_j-1\right\}$.

Array slices are handy for encapsulation. A slice looks and acts like an array, so you can pass it into functions that expect an array. For example, you can write a function for processing boundaries (as slices) of a structured grid without needing to tell that function properties of the entire grid.

Programming languages like Fortran 90, Matlab, and Python have a special "colon" notation for representing slices. For example, if `A` is an $M \times N$ array, then

* `A(:, :)` represents the whole array,
* `A(:, 3)` represents the fourth column (if the language has zero-based indices,
   or the third column if the language has one-based indices),
* `A(4, :)` represents the fifth row,
* `A(2:4, 3:7)` represent the sub-array of rows 3-4 and columns 4-7 (languages
   differ on whether the ranges are inclusive or exclusive of the last index --
   Kokkos, like Python, is exclusive), and
* `A(3, 4)` represents a "zero-dimensional" slice which views the entry
   in the fourth row and fifth column of the matrix.

These languages may have more elaborate notation for expressing sets of indices other than contiguous ranges.  These may include "strided" subsets of indices, like `3:2:9` = `{ 3, 5, 7, 9}`, or even arbitrary sets of indices.


## How to take a subview

To take a subview of a View, you can use the [`Kokkos::subview`](../API/core/view/subview) function. This function is overloaded for all different kinds of Views and index ranges. For example, the following code is equivalent to the above example `A(2:4, 3:7)`:

```c++
const size_t N0 = ...;
const size_t N1 = ...;
Kokkos::View<double**> A ("A", N0, N1);

auto A_sub = subview (A, make_pair (2, 4), make_pair (3, 7));
```

In the above example and those that follow in this chapter, we assume that [`Kokkos::View`](../API/core/view/view), [`Kokkos::subview`](../API/core/view/subview), `Kokkos::ALL`, `std::make_pair`, and `std::pair` have been imported into the working C++ namespace.

The Kokkos equivalent of a contiguous index range `3:7` is `pair<size_t, size_t>(3, 7)`. The Kokkos equivalent of
`:` (a colon by itself; the whole index range for that dimension) is `ALL()` (an instance of the `ALL` class, which Kokkos uses only for this purpose). Kokkos does not currently have equivalents of the strided or arbitrary index sets.

A subview has the same reference count as its parent [`View`](../API/core/view/view), so the parent [`View`](../API/core/view/view) won't be deallocated before all subviews go away. Every subview is also a [`View`](../API/core/view/view). This means that you may take a subview of a subview.

Another way of getting a subview is through the appropriate [`View`](../API/core/view/view) constructor.

```c++
const size_t N0 = ...;
const size_t N1 = ...;
Kokkos::View<double**> A ("A", N0, N1);

Kokkos::View<double**,LayoutStride> A_sub(A,make_pair(2,4),make_pair(3,7));
```

For this usage you must know the layout type of the subview. On the positive side, such a direct construction is generally a bit cheaper than through the `Kokkos::subview` function.

### C++11 type deduction

Note the use of the C++11 keyword `auto` in the above example. A subview may have a different type than its parent View. For instance, if `A` has `LayoutRight` and `A_sub` has `LayoutStride`, a subview of an entire row of `A` will have `LayoutRight` as well but a subview of an entire column of `A` will have `LayoutStride`. If you assign the result of the `subview` function to the wrong type, Kokkos will emit a compile-time error. The easiest way to avoid this is to use the C++11 `auto` keyword to let the compiler deduce the correct type for you. That said `auto` comes with its own cost. It generally is more expensive for compilers to deal with `auto` than with explicitly known types.

### Dimension of a subview

Suppose that a View has `k` dimensions. Then when calling `subview` on that View, you must pass in `k` arguments. Every argument that is a single index -- that is, not a pair or `ALL()` -- reduces the dimension of the resulting View by 1.

### Degenerate Views

Given a View with `k` dimensions, we call that View _degenerate_ if any of its dimensions is zero. Degenerate Views are useful for keeping code simple by avoiding special cases. For example, consider a MPI (Message-Passing Interface) distributed-memory parallel code that uses Kokkos to represent local (per-process) data structures. Suppose that the code distributes a dense matrix (2-D array) in block row fashion over the MPI processes in a communicator. It could be that some processes own zero rows of the matrix. This may not be efficient, since those processes do no work yet participate in collectives, but it might be possible. In this case, allowing Views with zero rows would reduce the number of special cases in the code.

## Obtaining the type of a subview

For some applications, it is useful to know the type of a subview, given the arguments. In these cases, using `auto` type deduction can be inconvenient. While it is possible to use `decltype()` and `std::declval` to obtain this from the `subview` function, Kokkos provides a convenient type alias:

```c++
using my_view_type = Kokkos::View<double **>;
using my_subview_type = Kokkos::Subview<my_view_type,
                                        std::remove_const_t<decltype(Kokkos::ALL)>,
                                        Kokkos::pair<unsigned, unsigned>>;

using my_subview_type_deduced = decltype(subview(std::declval<my_view_type>(), Kokkos::ALL, std::make_pair(1u, 1u)));

static_assert(std::is_same<my_subview_type, my_subview_type_deduced>::value);
```
