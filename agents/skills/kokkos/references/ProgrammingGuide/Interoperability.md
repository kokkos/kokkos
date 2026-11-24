# Interoperability and Legacy Codes

One goal of Kokkos is to support incremental adoption in legacy applications. This facilitates a step by step conversion allowing for continuous testing of functionality (and in certain bounds) of performance. One feature of this is full interoperability with the underlying backend programming models. This also allows for target specific optimizations written directly in the backend model in order to achieve maximal performance.

After reading this chapter, you should understand the following:

* Restriction on interoperability with raw OpenMP and Cuda.
* How to handle external data structures.
* How to incrementally convert legacy data structures.
* How to call non-Kokkos third party libraries safely.

In all code examples in this chapter, we assume that all classes in the `Kokkos` namespace have been imported into the working namespace.

## OpenMP, C++ Threads and CUDA interoperability

Since the implementation of Kokkos is achieved with a C++ library it provides full interoperability with the underlying backend programming models. In particular, it allows for mixing of OpenMP, CUDA and Kokkos code in the same compilation unit. This is true for both the parallel execution layers of Kokkos as for the data layer.

It is important to recognize that this does not lift certain restrictions. For example, it is not valid to allocate a view inside of an OpenMP parallel region, the same way as it is not valid to allocate a View inside a [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) kernel. Indeed, there are things which are slightly more cumbersome when mixing the models. Assigning one view to another inside a Cuda kernel or an OpenMP parallel region is only possible if the destination view is unmanaged. During dispatch of kernels with [`parallel_for()`](../API/core/parallel-dispatch/parallel_for), all Views referenced in the functor or lambda are automatically switched into unmanaged mode. This would not happen when simply entering an OpenMP parallel region.

### Cuda interoperability

The most important thing to know for Cuda interoperability is that the provided macro `KOKKOS_INLINE_FUNCTION` evaluates to `__host__ __device__ inline`. This means that calling a pure `__device__` function (for example Cuda intrinsics or device functions of libraries) must be protected with the `__CUDA_ARCH__` pragma.

```c++
__device__ SomeFunction(double* x) {
  ...
}

struct Functor {
  typedef Cuda execution_space;
  View<double*,Cuda> a;
  KOKKOS_INLINE_FUNCTION
 void operator(const int& i) {
    #ifdef __CUDA_ARCH__
    int block = blockIdx.x;
    int thread = threadIdx.x;
    if (thread == 0)
      SomeFunction(&a(block));
    #endif
  }
}
```

The [`RangePolicy`](../API/core/policies/RangePolicy) starts a 1D grid of 1D thread blocks so that the index `i` is calculated as `blockIdx.x * blockDim.x + threadIdx.x`. For the `TeamPolicy` the number of teams is the grid dimension, while the number of threads per team is mapped to the Y-dimension of the Cuda thread-block. The optional vector length is mapped to the X-dimension. For example, `TeamPolicy<Cuda>(100,12,16)` would start a 1D grid of size 100 with block-dimensions (16,12,1) while `TeamPolicy<Cuda>(100,96)` would result in a grid size of 100 with block-dimensions of (1,96,1). The restrictions on the vector length (power of two and smaller than 32 for the Cuda execution space) guarantee that vector loops are performed by threads which are part of a single warp.

### OpenMP

One restriction on OpenMP interoperability is that it is not valid to increase the number of threads via `omp_set_num_threads()` after initializing Kokkos. This restriction is necessary for bookkeeping when Kokkos does allocation for internal per-thread data structures. It is however valid to ask for the thread ID inside a Kokkos parallel kernel compiled for the OpenMP execution space. It is also valid to use OpenMP constructs such as OpenMP atomics inside a parallel kernel or functions called by it. However, what happens when mixing OpenMP and Kokkos atomics is undefined since those will not necessarily map to the same underlying mechanism.


## Legacy data structures

There are two principal mechanisms to facilitate interoperability with legacy data structures: 

1. Kokkos allocates data and raw pointers that are extracted to create legacy data structures and 
1. unmanaged views can be used to view externally allocated data. 

In both cases, it is mandatory to fix the Layout of the Kokkos view to the actual layout used in the legacy data structure. Note that the user is responsible for insuring proper access capabilities. For example, a pointer obtained from a view in the `CudaSpace` may only be accessed from Cuda kernels, and a View constructed from memory acquired through a call to `new` will typically only be accessible from Execution spaces which can access the `HostSpace`.

## Raw allocations through Kokkos

A simple way to add support for multiple memory spaces to a legacy app is to use [`kokkos_malloc`](../API/core/c_style_memory_management/malloc), [`kokkos_free`](../API/core/c_style_memory_management/free) and [`kokkos_realloc`](../API/core/c_style_memory_management/realloc). The functions are templated on the memory space and thus allow targeted placement of data structures:

```c++
// Allocate an array of 100 doubles in the default memory space
double* a = (double*) kokkos_malloc<>(100*sizeof(double));

// Allocate an array of 150 int* in the Cuda UVM memory space
// This allocation is accessible from the host
int** 2d_array = (int**) kokkos_malloc<CudaUVMSpace>
                         (150*sizeof(int*));

// Fill the pointer array with pointers to data in the Cuda Space
// Since it is not the UVM space you can access 2d_array[i][j] only inside a Cuda Kernel
for(int i=0;i<150;i++)
  2d_array[i] = (int*) kokkos_malloc<CudaSpace>(200*sizeof(int));
```

A common usage scenario of this capability is to allocate all memory in the CudaUVMSpace when compiling for GPUs. This allows all allocations to be accessible from legacy code sections as well as from parallel kernels written with Kokkos.

### External memory management

When memory is managed externally, for example because Kokkos is used in a library which is given pointers to data allocations as input, it can be necessary or convenient to wrap the data into Kokkos views. If the library anyway receives the data to create a copy, it is straight forward to allocate the internal data structure as a view and copy the data in a parallel kernel element by element. Note that you might need to first copy into a host view before copying to the actual destination memory space:

```c++
template<class ExecutionSpace>
void MyKokkosFunction(double* a, const double** b, int n, int m) {
  // Define the host execution space and the view types
  typedef HostSpace::execution_space host_space;
  typedef View<double*,ExecutionSpace> t_1d_device_view;
  typedef View<double**,ExecutionSpace> t_2d_device_view;

  // Allocate the view in the memory space of ExecutionSpace
  t_1d_device_view d_a("a",n);
  // Create a host copy of that view
  typename t_1d_device_view::host_mirror_type h_a = create_mirror_view(a);
  // Copy the data from the external allocation into the host view
  parallel_for(RangePolicy<host_space>(0,n),
    KOKKOS_LAMBDA (const int& i) {
    h_a(i) = a[i];
  });
  // Copy the data from the host view to the device view
  deep_copy(d_a,h_a);

  // Allocate a 2D view in the memory space of ExecutionSpace
  t_2d_device_view d_b("b",n,m);
  // Create a host copy of that view
  typename t_2d_device_view::host_mirror_type h_b = create_mirror_view(b);

  // Get the member_type of the team policy
  typedef TeamPolicy<host_space>::member_type t_team;
  // Run a 2D copy kernel using a TeamPolicy
  parallel_for(TeamPolicy<host_space>(n,m),
    KOKKOS_LAMBDA (const t_team& t) {
    const int i = t.team_rank();
    parallel_for(TeamThreadRange(t,m), [&] (const int& j) {
      h_b(i,j) = b[i][j];
    });
  });
  // Copy the data from the host to the device
  deep_copy(d_b,h_b);
}
```

Alternatively one can create a view which directly references the external allocation. If that data is a multidimensional view, it is important to specify the Layout explicitly. Furthermore, all data must be part of the same allocation.

```c++
void MyKokkosFunction(int* a, const double* b, int n, int m) {
  // Define the host execution space and the view types
  typedef View<int*, DefaultHostExecutionSpace, MemoryTraits<Unmanaged>> t_1d_view;
  typedef View<double**[3],LayoutRight, DefaultHostExecutionSpace,
               MemoryTraits<Unmanaged>> t_3d_view;
  // Unmanaged views cannot have labels

  // Create a 1D view of the external allocation
  t_1d_view d_a(a,n);

  // Create a 3D view of the second external allocation
  // This assumes that the data had a row major layout (i.e. the third index is stride 1)
  t_3d_view d_b(b,n,m);
}
```

### Views as the fundamental data owning structure

Another option is to let Kokkos handle the basic allocations using Views and then construct the legacy data structures around them. Again, it is important to fix the Layout of the Views to whatever the layout of the legacy data was.

```c++
// Allocate a 2D view with row major layout
View<double**,LayoutRight,HostSpace> a("A",n,m);

// Allocate an array of pointers
double** a_old = new double*[n];

// Fill the array with pointers to the rows of a
for(int i=0; i<n; i++)
  a_old[i] = &a(i,0);
```

### std::vector

One of the most common data objects in C++ codes is `std::vector`. Its semantics are unfortunately not compatible with Kokkos requirements and it is thus not well supported. A major problem is that functors and lambdas are passed as const objects to the parallel execution. This design choice was made to (i) prevent a common cause of race conditions and (ii) allow the underlying implementation more flexibility in how to share the functor and where to put it. In particular, this leaves the choice open for the implementation to give each thread an individual copy of the functor or place it in read only cache structures.

The semantics of `std::vector` would in this case prevent a kernel from modifying its entries since a const `std::vector` is read only. Furthermore, creating multiple copies of the functor would indeed replicate the vector data, since it has copy semantics.

Other issues with `std::vector` are its unrestricted support for resize as well as push functionality. In a threaded environment support for those capabilities would bring massive performance penalties. In particular access to the `std::vector` would require locks in order to prevent one thread from deallocating the view while another accesses its content. And last but not least `std::vector` is not supported on GPUs and thus would prevent portability.

Kokkos provides a drop in replacement for `std::vector` with `Kokkos::vector`. Outside of parallel kernels its semantics are mostly the same as that of `std::vector`; for example assignments perform deep copies and resize and push functionality are provided. One important difference is that it is valid to assign values to the elements of a const vector.

Inside of parallel sections `Kokkos::vector` switches to view semantics. That means in particular that assignments are shallow copies. Certain functions will also throw runtime errors when called inside a parallel kernel; this includes resize and push.

```c++
// Create a vector of 1000 double elements
Kokkos::vector<double> v(1000);
// Create another vector as a copy of v;
// This allocates another 1000 doubles
Kokkos::vector<double> x = v;

parallel_for(1000, KOKKOS_LAMBDA (const int& i) {
   // Create a view of x; m and x will reference the same data.
   Kokkos::vector<double> m = x;
   x[i] = 2*i+1;
   v[i] = m[i] - 1;
});

// Now x contains the first 1000 uneven numbers
// v contains the first 1000 even numbers
```

## Calling non-Kokkos libraries

There are no restrictions on calling non-Kokkos libraries outside of parallel kernels. However, due to the polymorphic layouts of Kokkos views it is often required to test layouts for compatibility with third party libraries. The usual BLAS interface for example, expects matrices to be laid out in column major format (i.e. LayoutLeft in Kokkos). Furthermore, it is necessary to test that the library can access the memory space of the view.

```c++
template<class Scalar, class Device>
Scalar dot(View<const Scalar* , Device> a,
           View<const Scalar*, Device> b) {
// Check for Cuda memory and call cublas if true
#ifdef KOKKOS_HAVE_CUDA
  if(std::is_same<typename Device::memory_space,
                           CudaSpace>::value ||
     std::is_same<typename Device::memory_space,
                           CudaUVMSpace>::value) {
    return call_cublas_dot(a.ptr_on_device(), b.ptr_on_device(),
                           a.extent(0) );
  }
#endif

// Call CBlas on the host otherwise
  if(std::is_same<typename Device::memory_space,HostSpace>::value) {
    return call_cblas_dot(a.ptr_on_device(), b.ptr_on_device(),
                          a.extent(0) );
  }
}
```
