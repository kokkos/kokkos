---
name: kokkos
description: Use the Kokkos C++ parallel programming model. Use this skill for writing, editing, or debugging code that uses Kokkos.
license: MIT
compatibility: opencode
---

## What I do

I help with Kokkos C++ parallel programming: writing kernels (`parallel_for`, `parallel_reduce`, `parallel_scan`), working with Views, execution/memory spaces, and Kokkos containers and algorithms.

## When to use me

Use me whenever you're working with Kokkos — implementing new kernels, converting C++ to Kokkos, or debugging Kokkos code.

# Kokkos Guide

**Do not access external or online Kokkos documentation. All documentation is available in `references/`.**
When writing code, consult the relevant reference files to verify your code is correct before outputting,
unless you are completely certain it's already correct.

## Kokkos Conventions

These rules apply to every Kokkos file you write:

- Include only `<Kokkos_Core.hpp>` unless the documentation tells you otherwise.
- Remember to annotate kernel bodies and functions called there e.g. `KOKKOS_INLINE_FUNCTION`/`KOKKOS_LAMBDA`.
- Try to make code generic and portable. Template on View types, execution space or `Device<Exec, Mem>` unless the code is explicitly backend-specific.
- Never call host-only functions (e.g. C++ std library) from device code. Use Kokkos equivalents (`Kokkos::complex`, `Kokkos::pair`, math functions).
- Prefer idiomatic Kokkos: e.g. use `deep_copy(view, value)` to fill a view instead of a `parallel_for` loop.
- Ensure memory accessibility: host code cannot touch device memory and vice versa. Some memory spaces are accessible to both. Use mirror views for cross-space access.
- Always use `typename` to access member typedefs of Kokkos classes, e.g. `typename View<...>::memory_space`.

## Execution and Memory Spaces

Kokkos execution spaces define where kernels run; memory spaces define where data lives.

- **Execution spaces**: `DefaultExecutionSpace`, `DefaultHostExecutionSpace`, plus backend-specific ones (`Serial`, `OpenMP`, `Threads`, `Cuda`, `HIP`, `SYCL`, `HPX`). See `references/API/core/execution_spaces.rst`.
- **Memory spaces**: `HostSpace`, `CudaSpace`, `HIPSpace`, `SYCLDeviceUSMSpace`, UVM/managed spaces, host-pinned spaces. See `references/API/core/memory_spaces.rst`.
- Device type: `Kokkos::Device<exec_space, memory_space>`. Use `typename exec_space::memory_space` for the default memory space, `typename memory_space::execution_space` for the best exec space.
- `Kokkos::DefaultExecutionSpace` is the canonical default device type.
- Split an execution space for concurrent kernel dispatch with `Kokkos::Experimental::partition_space`. See `references/API/core/spaces/partition_space.rst`.
- Backend detection: `KOKKOS_ENABLE_<BACKEND>` (all-caps).
- Common pattern is to template functions and classes on execution space for portability.

## Parallel Kernels

All three (for/reduce/scan) take a function to call as the body. The body and functions called there must be annotated with
`KOKKOS_LAMBDA` (instead of a C++ lambda) or `KOKKOS_INLINE_FUNCTION` (for free or member function) to mark it as device
and host code appropriately. If launching a `KOKKOS_LAMBDA` kernel in a non-functor class member, special handling is
needed to safely capture `*this` by value. See `references/API/core/macros-special/host_device_macros.rst`.

Kernel body must be either `KOKKOS_LAMBDA(...)` declared inline, or a functor object whose `KOKKOS_INLINE_FUNCTION void operator()(...) const`
overload implements the body. The functor must hold necessary Views and other data as members. The `const` is required; the `operator()` cannot modify
any members (accessing data through a const or nonconst View is OK). A common pattern is for a functor to have a `run` method or similar that launches itself
in a parallel kernel with `*this` as the body argument. A functor can never launch itself using a `KOKKOS_LAMBDA` that captures `this` by pointer.

### parallel_for

Parallel loop over a 1D or multi-dimensional iteration space.
Pass optional string label as first argument to a to aid in debugging and profiling.

```cpp
Kokkos::parallel_for("name", Kokkos::RangePolicy<>(20, 100), KOKKOS_LAMBDA(const int i) {
    // body
});
```

- **RangePolicy** for 1D iteration. See `references/API/core/policies/RangePolicy.rst`.
- **MDRangePolicy** for 2D-8D. Requires `Rank<N>` template arg. See `references/API/core/policies/MDRangePolicy.rst`.
- Both optionally take an `ExecSpace` template param and/or an exec space instance as first argument.
- **RangePolicy and MDRangePolicy cannot be nested** inside another kernel. They can only be used for device-level kernels. Use `TeamPolicy` for nested parallelism, or use ordinary for loops.

### parallel_reduce

Parallel reduction. Body receives `(indices..., local_result&)` where `local_result` is the partial reduction to update.

```cpp
double sum;
Kokkos::parallel_reduce("reduce", policy,
    KOKKOS_LAMBDA(const int i, double& local_sum) {
        local_sum += data(i);
    }, sum);
```

- Default reducer is `Sum`. Wrap in `Kokkos::Max<T>(maxval)`, `Kokkos::Min<T>`, etc. for other reductions.
- Built-in reducers: `Sum`, `BAnd`, `BOr`, `LAnd`, `LOr`, `Max`, `Min`, `Prod`, `MinMax`, `MaxLoc`, `MinLoc`, `MinMaxLoc`, `FirstLoc`, `LastLoc`, `MaxFirstLoc`, `MinFirstLoc`, `MinMaxFirstLastLoc`. See `references/API/core/builtin_reducers.rst`.
- Multiple reducers can be passed. C arrays or Views can be reduced (size <= 16).
- Scalar results return to host immediately. If result is `View`, the kernel is non-blocking. Fence before accessing result outside of a `create_mirror_view_and_copy(Kokkos::HostSpace(), result)` before host access.
- The output result is always initialized by `parallel_reduce`; do not manually set it to the reduction identity before the call.

### parallel_scan

Parallel prefix sum. Body receives `(index, update&, const bool final)`. `final == true` indicates the final pass where
`update` holds the complete (exclusive) prefix. Must still perform the update during this pass.

```cpp
Kokkos::parallel_scan("scan", policy,
    KOKKOS_LAMBDA(const int i, double& update, const bool final) {
        if (final) output_exclusive[i] = update;
        update += input[i];
        if (final) output_inclusive[i] = update;
    });
```

See `references/API/core/parallel-dispatch/parallel_scan.rst`.

### Synchronization and Execution Space Instances

- Parallel kernels (except parallel_reduce with a scalar result) are non-blocking.
  - But run in sequence on a given execution space instance
  - A space instance is like Cuda stream: operations are sequenced within a stream and different streams are independent.
  - It is not possible to get the execution space instance of a given policy, so prefer to pass the instance around and construct policies where they are needed.
- `Kokkos::fence()` waits for *all* pending operations on *all* execution space instances.
- `exec.fence()` waits for specific execution space instance only.
- See `references/API/core/parallel-dispatch/fence.rst`.

## Kokkos Views

`Kokkos::View<DataType, Layout, Device, Traits>` — a multidimensional array with reference-counted shallow-copy semantics. See `references/API/core/view/view.rst`.

- **DataType**: element type + array specifiers (`*` = runtime dim, `[k]` = compile-time dim). Runtime dims must precede compile-time dims.
- **Layout**: `LayoutRight` (row-major), `LayoutLeft` (column-major), `LayoutStride` (arbitrary). See `references/API/core/view/layoutRight.rst`, `references/API/core/view/layoutLeft.rst`, `references/API/core/view/layoutStride.rst`.
- **Device**: memory space or `Device<Exec, Mem>`. Omitting it uses the default exec space's memory.
- **Traits**: `MemoryTraits<Unmanaged | Atomic | RandomAccess | Restrict | Aligned>`.
  - MemoryTraits has members `is_unmanaged`, `is_atomic`, etc.

### Compile-Time Type Info

| Member | Meaning |
|---|---|
| `VT::rank` | number of dimensions |
| `VT::value_type` | scalar type (const for non-const View) |
| `VT::non_const_value_type` | non-const scalar type |
| `VT::data_type` | full DataType with array specifiers |
| `VT::execution_space` | associated exec space |
| `VT::memory_space` | where data is stored |
| `VT::size_type` | index type |
| `Kokkos::is_view_v<T>` | true if arbitrary T is a View |

### Construction

```cpp
View<T**> v("name", extent0, extent1);                              // managed, zero-initialized
View<T*> v(Kokkos::view_alloc(Kokkos::WithoutInitializing, "name"), ext); // no init
View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> uv(raw_ptr, extent); // unmanaged
```

See `references/API/core/view/view_alloc.rst`.

### Access and Subviews

- Access with `operator()` and `extent(dim)`.
- `Kokkos::subview(v, idx_or_range_per_dim)` — shallow copy subset. Use `Kokkos::ALL()`, `Kokkos::pair(start, end)`, or single index. See `references/API/core/view/subview.rst`.

### deep_copy

```cpp
Kokkos::deep_copy(b, a);            // view to view
Kokkos::deep_copy(a, 1.0);          // fill with constant (preferred over parallel_for init)
Kokkos::deep_copy(exec, dest, src); // async with exec space instance
```

See `references/API/core/view/deep_copy.rst`.

### Resize / Realloc Views

- `Kokkos::resize(v, new_extents...)` — preserves content like `realloc` in C.
- `Kokkos::realloc(v, new_extents...)` — does not preserve content.
See `references/API/core/view/resize.rst`, `references/API/core/view/realloc.rst`.

### Mirror Views

Required whenever host code needs to access data that is (or might be) in non-host-accessible memory. 

```cpp
auto m1 = Kokkos::create_mirror(v);                                    // m1 is always a new allocation
auto m2 = Kokkos::create_mirror_view(v);                               // m2 == v if v is host-accessible, otherwise a new allocation
auto m3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v); // one-liner; m3's device type is explicitly HostSpace.
```

`create_mirror` always allocates new memory; `create_mirror_view` returns the original if already host-accessible.
`create_mirror` and `create_mirror_view` do not actually copy any data; use `deep_copy` for this.
See `references/API/core/view/create_mirror.rst`.

## Hierarchical Parallelism (TeamPolicy)

`TeamPolicy` provides 3 parallelism levels: a league of teams, a team of threads, and a thread of vector lanes.
See `references/API/core/policies/TeamPolicy.rst`.
`parallel_for` and `parallel_reduce` are available at these levels, plus TeamVector (last two levels).
Versions of RangePolicy and MDRangePolicy exist at all levels; see references/API/core/Execution-Policies.rst - section "Nested Execution Policies"

TeamPolicy kernels can also allocate Views scratch memory to be shared within team. Level 0 is small/fast like Cuda shared; level 1 is larger and slower.
Must configure scratch size in policy before launching. See references/ProgrammingGuide/HierarchicalParallelism.md section "Team scratch pad memory" for usage.

```cpp
using Pol = Kokkos::TeamPolicy<ExecSpace>;
// Create a policy with league_size teams, each with an auto-selected number of threads, and each thread with vector_length lanes.
// On GPU, these are analogous to Cuda: `gridDim.x`, `blockDim.y`, `blockDim.x`.
// On CPU, the ThreadVector level just uses for loops with an auto-vectorization hint.
Pol policy(league_size, Kokkos::AUTO, vector_length);
Kokkos::parallel_for("teams", policy, KOKKOS_LAMBDA(const Pol::member_type& team) {
    // Statements directly in the outermost level are executed by all lanes in the whole team.
    // Must use nested parallel for/reduce to distribute work to threads and lanes.
    // TeamThread: uses threads in the team
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](int i) {
      // ThreadVector: uses "vector lanes" in the thread
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, M), [&](int j) { /* ... */ });
    });
    team.team_barrier();
    // TeamVector: uses all vector lanes in the whole team. Flattens the innermost 2 levels.
    Kokkos::parallel_reduce (Kokkos::TeamVectorRange(team, N), [&](int i) { /* ... */ });
});
```

- `Kokkos::single(Kokkos::PerTeam(team), [=]{...})` runs exactly once per team; `PerThread` runs once per thread.
  - Use this to restrict side effectful statements
- Team-level reductions: `parallel_reduce(TeamThreadRange, ...)` does an all-reduce across the team. When writing back results to a single location, use `Kokkos::single`.

## Algorithms

- Parallel random number gen: Create `Random_XorShift64_Pool<Exec>` then call `fill_random(view, pool, min, max)` for uniform distribution.
  See `references/API/algorithms/Random-Number.rst` for other modes.
- Sorting: `Kokkos::sort(view)` for device-level. Team/thread-level sort in `references/API/algorithms/Sort.rst`.
- Kokkos replacements for C++ STL algorithms: `references/API/algorithms/std-algorithms-index.rst`.

## Math

- Math functions in Kokkos namespace: (e.g. `sqrt`, `max`): `references/API/core/numerics/mathematical-functions.rst`.
  - Always prefer these over `math.h` or `std::` functions in both host and device code.
- `Kokkos::complex<T>` — device-compatible complex number. See `references/API/core/numerics/complex.rst`. Always use this and not `std::complex` in Views or kernels. Can convert to/from Kokkos::complex for interop.
  - Kokkos math functions, parallel kernels, random and algorithms all have built-in support for `Kokkos::complex`.

## Initialize/finalize

- Initialize all Kokkos backends with `Kokkos::initialize` / `Kokkos::finalize`, or use `Kokkos::ScopeGuard` for RAII.
- Make sure Kokkos objects (Views etc) go out of scope before calling finalize.
- See `references/API/core/initialize_finalize/ScopeGuard.rst`.

