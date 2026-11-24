# Atomic Operations

After reading this chapter, you should understand the following:

* Atomic operations can be used to resolve write conflicts
* How to use the free functions.
* How to use the Atomic memory trait.
* Using [ScatterView](../API/containers/ScatterView) for scatter add patterns

## Write Conflicts and Their Resolution With Atomic Operations

Consider the simple task of creating a histogram of a number set.

```c++
void create_histogram(View<int*> histogram, int min, int max, View<int*> values) {
  deep_copy(histogram,0);
  for(int i=0; i<values.extent(0); i++) {
    int index = (1.0*(values(i)-min)/(max-min)) * histogram.extent(0);
    histogram(index)++;
  }
}
```

When parallelizing this loop with a simple [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) multiple threads may try to
increment the same `index` at the same time. The increment on the other hand is actually
three operations:
  1. load `histogram(index)` into a register,
  2. increment the registers,
  3. store the register back to `&histogram(index)`

When two threads try to do this to the same index at the same time, it can happen that both
threads load the value, then increment and then store. Since both loaded the same original
value only one of the updates makes it through, while the second increment gets lost. This
is called a *race condition*.

Another typical situation for this situation are so called *scatter-add* algorithms.
For example in particle codes one often loops over all particles, and then for each particle
contribute something to each of its neighbours (such as a partial force). If two threads simultaneously
work on two particles with shared neighbours, they may race on contributing to the same neighbour particle.

```c++
void compute_force(View<int**> neighbours, View<double*> values) {
  parallel_for("ForceLoop", neighbors.extent(0), KOKKOS_LAMBDA (const int particle_id) {
    for(int i = 0; i < neighbours.extent(1); i++) {
      int neighbour_id = neighbours(i);
      // This contribution will potentially race with other threads
      values(neighbour_id) += some_contribution(...);
    }
  });
}
```

There are a number of approaches to resolve such situations: One can (i) apply colouring and run the algorithm multiple times in a way that no conflicts appear with the subset of each colour, (ii) replicate the output array for every thread, or (iii) use atomic operations. All of these come with disadvantages.

Colouring has the disadvantages that one has to create the sets. For the histogram example, the cost of creating the set is likely larger than the operation itself. Furthermore, since one has to run each colour separately, the total amount of memory transfer can be significantly larger, since you tend to loop through all the allocations multiple times while using only parts of each cache line.

Replicating the output array is a good strategy for low thread counts (2-8) but often tends to fall apart above that.

Atomic operations execute a whole logical operation uninterrupted. For example the load-modify-store cycle of the above examples will be executed with no other threads being able to access the modified library (via another atomic operation) until the atomic operation is finished. Note that non-atomic operations may still race with atomic operations. The disadvantage of atomic operation is that they hinder certain compiler optimizations, and the throughput of atomics may not be good depending on the architecture and the scalar type.

## Atomic Free Functions

Atomic free functions are functions which take a pointer to the to-be-updated value, plus the update. Every typical operation has its own atomic free function. The two example above would be like this:

```c++
void create_histogram(View<int*> histogram, int min, int max, View<int*> values) {
  deep_copy(histogram,0);
  parallel_for("CreateHistogram", values.extent(0), KOKKOS_LAMBDA(const int i) {
    int index = (1.0*(values(i)-min)/(max-min)) * histogram.extent(0);
    atomic_inc(&histogram(index));
  });
}
```
```c++
void compute_force(View<int**> neighbours, View<double*> values) {
  parallel_for("ForceLoop", neighbors.extent(0), KOKKOS_LAMBDA (const int particle_id) {
    for(int i = 0; i < neighbours.extent(1); i++) {
      int neighbour_id = neighbours(i);
      atomic_add(&values(neighbour_id), some_contribution(...));
    }
  });
}
```

There are also atomic operations which return the old or the new value. They follow the [`atomic_fetch_[op]`](../API/core/atomics/atomic_fetch_op) and [`atomic_[op]_fetch`](../API/core/atomics/atomic_op_fetch) naming scheme. For example if one would want to find all the indices of negative values in an array and store them in a list this would be the algorithm:
```c++
void find_indices(View<int*> indices, View<double*> values) {
  View<int> count("Count");
  parallel_for("FindIndices", values.extent(0), KOKKOS_LAMBDA(const int i) {
    if(values(i) < 0) {
      int index = atomic_fetch_add(&count(),1);
      indices(index) = i;
    }
  });
}
```

The full list of atomic operations can be found here:

| Name                                                                                  | Library                   | Category | Description                  |
|:--------------------------------------------------------------------------------------|:--------------------------|:-----------|:----------------------------|
| [atomic_compare_exchange](../API/core/atomics/atomic_compare_exchange)                | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value only if the old value matches a comparison value and returns the old value. |
| [atomic_exchange](../API/core/atomics/atomic_exchange)                                | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value and returns the old. |
| [atomic_fetch_\[op\]](../API/core/atomics/atomic_fetch_op)                            | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the old value. [op] might be `add`, `and`, `div`, `lshift`, `max`, `min`, `mod`, `mul`, `or`, `rshift`, `sub` or `xor` |
| [atomic_load](../API/core/atomics/atomic_load)                                        | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Atomic operation which loads a value. |
| [atomic_\[op\]](../API/core/atomics/atomic_op)                                        | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Atomic operation which don't return anything. [op] might be `and`, `add`, `dec`, `max`, `min`, `inc`, `or` or `sub` |
| [atomic_\[op\]_fetch](../API/core/atomics/atomic_op_fetch)                            | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the updated value. [op] might be `add`, `and`, `div`, `lshift`, `max`, `min`, `mod`, `mul`, `or`, `rshift`, `sub` or `xor` |
| [atomic_store](../API/core/atomics/atomic_store)                                      | [Core](../API/core-index) | [Atomic-Operations](Atomic-Operations) | Atomic operation which stores a value. |

## Atomic Memory Trait

If all operations on a specific `View` during a Kernel are atomic one can also use the atomic memory trait.
Generally one creates an *atomic* `View` from a *non-atomic* `View` just for the one kernel, and then uses normal
operations on it.

```c++
void create_histogram(View<int*> histogram, int min, int max, View<int*> values) {
  deep_copy(histogram,0);
  View<int*,MemoryTraits<Atomic> > histogram_atomic = histogram;
  parallel_for("CreateHistogram", values.extent(0), KOKKOS_LAMBDA(const int i) {
    int index = (1.0*(values(i)-min)/(max-min)) * histogram.extent(0);
    histogram_atomic(index)++;
  });
}
```

## ScatterView

On CPUs one often uses low thread counts, in particular if Kokkos is used in conjunction with MPI.
In such situations data replication often performs better than using atomic operations.
In order to still have portable code, one can use the [`ScatterView`](../API/containers/ScatterView). It allows the transparent switch at
compile time from using atomic operations to using data replication depending on the underlying hardware.

A full description can be found here: [ScatterView](../API/containers/ScatterView)
