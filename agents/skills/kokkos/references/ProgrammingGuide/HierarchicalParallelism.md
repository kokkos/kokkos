# Hierarchical Parallelism

This chapter explains how to use Kokkos to exploit multiple levels of shared-memory parallelism. These levels include thread teams, threads within a team, and vector lanes. You may nest these levels of parallelism, and execute [`parallel_for()`](../API/core/parallel-dispatch/parallel_for), [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce), or [`parallel_scan()`](../API/core/parallel-dispatch/parallel_scan) at each level. The syntax differs only by the execution policy,
which is the first argument to the `parallel_*` operation. Kokkos also exposes a "scratch pad" memory which provides thread private and team private allocations.

## Motivation

Node architectures on modern high-performance computers are characterized by ever more _hierarchical parallelism_.
A level in the hierarchy is determined by the hardware resources which are shared between compute units at that level.
Higher levels in the hierarchy also have access to all resources in its branch at lower levels of the hierarchy.
This concept is orthogonal to the concept of heterogeneity. For example, a node in a typical CPU-based cluster consists of a number of multicore CPUs.  Each core supports one or more hyper-threads, and each hyper-thread can execute vector instructions. This means there are 4 levels in the hierarchy of parallelism:

1. CPU sockets share access to the same memory and network resources,
1. cores within a socket typically have a shared last level cache (LLC),
1. hyper-threads on the same core have access to a shared L1 (and L2) cache and they submit instructions to the same execution units, and
1. vector units execute a shared instruction on multiple data items.

GPU-based systems also have a hierarchy of 4 levels:

1. multiple GPUs in the same node share access to the same host memory and network resources,
1. core clusters (e.g. the SMs on an NVIDIA GPU) have a shared cache and access to the same high bandwidth memory on a single GPU,
1. threads running on the same core cluster have access to the same L1 cache and scratch memory and
1. they are grouped in so-called Warps or Wave Fronts within which threads are always synchronous and can collaborate more closely, for example via direct register swapping.

Kokkos provides a number of abstract levels of parallelism, which it maps to the appropriate hardware features. This mapping is not necessarily static or predefined; it may differ for each kernel. Furthermore, some mapping decisions happen at run time. This enables adaptive kernels which map work to different hardware resources depending on the work set size. While Kokkos provides defaults and suggestions, the optimal mapping can be algorithm dependent. Hierarchical parallelism is accessible through execution policies.

You should use Hierarchical Parallelism in particular in a number of cases:

1. Non-tightly nested loops: Hierarchical Parallelism allows you to expose more parallelism.
1. Data gather + reuse: If you gather data for a particular iteration of an outer loop, and then repeatably use it in an inner loop, Hierarchical Parallelism with scratch memory may match the use case well.
1. Force Cache Blocking: Using Hierarchical Parallelism forces a developer into algorithmic choices which are good for cache blocking. This can sometimes lead to better performing algorithms than a simple flat parallelism.

On the other hand you should probably not use Hierarchical Parallelism if you have tightly nested loops. For that use case, a Multidimensional Range Policy ([`Kokkos::MDRangePolicy`](./Multi-Dimensional-Parallelism)) is the better fit.

(HP_thread_teams)=
## Thread teams

Kokkos' most basic hierarchical parallelism concept is a thread team. A _thread team_ is a collection of threads which can synchronize and which share a "scratch pad" memory (see [Section 7.3](Team_scratch_pad_memory)).

Instead of mapping a 1-D range of indices to hardware resources, Kokkos' thread teams map a 2-D index range. The first index is the _league rank_, the index of the team. The second index is the _team rank_, the thread index within a team. In CUDA this is equivalent to launching a 1-D grid of 1-D blocks. The league size is arbitrary -- that is, it is only limited by the integer size type -- while the team size must fit in the hardware constraints. As in CUDA, only a limited number of teams are actually active at the same time, and they must run to completion before new ones are executed. Consequently, it is not valid to use inter thread-team synchronization mechanisms such as waits for events initiated by other thread teams.

### Creating a Policy instance

Kokkos exposes use of thread teams with the [`Kokkos::TeamPolicy`](../API/core/policies/TeamPolicy) execution policy. To use thread teams you need to create a [`Kokkos::TeamPolicy`](../API/core/policies/TeamPolicy) instance. It can be created inline for the parallel dispatch call. The constructors require two arguments: a league size and a team size. In place of the team size, a user can utilize `Kokkos::AUTO` to let Kokkos guess a good team size for a given architecture. Doing that is the recommended way for most developers to utilize the [`TeamPolicy`](../API/core/policies/TeamPolicy). As with the  [`Kokkos::RangePolicy`](../API/core/policies/RangePolicy) a specific execution tag, a specific execution space, a `Kokkos::IndexType`, and a `Kokkos::Schedule` can be given as optional template arguments.

```c++
// Using default execution space and launching
// a league with league_size teams with team_size threads each
Kokkos::TeamPolicy<>
        policy( league_size, team_size );

// Using  a specific execution space to
// run an n_workset parallelism with Kokkos choosing the team size
Kokkos::TeamPolicy<ExecutionSpace>
        policy( league_size, Kokkos::AUTO() );

// Using a specific execution space and an execution tag
Kokkos::TeamPolicy<SomeTag, ExecutionSpace>
        policy( league_size, team_size );
```

### Basic kernels

The team policy's `member_type` provides the necessary functionality to use teams within a parallel kernel. It allows access to thread identifiers such as the league rank and size, and the team rank and size. It also provides team-synchronous actions such as team barriers, reductions and scans.

```c++
using Kokkos::atomic_add;
using Kokkos::PerTeam;
using Kokkos::Sum;
using Kokkos::TeamPolicy;
using Kokkos::parallel_for;

typedef TeamPolicy<ExecutionSpace>::member_type member_type;
// Create an instance of the policy
TeamPolicy<ExecutionSpace> policy (league_size, Kokkos::AUTO() );
// Launch a kernel
parallel_for (policy, KOKKOS_LAMBDA (member_type team_member) {
    // Calculate a global thread id
    int k = team_member.league_rank () * team_member.team_size () +
            team_member.team_rank ();

    // Calculate the sum of the global thread ids of this team
    int team_sum = k;
    team_member.team_reduce(Sum<int, typename ExecutionSpace::memory_space>(team_sum));

    // Atomically add the computed sum to a global value
    Kokkos::single (PerTeam (team_member), [=] () {
      atomic_add(&global_value(), team_sum);
    });
  });
```

The name [`TeamPolicy`](../API/core/policies/TeamPolicy) makes it explicit that a kernel using it constitutes a parallel region with respect to the team.

In order to allow for coordination of work between members of a team, i.e. some threads compute a value, store it in global memory and then everyone consumes it, teams provide barriers. These barriers are collectives for all team members in the same team, but have no relationship with other teams. Here is an example:

```c++
using Kokkos::TeamPolicy;
using Kokkos::parallel_for;

typedef TeamPolicy<ExecutionSpace>::member_type member_type;
// Create an instance of the policy
TeamPolicy<ExecutionSpace> policy (league_size, Kokkos::AUTO() );
// Launch a kernel
parallel_for (policy, KOKKOS_LAMBDA (member_type team_member) {
    // Thread 0 in each team gathers some data via indirection.
    if( team_member.team_rank() == 0 ) {
      a(team_member.league_rank()) = b(indices(team_member.league_rank()));
    }
    // Now do a barrier for every team member to wait for a to be updated
    team_member.team_barrier();

    // Now a can be used by every team member
    c(team_member.league_rank(),team_member.team_rank()) = a(team_member.league_rank();
  });
```

(Team_scratch_pad_memory)=
## Team scratch pad memory

Each Kokkos team has a "scratch pad." This is an instance of a memory space accessible only by threads in that team. Scratch pads let an algorithm load a workset into a shared space and then collaboratively work on it with all members of a team. The lifetime of data in a scratch pad is the lifetime of the team. In particular, scratch pads are recycled by all logical teams running on the same physical set of cores. During the lifetime of the team all operations allowed on global memory are allowed on the scratch memory. This includes taking addresses and performing atomic operations on elements located in scratch space. Team-level scratch pads correspond to the per-block shared memory in Cuda, or to the "local store" memory on the Cell processor.

Kokkos exposes scratch pads through a special memory space associated with the execution space: `execution_space::scratch_memory_space`. You may allocate a chunk of scratch memory through the [`TeamPolicy`](../API/core/policies/TeamPolicy) member type. You may request multiple allocations from scratch, up to a user-provided maximum aggregate size. The maximum is provided either through a `team_shmem_size` function in the functor which returns a potentially team-size dependent value, or it can be specified through a setting of the TeamPolicy `set_scratch_size`. It is not valid to provide both values at the same time. The argument to the TeamPolicy can be used to set the shared memory size when using functors. One restriction on shared memory allocations is that they can not be freed during the lifetime of the team. This avoids the complexity of a memory pool, and reduces the time it takes to obtain an allocation (which currently is a few tens of integer operations to calculate the offset).

The following is an example of using the functor interface:

```c++
template<class ExecutionSpace>
struct functor {
  typedef ExecutionSpace execution_space;
  typedef execution_space::member_type member_type;

  KOKKOS_INLINE_FUNCTION
  void operator() (member_type team_member) const {
    size_t double_size = 5*team_member.team_size()*sizeof(double);

    // Get a shared team allocation on the scratch pad
    double* team_shared_a = (double*)
      team_member.team_shmem().get_shmem(double_size);

    // Get another allocation on the scratch pad
    int* team_shared_b = (int*)
      team_member.team_shmem().get_shmem(160*sizeof(int));

    // ... use the scratch allocations ...
  }

  // Provide the shared memory capacity.
  // This function takes the team_size as an argument,
  // which allows team_size dependent allocations.
  size_t team_shmem_size (int team_size) const {
    return sizeof(double)*5*team_size +
           sizeof(int)*160;
  }
};
```
The `set_scratch_size` function of the [`TeamPolicy`](../API/core/policies/TeamPolicy) takes two or three arguments. The first argument specifies the level in the scratch hierarchy for which a specific size is requested. Different levels have different restrictions. Generally, the first level is restricted to a few tens of kilobytes roughly corresponding to L1 cache size. The second level can be used to get an aggregate over all teams of a few gigabyte, corresponding to available space in high-bandwidth memory. The third level mostly falls back to capacity memory in the node. The second and third argument are either per-thread or per-team sizes for scratch memory.

Here are some examples:

```c++
TeamPolicy<> policy_1 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(1, PerTeam(1024), PerThread(32));
TeamPolicy<> policy_2 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(1, PerThread(32));
TeamPolicy<> policy_3 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(0, PerTeam(1024));
```

The total amount of scratch space available for each team will be the per-team value plus the per-thread value multiplied by the team-size. The interface allows users to specify those settings inline:

```c++
parallel_for(TeamPolicy<>(league_size, team_size).set_scratch_size(1, PerTeam(1024)),
  KOKKOS_LAMBDA (const TeamPolicy<>::member_type& team) {
    ...
});
```

Instead of simply getting raw allocations in memory, users can also allocate Views directly in scratch memory. This is achieved by providing the shared memory handle as the first argument of the View constructor. Views also have a static member function which return their shared memory size requirements. The function expects the run-time dimensions as arguments, corresponding to View's constructor. Note that the view must be unmanaged (i.e. it must have the `Unmanaged` memory trait).

```c++
typedef Kokkos::DefaultExecutionSpace::scratch_memory_space
  ScratchSpace;
// Define a view type in ScratchSpace
typedef Kokkos::View<int*[4],ScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_int_2d;

// Get the size of the shared memory allocation
size_t shared_size = shared_int_2d::shmem_size(team_size);
Kokkos::parallel_for(Kokkos::TeamPolicy<>(league_size,team_size).
                       set_scratch_size(0,Kokkos::PerTeam(shared_size)),
                     KOKKOS_LAMBDA ( member_type team_member) {
  // Get a view allocated in team shared memory.
  // The constructor takes the shared memory handle and the
  // runtime dimensions
  shared_int_2d A(team_member.team_scratch(0), team_member.team_size());
  ...
});
```

## Nested parallelism

Instead of writing code which explicitly uses league and team rank indices, one can use nested parallelism to implement hierarchical algorithms. Kokkos lets the user have up to three nested layers of parallelism. The team and thread levels are the first two levels. The third level is _vector_ parallelism.

You may use any of the three parallel patterns -- for, reduce, or scan -- at each level<sup>1</sup>.
You may nest them and use them in conjunction with code that is aware of the league and team rank. The different layers are accessible via special execution policies: `TeamThreadLoop` and `ThreadVectorLoop`.

***
<sup>1</sup> The parallel scan operation is not implemented for all execution spaces on the thread level, and it doesn't support a TeamPolicy on the top level.
***

### Team loops

The first nested level of parallel loops splits an index range over the threads of a team. This motivates the policy name [`TeamThreadRange`](../API/core/policies/TeamThreadRange), which indicates that the loop is executed once by the team with the index range split over threads. The loop count is not limited to the number of threads in a team, and how the index range is mapped to threads is architecture dependent. It is not legal to nest multiple parallel loops using the [`TeamThreadRange`](../API/core/policies/TeamThreadRange) policy. However, it is valid to have multiple parallel loops using the [`TeamThreadRange`](../API/core/policies/TeamThreadRange) policy follow each other in sequence, in the same kernel. Note that it is not legal to make a write access to POD data outside the closure of a nested parallel layer. This is a conscious choice to prevent difficult-to-debug issues related to thread private, team shared and globally shared variables. A simple way to enforce this is by using the "capture by value"' clause with lambdas,
but "capture by reference" is recommended for release builds since it typically results in better performance.
With the lambda being considered as `const` inside the [`TeamThreadRange`](../API/core/policies/TeamThreadRange) loop, the compiler will catch illegal accesses at compile time as a `const` violation.

The simplest use case is to have another [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) nested inside a kernel.

```c++
using Kokkos::parallel_for;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;

parallel_for (TeamPolicy<> (league_size, team_size),
                    KOKKOS_LAMBDA (member_type team_member)
{
  Scalar tmp;
  parallel_for (TeamThreadRange (team_member, loop_count),
    [=] (int& i) {
      // ...
      // tmp += i; // This would be an illegal access
    });
});
```

The [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce)  construct can be used to perform optimized team-level reductions:

```c++
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
parallel_for (TeamPolicy<> (league_size, team_size),
                 KOKKOS_LAMBDA (member_type team_member) {
    // The default reduction uses Scalar's += operator
    // to combine thread contributions.
    Scalar sum;
    parallel_reduce (TeamThreadRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum += ...;
      }, sum);

    // Introduce a team barrier here to synchronize threads
    team_member.team_barrier();

    // You may provide a custom reduction as a functor,
    // including one of the Kokkos-provided ones, e.g. Prod<Scalar>.
    Scalar product;
    Scalar init_value = 1;
    parallel_reduce (TeamThreadRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum *= ...;
      }, Kokkos::Experimental::Prod<Scalar>(product);
  });
```
Note that custom reductions must employ one of the functor join patterns recognized by Kokkos; these include `Sum, Prod, Min, Max, LAnd, LOr, BAnd, BOr, ValLocScalar, MinLoc, MaxLoc, MinMaxScalar, MinMax, MinMaxLocScalar, MinMaxLoc, FirstLoc, FirstLocScalar, LastLoc, LastLocScalar, MinFirstLoc, MaxFirstLoc` and `MinMaxFirstLastLoc`.

The third pattern is [`parallel_scan()`](../API/core/parallel-dispatch/parallel_scan) which can be used to perform prefix scans.

#### Team Barriers

In instances where one loop operation might need to be sequenced with a different loop operation, such as filling of arrays as a preparation stage for following computations on that data, it is important to be able to control threads in time; this can be done through the use of barriers. In nested loops, the outside loop ( [`TeamPolicy<> ()`](../API/core/policies/TeamPolicy) ) has a built-in (implicit) team barrier; inner loops ( [`TeamThreadRange ()`](../API/core/policies/TeamThreadRange) ) do not. This latter condition is often referred to as a 'non-blocking' condition. When necessary, an explicit barrier can be introduced to synchronize team threads; an example is shown in the previous example. 

### Vector loops

The innermost level of nesting parallel loops in a kernel comprises the _vector_-loop. Vector level parallelism works identically to the team level loops using the execution policy [`ThreadVectorRange`](../API/core/policies/ThreadVectorRange). In contrast to the team-level, there is no legal way to exploit the vector level outside a parallel pattern using the [`ThreadVectorRange`](../API/core/policies/ThreadVectorRange). However, one can use such a parallel construct in- and outside- of a [`TeamThreadRange`](../API/core/policies/TeamThreadRange) parallel operation.

```c++
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
parallel_for (TeamPolicy<> (league_size, team_size),
                 KOKKOS_LAMBDA (member_type team_member) {

    int k = team_member.team_rank();
    // The default reduction uses Scalar's += operator
    // to combine thread contributions.
    Scalar sum;
    parallel_reduce (ThreadVectorRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum += ...;
      }, sum);

    parallel_for (TeamThreadRange (team_member, workset_size),
      [&] (int& j) {
      // You may provide a custom reduction as a functor
      // including one of the Kokkos-provided ones, e.g., Prod<Scalar>.
      Scalar product;
      Scalar init_value = 1;
     parallel_reduce (ThreadVectorRange (team_member, loop_count),
        [=] (int& i, Scalar& lsum) {
          // ...
          lsum *= ...;
        }, Kokkos::Experimental::Prod<Scalar>(product);
      });
  });
```

As the name indicates the vector-level must be vectorizable. The parallel patterns will exploit available mechanisms to encourage vectorization by the compiler. When using the Intel compiler for example, the vector level loop will be internally decorated with `#pragma ivdep`, telling the compiler to ignore assumed vector dependencies.

### Restricting execution to a single executor

As stated above, a kernel is a parallel region with respect to threads (and vector lanes) within a team. This means that global memory accesses outside of the respective nested levels potentially have to be protected against repetitive execution. A common example is the case where a team performs some calculation but only one result per team has to be written back to global memory.

Kokkos provides the `Kokkos::single(Policy,Lambda)` function for this case. It currently accepts two policies:

* `Kokkos::PerTeam` restricts execution of the lambda's body to once per team
* `Kokkos::PerThread` restricts execution of the lambda's body to once per thread (that is, to only one vector lane in a thread)

The `single` function takes a lambda as its second argument. That lambda takes zero arguments or one argument by reference. If it takes no argument, its body must perform side effects in order to have an effect. If it takes one argument, the final value of that argument is broadcast to every executor on the level: i.e. every vector lane of the thread, or every thread (and vector lane) of the team. It must always be correct for the lambda to capture variables by value (`[=]`, not `[&]`). Thus, if the lambda captures by reference, it must _not_ modify variables that it has captured by reference.

```c++
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::PerThread;

TeamPolicy<...> policy (...);
typedef TeamPolicy<...>::member_type team_member;

parallel_for (policy, KOKKOS_LAMBDA (const team_member& thread) {
 // ...

  parallel_for (TeamThreadRange (thread, 100),
    KOKKOS_LAMBDA (const int& i) {
      double sum = 0;
      // Perform a vector reduction with a thread
      parallel_reduce (ThreadVectorRange (thread, 100),
        [=] (int i, double& lsum) {
          // ...
          lsum += ...;
      }, sum);
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (), [=] () {
          shared_array(i) += sum;
      });
  });

  double sum;
  parallel_reduce (TeamThreadRange (thread, 99),
    KOKKOS_LAMBDA (int i, double& lsum) {
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (thread), [=] () {
          lsum += someFunction (shared_array(i),
                                shared_array(i+1));
      });
  }, sum);

  // Add the per team contribution to global memory.
  Kokkos::single (PerTeam (thread), [=] () {
    global_array(thread.league_rank()) = sum;
  });
});
```

Here is an example of using the broadcast capabilities to determine the start offset for a team in a buffer.
All threads within the team will observe the same value for ``team_offset`` after the call to ``single``.

```c++
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::PerThread;

TeamPolicy<...> policy (...);
typedef TeamPolicy<...>::member_type team_member;

Kokkos::View<int> offset("Offset");
offset() = 0;

parallel_for (policy, KOKKOS_LAMBDA (const team_member& thread) {
  // ...

  parallel_reduce (TeamThreadRange (thread, 100),
    KOKKOS_LAMBDA (const int& i, int& lsum) {
      if(...) lsum++;
  });
  int team_offset;
  Kokkos::single (PerTeam (thread), [=] (int& my_offset) {
   my_offset = Kokkos::atomic_fetch_add(&offset(),lsum);
  }, team_offset);
  ...
});
```

To further illustrate the "parallel region" semantics of the team execution consider the following code:

```c++
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::TeamPolicy;

parallel_reduce(TeamPolicy<>(N,team_size),
  KOKKOS_LAMBDA (const member_type& teamMember, int& lsum) {
    int s = 0;
    for(int i = 0; i<10; i++) s++;
    lsum += s;
},sum);
```

In this example `sum` will contain the value `N * team_size * 10`. Every thread in each team will compute `s=10` and then contribute it to the sum.

Let's go one step further and add a nested [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce). By choosing the loop bound to be `team_size` every thread still only runs once through the inner loop.

```c++
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::TeamPolicy;

parallel_reduce(TeamPolicy<>(N,team_size),
  KOKKOS_LAMBDA (const member_type& teamMember, int& lsum) {

  int s = 0;
  parallel_reduce(TeamThreadRange(teamMember, team_size),
    [=] (const int k, int & inner_lsum) {
    int inner_s = 0;
    for(int i = 0; i<10; i++) inner_s++;
    inner_lsum += inner_s;
  },s);
  lsum += s;
},sum);
```

The answer in this case is nevertheless `N * team_size * team_size * 10`. Each thread computes `inner_s = 10`. But all threads in the team combine their results to compute a `s` value of `team_size * 10`. Since every thread in each team contributes that value to the global sum, we arrive at the final value of `N * team_size * team_size * 10`. If the intended goal was for each team to only contribute `s` once to the global sum, the contribution should have been protected with a `single` clause.
