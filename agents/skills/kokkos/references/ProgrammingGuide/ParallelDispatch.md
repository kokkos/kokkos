# Parallel dispatch

You probably started reading this Guide because you wanted to learn how Kokkos can parallelize your code. This chapter will teach you different kinds of parallel operations that Kokkos can execute. We call these operations collectively _parallel dispatch_, because Kokkos "dispatches" them for execution by a particular execution space. Kokkos provides three different parallel operations:

* [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) implements a "for loop" with independent iterations.
* [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce) implements a reduction.
* [`parallel_scan()`](../API/core/parallel-dispatch/parallel_scan) implements a prefix scan.

Kokkos gives users two options for defining the body of a parallel loop: functors and lambdas. It also lets users control how the parallel operation executes, by specifying an _execution policy_. Later chapters will cover more advanced execution policies that allow nested parallelism.

Important notes on syntax:

* Use the `KOKKOS_INLINE_FUNCTION` macro to mark a   functor's methods that Kokkos will call in parallel
* Use the `KOKKOS_LAMBDA` macro to replace a lambda's capture clause when giving the lambda to Kokkos for parallel
  execution

## Specifying the parallel loop body

### Functors

A _functor_ is one way to define the body of a parallel loop. It is a class or struct<sup>1</sup> with a public `operator()` instance method. That method's arguments depend on both which parallel operation you want to execute (for, reduce, or scan), and on the loop's execution policy (e.g., range or team). For an example of a functor see the section in this chapter for each type of parallel operation. In the most common case of a [`parallel_for()`](../API/core/parallel-dispatch/parallel_for), it takes an integer argument which is the for loop's index. Other arguments are possible; see [Hierarchical Parallelism](HierarchicalParallelism).

The `operator()` method must be const, and must be marked with the `KOKKOS_FUNCTION` or `KOKKOS_INLINE_FUNCTION` macro. For some backends (such as CUDA and HIP) this macro is necessary to mark your method as suitable for running on both accelerator devices and the host. If not building with any backends requiring markup, `KOKKOS_INLINE_FUNCTION` expands to `inline`, and `KOKKOS_FUNCTION` is unnecessary but harmless. Here is an example of the signature of such a method:

```c++
KOKKOS_INLINE_FUNCTION void operator() (...) const;
```

The entire parallel operation (for, reduce, or scan) shares the same instance of the functor. However, any variables declared inside the `operator()` method are local to that iteration of the parallel loop. Kokkos may pass the functor instance by "copy," not by pointer or reference, to the execution space that executes the code. In particular, the functor might need to be copied to a different execution space than the host. For this reason, it is generally not valid to have any pointer or reference members in the functor. Pass in Kokkos Views by copy as well; this works by shallow copy. The functor is also passed as a const object, so it is not valid to change members of the functors. (However, it is valid for the functor to change the contents of, for example, a View or a raw array which is a member of the functor.)

***
<sup>1</sup>  A "struct" in C++ is just a class, all of whose members are public by default.
***

### Lambdas

The 2011 version of the C++ standard ("C++11") provides a new language construct, the _lambda_, also called "anonymous function" or "closure." Kokkos lets users supply parallel loop bodies as either functors (see above) or lambdas. Lambdas work like automatically generated functors. Just like a class, a lambda may have state.  The only difference is that with a lambda, the state comes in from the environment. (The name "closure" means that the function "closes over" state from the environment.) Just like with functors, lambdas must bring in state by "value" (copy), not by reference or pointer.

By default, lambdas capture nothing (as the default capture specifier `[]` specifies). This is not likely to be useful, since [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) generally works by its side effects. Because Kokkos reserves the right to make copies of the closure, and its operations are potentially asynchronous users must "capture by value" to be semantically correct. We recommend doing so via the KOKKOS_LAMBDA macro for the outermost level of parallelism (see [Hierarchical Parallelism](HierarchicalParallelism)).
For some backends, this just turns into the usual capture-by-value clause `[=]`. That captures variables from the surrounding scope by value. Do NOT capture them by reference! For other backends (e.g. CUDA and HIP), this macro may have a special definition
that makes the lambda work correctly, same as the `KOKKOS_INLINE_FUNCTION` macro. 

It is a violation of Kokkos semantics to capture by reference `[&]` for two reasons. First Kokkos might give the lambda to an execution space which can not access the stack of the dispatching thread. Secondly, capturing by reference allows the programmer to violate the const semantics of the lambda. For correctness and portability reasons lambdas and functors are treated as const objects inside the parallel code section. Capturing by reference allows a circumvention of that const property, and enables many more possibilities of writing non-threads-safe code.

When using lambdas for nested parallelism (see [Hierarchical Parallelism](HierarchicalParallelism) for details) using capture by reference can be useful for performance reasons, but the code is only valid Kokkos code if it also works with capturing by copy.

### Should I use a functor or a lambda?

Kokkos lets users choose whether to use a functor or a lambda. Lambdas are convenient for short loop bodies. For a much more complicated loop body, you might find it easier for testing to separate it out and name it as a functor. Lambdas by definition are "anonymous functions," meaning that they have no name. This makes it harder to test them. Furthermore, certain restrictions regarding lambdas apply depending on the backend [^footnotecudalambda].

[^footnotecudalambda]: Current restrictions for CUDA lambdas are explained at https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambda-restrictions.

Finally, the "execution tag" feature, which lets you put together several different parallel loop bodies into a single functor, only works with functors.  (See [Hierarchical Parallelism](HierarchicalParallelism) for details.)

### Specifying the execution space

If a functor has an `execution_space` public typedef, a parallel dispatch will only run the functor in that execution space. That's a good way to mark a functor as specific to an execution space. If the functor lacks this typedef, [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) will run it in the default execution space unless you tell it otherwise (that's an advanced topic; see "execution policies"). Lambdas do not have typedefs, so they run on the default execution space unless you tell Kokkos otherwise.

## Parallel for

The most common parallel dispatch operation is a [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) call. It corresponds to the OpenMP construct `#pragma omp parallel for`. [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) splits the index range over the available hardware resources and executes the loop body in parallel. Each iteration is executed independently. Kokkos promises nothing about the loop order or the amount of work which actually runs concurrently. This means in particular that not all loop iterations are active at the same time. Consequently, it is not legal to use wait constructs (e.g., wait for a prior iteration to finish). Kokkos also doesn't guarantee that it will use all available parallelism. For example, it can decide to execute in serial if the loop count is very small, and it would typically be faster to run in serial instead of introducing parallelization overhead. The `RangePolicy` allows you to specify minimal chunk sizes in order to control potential concurrency for low trip count loops.

The lambda or the `operator()` method of the functor takes one argument. That argument is the parallel loop "index." The type of the index depends on the execution policy used for the [`parallel_for()`](../API/core/parallel-dispatch/parallel_for). It is an integer type for the implicit or explicit [`RangePolicy`](../API/core/policies/RangePolicy). The former is used if the first argument to [`parallel_for()`](../API/core/parallel-dispatch/parallel_for) is an integer.

## Parallel reduce

Kokkos' [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce) operation implements a reduction. It is like [`parallel_for()`](../API/core/parallel-dispatch/parallel_for), except that each iteration produces a value and these iteration values are accumulated into a single value with a user-specified associative binary operation. It corresponds to the OpenMP construct `#pragma omp parallel reduction` but with fewer restrictions on the reduction operation.

In addition to the execution policy and the functor, [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce) takes an additional argument which is either the place where the final reduction result is stored (a simple scalar, or a [`Kokkos::View`](../API/core/view/view)) or a reducer argument which encapsulates both the place where to store the final result as well as the type of reduction operation desired (see [Custom Reductions](Custom-Reductions)). 

The lambda or the `operator()` method of the functor takes two arguments. The first argument is the parallel loop "index," the type of which depends on the execution policy used for the [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce). For example: when calling [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce) with a [`RangePolicy`](../API/core/policies/RangePolicy), the first argument to the operator is an integer type, but if you call it with a [`TeamPolicy`](../API/core/policies/TeamPolicy) the first argument is a *team handle*. The second argument is a non-const reference to a thread-local variable of the same type as the reduction result.

When not providing a `reducer` the reduction is performed with a sum reduction using the + or += operator of the scalar type. Custom reduction can also be implemented by providing a functor with a `join` and an `init` function. 

### Example using lambda

Here is an example reduction using a lambda, where the reduction result is a `double`.

```c++
const size_t N = ...;
View<double*> x ("x", N);
// ... fill x with some numbers ...
double sum = 0.0;
// KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
parallel_reduce ("Reduction", N, KOKKOS_LAMBDA (const int i, double& update) {
  update += x(i); 
}, sum);
```

### Example using functor with `join` and `init`.

The following example shows a reduction using the _max-plus semiring_, where `max(a,b)` corresponds to addition and ordinary addition corresponds to multiplication:

```c++
class MaxPlus {
public:
  // Kokkos reduction functors need the value_type typedef.
  // This is the type of the result of the reduction.
  using value_type = double;

  // Just like with parallel_for functors, you may specify
  // an execution_space typedef. If not provided, Kokkos
  // will use the default execution space by default.

  // Since we're using a functor instead of a lambda,
  // the functor's constructor must do the work of capturing
  // the Views needed for the reduction.
  MaxPlus (const View<double*>& x) : x_ (x) {}

  // This is helpful for determining the right index type,
  // especially if you expect to need a 64-bit index.
  using size_type = View<double*>::size_type;

  KOKKOS_INLINE_FUNCTION void
  operator() (const size_type i, value_type& update) const
  { // max-plus semiring equivalent of "plus"
    if (update < x_(i)) {
      update = x_(i);
    }
  }

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above.
  KOKKOS_INLINE_FUNCTION void
  join (value_type& dst,
        const value_type& src) const
  { // max-plus semiring equivalent of "plus"
    if (dst < src) {
      dst = src;
    }
  }

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION void
  init (value_type& dst) const
  { // The identity under max is -Inf.
     dst = reduction_identity<value_type>::max();
  }

private:
  View<double*> x_;
};
```

This example shows how to use the above functor:

```c++
const size_t N = ...;
View<double*> x ("x", N);
// ... fill x with some numbers ...

double result;
parallel_reduce ("Reduction", N, MaxPlus (x), result);
```

### Reductions with an array of results

Kokkos lets you compute reductions with an array of reduction results, as long as that array has a (run-time) constant number of entries. This currently only works with functors. Here is an example functor that computes column sums of a 2-D View.

```c++
struct ColumnSums {
  // In this case, the reduction result is an array of float.
  using value_type = float[];

  using size_type = View<float**>::size_type;

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  size_type value_count;

  View<float**> X_;

  // As with the above examples, you may supply an
  // execution_space typedef. If not supplied, Kokkos
  // will use the default execution space for this functor.

  // Be sure to set value_count in the constructor.
  ColumnSums (const View<float**>& X) :
    value_count (X.extent(1)), // # columns in X
    X_ (X)
  {}

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void
  operator() (const size_type i, value_type sum) const {
    // You may find it helpful to put pragmas above this loop
    // to convince the compiler to vectorize it. This is 
    // probably only helpful if the View type has LayoutRight.
    for (size_type j = 0; j < value_count; ++j) {
      sum[j] += X_(i, j);
    }
  }

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void
  join (value_type dst,
        const value_type src) const {
    for (size_type j = 0; j < value_count; ++j) {
      dst[j] += src[j];
    }
  }

  KOKKOS_INLINE_FUNCTION void init (value_type sum) const {
    for (size_type j = 0; j < value_count; ++j) {
      sum[j] = 0.0;
    }
  }
};
```

We show how to use this functor here. The results are
stored in the 1D View `sums`.
```c++
const size_t numRows = 10000;
const size_t numCols = 10;

View<float**> X ("X", numRows, numCols);
// ... fill X before the following ...
ColumnSums cs (X);
Kokkos::View<float*> sums ("sums", numCols);
parallel_reduce (X.extent(0), cs, sums);
```

The result view could also use `Kokkos::HostSpace`, in which case
accessing the results on the host requires a fence:

```c++
Kokkos::View<float*, Kokkos::HostSpace> sums ("sums", numCols);
parallel_reduce (X.extent(0), cs, sums);
Kokkos::fence();
std::cout << sums(0) << '\n';
```

If the number of elements in the reduced array is a compile-time constant,
it is also possible to place the results directly into a C array:
```c++
float sums[10];
parallel_reduce (X.extent(0), cs, sums);
```

## Parallel scan

Kokkos' [`parallel_scan()`](../API/core/parallel-dispatch/parallel_scan) operation implements a _prefix scan_. A prefix scan is like a reduction over a 1-D array, but it also stores all intermediate results ("partial sums"). It can use any associative binary operator. The default is `operator+` and we call a scan with that operator a "sum scan" if we need to distinguish it from scans with other operators. The scan operation comes in two variants. An _exclusive scan_ excludes (hence the name) the i<sup>th</sup> entry of the input array while computing the i<sup>th</sup> prefix scan and an _inclusive scan_ includes that entry. Given an example array `(1, 2, 3, 4, 5)`, an exclusive sum scan overwrites the array with `(0, 1, 3, 6, 10)`, and an inclusive sum scan overwrites the array with `(1, 3, 6, 10, 15)`.

Many operations that "look" sequential can be parallelized with a scan. To learn more, refer to Guy Blelloch's book<sup>2</sup> (a version of his PhD dissertation).

Kokkos lets users specify a scan by either a functor or a lambda. Both look like their [`parallel_reduce()`](../API/core/parallel-dispatch/parallel_reduce) equivalents, except that the `operator()` method or lambda takes three arguments: the loop index, the "update" value by non const reference, and a `bool`. Here is a lambda example where the intermediate results have type `float`.

```c++
View<float*> x = ...; // assume filled with input values
const size_t N = x.extent(0);
parallel_scan (N, KOKKOS_LAMBDA (const int i,
          float& update, const bool final) {
    // Load old value in case we update it before accumulating
    const float val_i = x(i); 
    if (final) {
      x(i) = update; // only update array on final pass
    }
    // For exclusive scan, change the update value after
    // updating array, like we do here. For inclusive scan,
    // change the update value before updating array.
    update += val_i;
  });
```

Kokkos may use a multiple-pass algorithm to implement scan. This means that it may call your `operator()` or lambda multiple times per loop index value. The `final` Boolean argument tells you whether Kokkos is on the final pass. You must only update the array on the final pass.

For an exclusive scan, change the `update` value after updating the array, as in the above example. For an inclusive scan, change `update` _before_ updating the array. Just as with reductions, your functor may need to specify a non-default `join` or `init` method if the defaults do not do what you want.

***
<sup>2</sup>  Blelloch, Guy, _Vector Models for Data-Parallel Computing_, The MIT Press, 1990.
***

## Function Name Tags

When writing class-based applications it is often useful to make the classes themselves functors. Using that approach allows the kernels to access all other class members, both data and functions. An issue coming up in that case is the necessity for multiple parallel kernels in the same class. Kokkos supports that through function name tags. An application can use optional (unused) first arguments to differentiate multiple operators in the same class. Execution policies can take the type of that argument as an optional template parameter. The same applies to init, join and final functions.

```c++
class Foo {
  struct BarTag {};
  struct RabTag {};

  void compute() {
     Kokkos::parallel_for(RangePolicy<BarTag>(0, 100), *this);
     Kokkos::parallel_for(RangePolicy<RabTag>(0, 1000), *this);
  }

 KOKKOS_FUNCTION
  void operator() (BarTag, const int i) const {
    ...
    foobar();
    ...
  }

  KOKKOS_FUNCTION
  void operator() (RabTag, const int i) const {
    ...
    foobar();
    ...
  }

  KOKKOS_FUNCTION
  void foobar() {
    ...
  }
};
```

This approach can also be used to template the operators by templating the tag classes which is useful to enable compile time evaluation of appropriate conditionals.
