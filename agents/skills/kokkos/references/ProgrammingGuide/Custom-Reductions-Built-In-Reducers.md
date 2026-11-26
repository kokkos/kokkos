# Built-In-Reducers

Kokkos provides Reducers for the most common reduction types:
* [BAnd](../API/core/builtinreducers/BAnd): Do a binary “and” reduction
* [BOr](../API/core/builtinreducers/BOr): Do a binary “or” reduction
* [LAnd](../API/core/builtinreducers/LAnd): Do a logical “and” reduction
* [LOr](../API/core/builtinreducers/LOr): Do a logical “or” reduction
* [Max](../API/core/builtinreducers/Max): Finding the maximum value
* [MaxLoc](../API/core/builtinreducers/MaxLoc): Retrieve the maximum value as well as its associated index
* [Min](../API/core/builtinreducers/Min): Finding the minimum value
* [MinLoc](../API/core/builtinreducers/MinLoc): Retrieve the minimum value as well as its associated index
* [MinMax](../API/core/builtinreducers/MinMax): Finding the minimum and the maximum value
* [MinMaxLoc](../API/core/builtinreducers/MinMaxLoc): Find both the maximum and minimum value as well as their associated indices
* [Prod](../API/core/builtinreducers/Prod): Computing the product of all input values
* [Sum](../API/core/builtinreducers/Sum): For simple Summations

These reducers work only for scalar data, i.e. you can’t have a runtime length array as the reduction type (for example finding the minimum values for each vector in a multi vector concurrently).
Generally the Reducers are templated on the Scalar type for the reduction as well as an optional template parameter for the memory space of the result (more on that later). The [`MinLoc`](../API/core/builtinreducers/MinLoc), [`MaxLoc`](../API/core/builtinreducers/MaxLoc) and [`MinMaxLoc`](../API/core/builtinreducers/MinMaxLoc) reducers are additionally templated on the index type. 

The following is an example for doing a simple min-reduction, finding the minimal value in a discretization of a parable.

```c++
double min;

Kokkos::parallel_reduce( "MinReduce", N, KOKKOS_LAMBDA (const int& x, double& lmin) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  if( val < lmin ) lmin = val; 
}, Kokkos::Min<double>(min));

printf("Min: %lf\n", min);
```

In this example the [`Min`](../API/core/builtinreducers/Min) reducer was templated on the reducing type `double` and the variable to store the result was taken in by reference. Note that the reducer is only used to combine values from different threads. The per thread reduction is still performed explicitly. One could have used the reducer for that as well through a reducer instance:

```c++
double min;

Kokkos::Min<double> min_reducer(min);
Kokkos::parallel_reduce( "MinReduce", N, KOKKOS_LAMBDA (const int& x, double& lmin) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  min_reducer.join(lmin, val); 
}, min_reducer);

printf("Min: %lf\n", min);
```

For the [`MinLoc`](../API/core/builtinreducers/MinLoc), [`MaxLoc`](../API/core/builtinreducers/MaxLoc) and [`MinMaxLoc`](../API/core/builtinreducers/MinMaxLoc) reducers the reduction type is a complex scalar type which is accessible through a `value_type` typedef. 
[`MinLoc`](../API/core/builtinreducers/MinLoc) and [`MaxLoc`](../API/core/builtinreducers/MaxLoc) have value types which contain a `val` and `loc` member to store the reduction value and the index respectively. Note that index (`loc`) can be a struct itself, for example to store a multidimensional index result (see later). 

```c++
typedef Kokkos::MinLoc<double,int>::value_type minloc_type;
minloc_type minloc;

Kokkos::parallel_reduce( "MinLocReduce", N, KOKKOS_LAMBDA (const int& x, minloc_type& lminloc) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = x; }
}, Kokkos::MinLoc<double,int>(minloc));

printf("Min: %lf at %i\n", minloc.val, minloc.loc);
```

Reducers can be used in nested reductions. This example also makes use of a 2D index type to find the minimum and maximum value of a matrix as well as their indices. 

```c++
Kokkos::View<double**> A("A",N,M);
// fill A

// Create a variable for the result
typedef Kokkos::MinMaxLoc<double, Kokkos::pair<int,int>> reducer_type;
typedef reducer_type::value_type value_type;
value_type minmaxloc

typedef Kokkos::TeamPolicy<>::member_type team_type;

// Start a team parallel reduce
Kokkos::parallel_reduce( "MinLocReduce", Kokkos::TeamPolicy<>(N,AUTO), 
    KOKKOS_LAMBDA (const team_type& team, value_type& team_minmaxloc) {

  // Create a temporary to store the reduction value for the row
  value_type row_minmaxloc;
  int n = team.league_rank();

  // Run a nested parallel reduce with the team over the row
  Kokkos::parallel_reduce( Kokkos::TeamThreadRange(team, M), 
      [=] (const int& m, value_type& thread_minmaxloc) {
    double val = A(n,m);
    
    // Check whether this is a new minimum or maximum value
    if(val < thread_minmaxloc.min_val) {
      thread_minmaxloc.min_val = val;
      thread_minmaxloc.min_loc = Kokkos::pair<int,int>(n,m);
    }
    if(val > thread_minmaxloc.max_val) {
      thread_minmaxloc.max_val = val;
      thread_minmaxloc.max_loc = Kokkos::pair<int,int>(n,m);
    }

  }, reducer_type(row_minmaxloc));
  
  // One guy in the team should contribute to the whole
  // Note: for a min or max reduction it wouldn't hurt if 
  //       every team member did this
  Kokkos::single(Kokkos::PerTeam(team), [=] () {
    if( row_minmaxloc.min_val < team_minmaxloc.min_val ) {
      team_minmaxloc.min_val = row_minmaxloc.min_val;
      team_minmaxloc.min_loc = row_minmaxloc.min_loc;
    }
    if( row_minmaxloc.max_val > team_minmax.max_val ) {
      team_minmaxloc.max_val = row_minmaxloc.max_val;
      team_minmaxloc.max_loc = row_minmaxloc.max_loc;
    }
  }
}, reducer_type(minmaxloc));

printf("Min %lf at (%i, %i)\n",minmaxloc.min_val, minmaxloc.min_loc.first, minmaxloc.min_loc.second);
printf("Max %lf at (%i, %i)\n",minmaxloc.max_val, minmaxloc.max_loc.first, minmaxloc.max_loc.second);
```
