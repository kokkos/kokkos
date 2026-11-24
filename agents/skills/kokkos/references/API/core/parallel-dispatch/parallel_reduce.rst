``parallel_reduce``
===================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::parallel_reduce(name, policy, functor, reducer...);
    Kokkos::parallel_reduce(name, policy, functor, result...);
    Kokkos::parallel_reduce(name, policy, functor);
    Kokkos::parallel_reduce(policy, functor, reducer...);
    Kokkos::parallel_reduce(policy, functor, result...);
    Kokkos::parallel_reduce(policy, functor);

Dispatches parallel work defined by ``functor`` according to the *ExecutionPolicy* and performs a reduction of the contributions provided by workers as defined by the execution policy. The optional label name is used by profiling and debugging tools. The reduction type is either a ``sum``, is defined by the ``reducer`` or is deduced from an optional ``join`` operator on the functor. The reduction result is stored in ``result``, or through the ``reducer`` handle. It is also provided to the ``functor.final()`` function if such a function exists. Multiple ``reducers`` can be used in a single ``parallel_reduce`` and thus, it is possible to compute the ``min`` and the ``max`` values in a single ``parallel_reduce``.

Interface
---------

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType>
    Kokkos::parallel_reduce(const std::string& name,
                            const ExecPolicy& policy,
                            const FunctorType& functor);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType>
    Kokkos::parallel_reduce(const ExecPolicy& policy,
                            const FunctorType& functor);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class... ReducerArgument>
    Kokkos::parallel_reduce(const std::string& name,
                            const ExecPolicy& policy,
                            const FunctorType& functor,
                            const ReducerArgument&... reducer);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class... ReducerArgument>
    Kokkos::parallel_reduce(const ExecPolicy& policy,
                            const FunctorType& functor,
                            const ReducerArgument&... reducer);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class... ReducerArgumentNonConst>
    Kokkos::parallel_reduce(const std::string& name,
                            const ExecPolicy& policy,
                            const FunctorType& functor,
                            ReducerArgumentNonConst&... reducer);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class... ReducerArgumentNonConst>
    Kokkos::parallel_reduce(const ExecPolicy& policy,
                            const FunctorType& functor,
                            ReducerArgumentNonConst&... reducer);

Parameters:
~~~~~~~~~~~

* ``name``: A user provided string which is used in profiling and debugging tools via the Kokkos Profiling Hooks.
* ExecPolicy: An *ExecutionPolicy* which defines iteration space and other execution properties. Valid policies are:

  - ``IntegerType``: defines a 1D iteration range, starting from 0 and going to a count.
  - `RangePolicy <../policies/RangePolicy.rst>`_: defines a 1D iteration range.
  - `MDRangePolicy <../policies/MDRangePolicy.rst>`_: defines a multi-dimensional iteration space.
  - `TeamPolicy <../policies/TeamPolicy.rst>`_: defines a 1D iteration range, each of which is assigned to a thread team.
  - `TeamVectorRange <../policies/TeamVectorRange.rst>`_: defines a 1D iteration range to be executed by a thread-team. Only valid inside a parallel region executed through a ``TeamPolicy``.
  - `TeamVectorMDRange <../policies/TeamVectorMDRange.rst>`_: defines a multi-dimensional iteration space to be executed by a thread-team. Only valid inside a parallel region executed through a ``TeamPolicy``.
  - `TeamThreadRange <../policies/TeamThreadRange.rst>`_: defines a 1D iteration range to be executed by a thread-team. Only valid inside a parallel region executed through a ``TeamPolicy``.
  - `TeamThreadMDRange <../policies/TeamThreadMDRange.rst>`_: defines a multi-dimensional iteration space to be executed by a thread-team. Only valid inside a parallel region executed through a ``TeamPolicy``.
  - `ThreadVectorRange <../policies/ThreadVectorRange.rst>`_: defines a 1D iteration range to be executed through vector parallelization dividing the threads within a team.  Only valid inside a parallel region executed through a ``TeamPolicy``.
  - `ThreadVectorMDRange <../policies/ThreadVectorMDRange.rst>`_: defines a multi-dimensional iteration space to be executed through vector parallelization dividing the threads within a team.  Only valid inside a parallel region executed through a ``TeamPolicy``.
* FunctorType: A valid functor with (at minimum) an ``operator()`` with a matching signature for the ``ExecPolicy`` combined with the reduced type.
* ReducerArgument: Either a class fulfilling the "Reducer" concept or a ``Kokkos::View``.
* ReducerArgumentNonConst: A scalar type or an array type; see below for functor requirements.

Requirements:
~~~~~~~~~~~~~

* If ``ExecPolicy`` is not ``MDRangePolicy``, the ``functor`` has a member function of the form ``operator() (const HandleType& handle, ReducerValueType& value) const`` or ``operator() (const WorkTag, const HandleType& handle, ReducerValueType& value) const``.

  - If ``ExecPolicy::work_tag`` is ``void`` or if ``ExecPolicy`` is an ``IntegerType``, the overload without a ``WorkTag`` argument is used.
  - ``HandleType`` is an ``IntegerType`` if ``ExecPolicy`` is an ``IntegerType`` else it is ``ExecPolicy::member_type``.
* If ``ExecPolicy`` is ``MDRangePolicy`` the ``functor`` has a member function of the form ``operator() (const IntegerType& i0, ... , const IntegerType& iN, ReducerValueType& value) const`` or ``operator() (const WorkTag, const IntegerType& i0, ... , const IntegerType& iN, ReducerValueType& value) const``.

  - If ``ExecPolicy::work_tag`` is ``void``, the overload without a ``WorkTag`` argument is used.
  - ``N`` must match ``ExecPolicy::rank``.
* If the ``functor`` is a lambda, ``ReducerArgument`` must satisfy the ``Reducer`` concept or ``ReducerArgumentNonConst`` must be a POD type with ``operator +=`` and ``operator =`` or a ``Kokkos::View``.  In the latter case, a sum reduction is applied where the identity is assumed to be given by the default constructor of the value type (and not by ``reduction_identity```). If provided, the ``init``/ ``join``/ ``final`` member functions must not take a ``WorkTag`` argument even for tagged reductions.
* If ``ExecPolicy`` is ``TeamThreadRange`` a "reducing" ``functor`` is not allowed and the ``ReducerArgument`` must satisfy the ``Reducer`` concept or ``ReducerArgumentNonConst`` must be a POD type with ``operator +=`` and ``operator =`` or a ``Kokkos::View``.  In the latter case, a sum reduction is applied where the identity is assumed to be given by the default constructor of the value type (and not by ``reduction_identity```).
* If ``ExecPolicy`` is ``TeamVectorMDRange``, ``TeamThreadMDRange``, or ``ThreadVectorMDRange``, only a ``ReducerArgumentNonConst`` is allowed, and it must be a POD type with ``operator +=`` and ``operator =``.
* The reduction argument type ``ReducerValueType`` of the ``functor`` operator must be compatible with the ``ReducerArgument`` (or ``ReducerArgumentNonConst``) and must match the arguments of the ``init``, ``join`` and ``final`` functions of the functor if those exist and no reducer is specified (``ReducerArgument`` doesn't satisfy the ``Reducer`` concept but is a scalar, array or ``Kokkos::View``). In case of tagged reductions, i.e., when specifying a tag in the policy, the functor's potential ``init``/ ``join``/ ``final`` member functions must also be tagged.
* If ``ReducerArgument`` (or ``ReducerArgumentNonConst``)

  - is a scalar type then ``ReducerValueType`` must be of the same type.
  - is a rank-0 ``Kokkos::View``, then ``ReducerArgument::non_const_value_type`` must match ``ReducerValueType``.
  - satisfies the ``Reducer`` concept then ``ReducerArgument::value_type`` must match ``ReducerValueType``.
  - is an array or rank-1 ``Kokkos::View``, then:
    + ReducerValueType must be ``T[]`` where ``T`` is the element type of the array or View.
    + If ``ReducerArgument`` is an array then it must be statically sized.
    + the functor must define FunctorType::value_type the same as ReducerValueType.
    + the functor must declare a public member variable ``int value_count`` which is the length of the array.
    + the functor must implement the function ``void init( ReducerValueType dst[] ) const``.
    + the functor must implement the function ``void join( ReducerValueType dst[], ReducerValueType src[] ) const``.
    + If the functor implements the ``final`` function, the argument must also match those of init and join.

Semantics
---------

* For every element of the iteration space defined in ``policy`` the functors call operator is invoked exactly once, except for ``TeamPolicy`` and ``TeamThreadRange`` where the call operator is invoked by each vector lane of the team and thread respectively.
* Neither concurrency nor order of execution are guaranteed.
* The call is potentially asynchronous if the ``ReducerArgument`` is not a scalar type.
* The ``ReducerArgument`` content will be overwritten, i.e. the value does not need to be initialized to the reduction-neutral element.
* The input value to the operator may contain a partial reduction result, Kokkos may only combine the thread local contributions in the end. The operator must modify the input reduction value according to the requested reduction type.
* Plain scalar results are returned to the host immediately and do not require synchronization.
* If the reduction result is a ``Kokkos::View`` and the execution space is non-host (e.g. GPU), the result lives in device memory. Before reading it on the host, use either ``Kokkos::fence()`` followed by ``result()`` access, or ``Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result)``.

Examples
--------

Further examples are provided in the `Custom Reductions <../../../ProgrammingGuide/Custom-Reductions.rst>`_ and `ExecutionPolicy <../policies/ExecutionPolicyConcept.rst>`_ documentation.

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cstdio>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);

        int N = atoi(argv[1]);
        double result;
        Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum) {
            lsum += 1.0*i;
        }, result);

        printf("Result: %i %lf\n", N, result);
        Kokkos::finalize();
    }

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cstdio>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);

        int N = atoi(argv[1]);
        double sum, min;
        Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum, double& lmin) {
            lsum += 1.0*i;
            lmin = lmin < 1.0*i ? lmin : 1.0*i;
        }, sum, Kokkos::Min<double>(min));

        printf("Result: %i %lf %lf\n", N, sum, min);
        Kokkos::finalize();
    }

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cstdio>

    struct TagMax {};
    struct TagMin {};

    struct Foo {
        KOKKOS_INLINE_FUNCTION
        void operator() (const TagMax, const Kokkos::TeamPolicy<>::member_type& team, double& lmax) const {
            if (team.league_rank() % 17 + team.team_rank() % 13 > lmax)
                lmax = team.league_rank() % 17 + team.team_rank() % 13;
        }
        KOKKOS_INLINE_FUNCTION
        void operator() (const TagMin, const Kokkos::TeamPolicy<>::member_type& team, double& lmin) const {
            if (team.league_rank() % 17 + team.team_rank() % 13 < lmin)
                lmin = team.league_rank() % 17 + team.team_rank() % 13;
        }
    };

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);

        int N = atoi(argv[1]);

        Foo foo;
        double max, min;
        Kokkos::parallel_reduce(Kokkos::TeamPolicy<TagMax>(N,Kokkos::AUTO), foo, Kokkos::Max<double>(max));
        Kokkos::parallel_reduce("Loop2", Kokkos::TeamPolicy<TagMin>(N,Kokkos::AUTO), foo, Kokkos::Min<double>(min));
        Kokkos::fence();

        printf("Result: %lf %lf\n", min, max);

        Kokkos::finalize();
    }
