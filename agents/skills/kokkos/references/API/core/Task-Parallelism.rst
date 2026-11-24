Task-Parallelism
================

.. role:: cpp(code)
    :language: cpp

.. warning::
   Deprecated since Kokkos 4.5, removed in Kokkos 5.0.

Kokkos has support for lightweight task-based programming, which is currently pretty limited but we plan to substantially expand in the future.

Will Kokkos Tasking work for my problem?
----------------------------------------

Not all task-based problems are a good fit for the current Kokkos approach to tasking. Currently, the tasking interface in Kokkos is targeted at problems with kernels far too small to overcome the inherent overhead of top-level Kokkos data parallel launches—that is, small but plentiful data parallel tasks with a non-trivial dependency structure. For tasks that fit this general scale model but have (very) trivial dependency structures, it may be easier to use `hierarchical parallelism <../../ProgrammingGuide/HierarchicalParallelism.rst>`_, potentially with a ``Kokkos::Schedule<Dynamic>`` scheduling policy (see, for instance, `this page <policies/RangePolicy.rst>`_) for load balancing if necessary. 

Basic Usage
-----------

Fundamentally, task parallelism is just another form of parallelism in Kokkos. The same general idiom of pattern, policy, and functor applies as for ordinary `parallel dispatch <../../ProgrammingGuide/ParallelDispatch.rst>`_:

.. image:: ../../ProgrammingGuide/figures/parallel-dispatch.png

Similarly, for tasking, we have:

.. image:: ../../ProgrammingGuide/figures/task-dispatch.png

Task Functor
------------

In most ways, the functor portion of the task parallelism idiom in Kokkos is similar to that for data parallelism.  Task functors haeve the additional requirement that the task output type needs to be provided by the user by giving a nested type (a.k.a. "typedef") named ``value_type``:

.. code-block:: cpp

    struct MyTask {
        using value_type = double;
        template <class TeamMember>
        KOKKOS_INLINE_FUNCTION
        void operator()(TeamMember& member, double& result);
    };

Similar to `team parallelism <../../ProgrammingGuide/HierarchicalParallelism.rst>`_, the first parameter is the team handle, which has all of the same functionality as the one produced by a ``Kokkos::TeamPolicy``, with a few extras. Like with ``Kokkos::parallel_reduce()``, the output is expressed through the second parameter. Note that there is currently no lambda interface to Kokkos Tasking.

Task Patterns
-------------

The primary analogs of ``Kokkos::parallel_for()`` and friends for tasking are ``Kokkos::task_spawn()`` and ``Kokkos::host_spawn()``. They both return a ``Kokkos::Future`` associated with the appropriate ``Scheduler``, but ``host_spawn`` can only be called from *outside* of a task functor, while ``task_spawn`` can only be called from *inside* of one.

Task Policies
-------------

There are currently two task policies in Kokkos Tasking: ``TaskSingle`` and ``TaskTeam``. The former tells Kokkos to launch the associated task functor with a single worker when its predecessors are done (more on this below), while the latter tells Kokkos to launch with a team of workers, similar to a single team from a ``Kokkos::TeamPolicy`` launch in data parallelism. In a task spawned with ``TaskTeam``, users are only allowed to call ``task_spawn`` from a single worker; use ``Kokkos::single`` for this purpose.

Predecessors and Schedulers
---------------------------

Dependency relationships in Kokkos are represented by instances of the ``Kokkos::BasicFuture`` class template. Each task (created with the ``task_spawn`` or ``host_spawn`` patterns) can have zero or one predecessors (to create task graphs with more predecessors, use ``Kokkos::when_all``, described below). Predecessors are given to a pattern as an argument to the policy:

.. code-block:: cpp

    using scheduler_type = /* ... discussed below ... */;
    auto scheduler = scheduler_type(/* ... discussed below ... */);
    // Launch with no predecessor:
    auto fut = Kokkos::host_spawn(
        Kokkos::TaskSingle(scheduler),
        MyTaskFunctor()
    );
    // Launch when `fut` is ready, at the earliest:
    auto fut2 = Kokkos::host_spawn(
        Kokkos::TaskSingle(scheduler, fut),
        MyOtherTaskFunctor()
    );
    /* ... */

The Kokkos ``TaskScheduler`` concept is an abstraction that generalizes over the many possible strategies for scheduling tasks in a task-based system. Like other concepts in Kokkos, users should not write code that depends directly on a specific ``TaskScheduler``, but rather to the generic model that all ``TaskScheduler`` types guarantee.

The ``Kokkos::BasicFuture`` class template, used for representing dependency relationships, is templated on the return type of the task it represents and on the type of the scheduler that was used to execute that task:

.. code-block:: cpp

    template <class Scheduler>
    void my_function(Scheduler sched) {
        // use auto until you need to name the type for some reason
        auto fut = Kokkos::host_spawn(
            Kokkos::TaskSingle(sched),
            MyTaskFunctor()
        );
        /* ... */
        using my_result_type = MyTaskFunctor::value_type;
        // convertibility is guaranteed:
        Kokkos::BasicFuture<my_task_result, Scheduler> ff = fut;
    }

(Note: Kokkos does not guarantee the specific return type of task parallel patterns, only that they will be convertible to the appropriate ``Kokkos::BasicFuture`` type. Use ``auto`` until you need to name the type for some reason—like storing it in a container, for instance. Otherwise, Kokkos may be able to provide better performance if the future type is never required to be converted to a specific ``Kokkos::BasicFuture`` type.)

``TaskScheduler`` types in Kokkos have shared reference semantics; a copy of a given scheduler represents the same underlying entity and strategy as the scheduler it was copied from. Inside of a task functor, users should retrieve the scheduler instance from the team member handle passed in as the first argument rather than storing the scheduler themselves.  Use ``auto`` for this as well until you need to store it for some reason.

When a future is ready, the result of the task that a future represents as a predecessor can be retrieved using the ``get()`` method.  However, this can **only** be called from a context where the future is guaranteed to be ready—that is, in a task that was spawned with the future as a predecessor, or a task that transitively depends on that future via another task, or after a ``Kokkos::wait`` on the scheduler that spawned the task associated with the future (see below).  **Calling the** ``get()`` **method of a future in any other context results in undefined behavior** (and the worst kind of bug, at that: it may not even result in a segfault or anything until hours of execution!).  Note that this is different from ``std::future``, where the ``get()`` method blocks until it's ready.

Future types in Kokkos have shared reference semantics; a copy of a given future represents the same underlying dependency as the future it was copied from.  A default-constructed ``Kokkos::BasicFuture`` represents an always-ready dependency with no value (that is, retrieving the value is undefined behavior—practically speaking, probably a segfault).  A default-constructed future will return ``true`` for the ``is_null()`` method.  In addition to convertibility to a ``Kokkos::BasicFuture`` of the appropriate value type and scheduler type, all Kokkos futures are convertible to a ``Kokkos::BasicFuture`` of ``void`` and the appropriate scheduler type.

Waiting in Kokkos Tasking
-------------------------

Kokkos generally provides no way to block a thread of execution to wait on an individual future, and it provides no guarantee of correct execution if the user attempts to do so via external means (for instance, polling on the ``is_ready()`` method in a ``while`` loop is forbidden).  *Outside of* a Kokkos task functor (that is, anywhere that ``host_spawn`` would be allowed), Kokkos provides the ability to wait on *all* of the futures created on a given scheduler (including those created, transitively, by tasks spawned not yet completed, or potentially not even started).  This is done using the ``Kokkos::wait`` function on the scheduler:

.. code-block:: cpp

    template <class Scheduler>
    void my_function(Scheduler sched) {
        // use auto until you need to name the type for some reason
        auto fut = Kokkos::host_spawn(
            Kokkos::TaskSingle(sched),
            MyTaskFunctor()
        );
        Kokkos::wait(sched);
        auto value = fut.get();
        /* ... */
    }

Users should think of ``Kokkos::wait`` as an *extremely* expensive operation (a "sledgehammer") and use it as sparingly as possible.

"Waiting" in a task functor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Kokkos tasking, all task functors must be able to run to completion without blocking once they are started (the Kokkos scheduler *can* run other tasks at any point that the functor calls back into the Kokkos tasking system, such as any ``task_spawn``, but it is allowed to assume user functors will run to completion if left alone).  This means that there is no way to block a task pending the result of another task.  Other tasking systems that make this kind of design decision require the user to spawn a new task for each new piece of predicated work, which is an option in Kokkos as well, but Kokkos also provides another option.  To help reduce the allocation cost associated with the traditional approach to never-blocking task systems, Kokkos allows users to "reuse" the current task as a successor to some future.  Kokkos provides the ``Kokkos::respawn()`` function.  For example:

.. code-block:: cpp

    template <class Scheduler>
    struct MyTaskFunctor {
        using value_type = void;
        using future_type = Kokkos::BasicFuture<double, Scheduler>;
        future_type f;
        template <class TeamMember>
        KOKKOS_INLINE_FUNCTION
        void operator()(TeamMember& member) {
            if(f.is_null()) {
            f = Kokkos::task_spawn(
                Kokkos::TaskSingle(member.scheduler()),
                MyOtherTaskFunctor()
            );
            Kokkos::respawn(this, f);
            }
            else {
            // This is after the respawn so we're guaranteed that f is ready
            printf("Got result %f\n", f.get());
            }
        }
    };

A task functor can only be respawned up to once *per execution of* ``operator()`` (that is, once per time it is spawned or respawned).  Multiple calls to ``respawn`` in the same execution of ``operator()`` are redundant and lead to undefined behavior.  Calls to ``respawn`` are always lazy—the subsequent call to ``operator()`` by Kokkos will only happen after the currently executing one returns (and after the predecessors, if any, are ready) at the earliest.

The first argument to ``Kokkos::respawn`` must always be a pointer to the currently executing task functor (or one of its base classes) from which ``Kokkos::respawn`` is called.  The second argument can be either a future of the same scheduler as the currently executing task functor or an instance of the scheduler itself.  The third (optional) argument is a task priority, discussed below.

Aggregate Predecessors
----------------------

Kokkos tasking provides two forms of the ``when_all()`` method on every ``TaskScheduler`` type. Both serve to aggregate multiple predecessors into one, and both return a value convertible to a ``Kokkos::BasicFuture`` of ``void`` and that scheduler type.  The first takes an array of ``Kokkos::BasicFuture`` of the scheduler type and a count of entries in that array.  The second takes a ``count`` and a unary function or functor that should expect to be called with each integer in the range ``[0, count)``.  In both cases, the return value is a future that will become ready when all of the input futures become ready.

Task Priorities
---------------

Kokkos allows users to provide a priority hint to task parallel execution policies as an optional third argument, or as an optional third argument to ``Kokkos::respawn``.  This has no observable effect on the programming model—only on the performance.  A scheduler may ignore these priorities.  The allowed task priorities are ``Kokkos::TaskPriority::High``, ``Kokkos::TaskPriority::Regular``, and ``Kokkos::TaskPriority::Low``, which the second being the default if the argument isn't given.

..
    Invariants in the Kokkos Tasking Programming Model
    ==================================================

..
    TODO
