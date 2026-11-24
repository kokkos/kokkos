``TeamPolicy``
==============

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::TeamPolicy<>( league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<ARGS>(league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<>(Space, league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<ARGS>(Space, league_size, team_size [, vector_length])

Execution policy for a 1D iteration space starting at begin and going to end with an open interval.

See also: `TeamMember <TeamHandleConcept.rst>`_

Description
-----------

.. cpp:class:: template<class ...Args> TeamPolicy

   .. rubric:: Template Arguments

   Valid template arguments for TeamPolicy are described `here <../Execution-Policies.rst#common-arguments-for-all-execution-policies>`_

   .. rubric:: Public nested typedefs

   .. cpp:type:: execution_space
   .. cpp:type:: schedule_type
   .. cpp:type:: work_tag
   .. cpp:type:: index_type
   .. cpp:type:: iteration_pattern
   .. cpp:type:: launch_bounds
   .. cpp:type:: member_type


   .. rubric:: Constructors

   .. cpp:function:: TeamPolicy()

      Default constructor uninitialized policy.

   .. cpp:function:: TeamPolicy(const TeamPolicy&) = default;

      Copy constructor

   .. cpp:function:: TeamPolicy(TeamPolicy&&) = default;

      Move constructor

   .. cpp:function:: TeamPolicy(index_type league_size, index_type team_size, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads
      with ``team_size`` threads, using a vector length of ``vector_length``. If the team size is not possible when
      calling a parallel policy, that kernel launch may throw.

   .. cpp:function:: TeamPolicy(index_type league_size, AUTO_t, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads of a
      size determined by Kokkos, using a vector length of ``vector_length``. The team size may be determined
      lazily at launch time, taking into account properties of the functor.

   .. cpp:function:: TeamPolicy(execution_space space, index_type league_size, index_type team_size, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads with ``team_size`` threads,
      using a vector length of ``vector_length``. If the team size is not possible when calling a parallel policy,
      that kernel launch may throw. Use the provided execution space instance during a kernel launch.

   .. cpp:function:: TeamPolicy(execution_space space, index_type league_size, AUTO_t, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads of a size determined by Kokkos,
      using a vector length of ``vector_length``. The team size may be determined lazily at launch time, taking into
      account properties of the functor. Use the provided execution space instance during a kernel launch.

   .. rubric:: Runtime Settings

   .. cpp:function:: inline TeamPolicy& set_chunk_size(int chunk);

      Set the chunk size. Each physical team of threads will get assigned ``chunk`` consecutive teams. Default is 1.

      Returns: reference to ``*this``

   .. cpp:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team);

   .. cpp:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread);

   .. cpp:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team, const Impl::PerThreadValue& per_thread);

   .. cpp:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread, const Impl::PerTeamValue& per_team);

      Set the per team and per thread scratch size.

      - ``level``: set the storage level. 0 is closest cache. 1 is closest storage (e.g. high bandwidth memory)

      - ``per_team``: wrapper for the per team size of scratch in bytes. Returned by the function ``PerTeam(int)``.

      - ``per_thread``: wrapper for the per thread size of scratch in bytes. Returned by the function ``PerThread(int)``.

      One can set the scratch size for level 0 and 1 independently by calling the function twice. Subsequent calls with the same level overwrite the previous value.
      Returns: reference to ``*this``

   .. rubric:: Query Limits of Runtime Settings

   .. _parallelFor: ../parallel-dispatch/parallel_for.rst

   .. |parallelFor| replace:: :cpp:func:`parallel_for`

   .. _parallelReduce: ../parallel-dispatch/parallel_reduce.rst

   .. |parallelReduce| replace:: :cpp:func:`parallel_reduce`

   .. cpp:function:: template<class FunctorType> int team_size_max(const FunctorType& f, const ParallelForTag&) const;

   .. cpp:function:: template<class FunctorType> int team_size_max(const FunctorType& f, const ParallelReduceTag&) const;

      Query the maximum team size possible given a specific functor. The tag denotes whether this is for a |parallelFor|_ or a |parallelReduce|_.
      Note: this is not a static function! The function will take into account settings for vector length and scratch size of ``*this``. Using a value larger than the return value will result in dispatch failure. If the value returned is non-positive, no valid team size could be found. A common reason is that too much scratch cache memory was requested.
      Returns: The maximum value for ``team_size`` allowed to be given to be used with an otherwise identical ``TeamPolicy`` for dispatching the functor ``f``.

   .. cpp:function:: template<class FunctorType> int team_size_recommended(const FunctorType& f, const ParallelForTag&) const;

   .. cpp:function:: template<class FunctorType> int team_size_recommended(const FunctorType& f, const ParallelReduceTag&) const;

      Query the recommended team size for the specific functor ``f``. The tag denotes whether this is for a |parallelFor|_ or a |parallelReduce|_.
      Note: this is not a static function! The function will take into account settings for vector length and scratch size of ``*this``. If the value returned is non-positive, no valid team size could be found. A common reason is that too much scratch cache memory was requested.
      Returns: The recommended value for ``team_size`` to be given to be used with an otherwise identical ``TeamPolicy`` for dispatching the functor ``f``.

   .. cpp:function:: static int vector_length_max();

      Returns: the maximum valid value for vector length.

   .. cpp:function:: static int scratch_size_max(int level);

      Returns: the maximum total scratch size in bytes, for the given level.
      Note: If a kernel performs team-level reductions or scan operations, not all of this memory will be
      available for dynamic user requests. Some of that maximal scratch size is being used for internal operations.
      The actual size of these internal allocations depends on the value type used in the reduction or scan.

   .. rubric:: Query Runtime Settings

   .. cpp:function:: int team_size() const;

      Returns: the requested team size.

   .. cpp:function:: int league_size() const;

      Returns: the requested league size.

   .. cpp:function:: int scratch_size(int level, int team_size_ = -1) const;

      This function returns the total scratch size requested. If ``team_size`` is not provided, the team size
      for the calculation is used from the internal setting (i.e. the result of calling ``this->team_size()``). Otherwise, the provided team size is used.
      Returns: the value for the total scratch size in bytes in the specified scratch level.

   .. cpp:function:: int team_scratch_size(int level) const;

      Returns: the value for the per team scratch size in bytes in the specified scratch level.

   .. cpp:function:: int thread_scratch_size(int level) const;

      Returns: the value for the per thread scratch size in bytes in the specified scratch level.

   .. cpp:function:: int chunk_size() const;

      Returns: the chunk size, set via ``set_chunk_size()``.

``Kokkos::single``
------------------

``Kokkos::single(Kokkos::PerTeam(team), [=]() { statements... })`` restricts execution of ``statements`` to happen exactly once for the whole team. A single arbitrary thread executes the statements while the rest are idle. This is useful when the statements have side effects (e.g., writing to global memory from a reduction result).

``Kokkos::single(Kokkos::PerThread(team), [=]() { statements... })`` is the same except it runs ``statements`` once per *thread*.

.. warning::

    Avoid ``ThreadVectorRange(team, 1)`` — it defeats vectorization and causes idle lanes.

Examples
--------

.. code-block:: cpp

    TeamPolicy<> policy_1(N,AUTO);
    TeamPolicy<Cuda> policy_2(N,T);
    TeamPolicy<Schedule<Dynamic>, OpenMP> policy_3(N,AUTO,8);
    TeamPolicy<IndexType<int>, Schedule<Dynamic>> policy_4(N,1,4);
    TeamPolicy<OpenMP> policy_5(OpenMP(), N, AUTO);

.. code-block:: cpp

    using team_handle = TeamPolicy<>::member_type;
    parallel_for(TeamPolicy<>(N,AUTO), KOKKOS_LAMBDA (const team_handle& team) {

        // each team initializes a row of A
        int n = team.league_rank();
        parallel_for(TeamThreadRange(team,M), [&] (const int& i) {
            A(n,i) = B(n) + i;
        });
        team.team_barrier();

        // Compute the sum of the nth row of matrix A, stored as a rank-2 view
        int team_sum;
        parallel_reduce(TeamThreadRange(team,M), [&] (const int& i, int& lsum) {
            lsum += A(n,i);
        },team_sum);
        
        // store the sum of the row in corresponding entry of A_rowsum vector
        single(PerTeam(team),[&] () {
            A_rowsum(n) = team_sum;
        });
    });
