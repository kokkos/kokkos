``TeamHandleConcept``
=====================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

TeamHandleConcept defines the concept for the ``member_type`` of ``TeamPolicy``.
The actual type is defined through the policies, but meets the following API.
Note that the specific classes are only part of the public API as provided by ``TeamPolicy`` and only their parts as defined by the ``TeamHandleConcept``.
The actual name of the class, as well as potential template parameters, existing
constructors, member functions going beyond the concept, etc., are NOT part of the public Kokkos API
and are thus subject to change.


Description
-----------

.. cpp:class:: TeamHandleConcept


   .. rubric:: Public nested aliases

   .. cpp:type:: execution_space

      Specifies the `execution space <https://kokkos.github.io/kokkos-core-wiki/API/core/execution_spaces.html>`_ associated to the team

   .. cpp:type:: scratch_memory_space

      The scratch memory space associated to this team's execution space

   .. rubric:: Constructors

   .. cpp:function:: TeamHandleConcept() = default

      Default constructor.

   .. cpp:function:: TeamHandleConcept( TeamHandleConcept && ) = default

      Move constructor.

   .. cpp:function:: TeamHandleConcept( TeamHandleConcept const & ) = default

      Copy constructor.

   .. cpp:function:: ~TeamHandleConcept() = default

      Destructor.

   .. rubric:: Assignment

   .. cpp:function:: TeamHandleConcept & operator = ( TeamHandleConcept && ) = default

      Move assignment.

   .. cpp:function:: TeamHandleConcept & operator = ( TeamHandleConcept const & ) = default

      Assignment operators. Returns: ``*this``.

   .. rubric:: Index Queries

   .. cpp:function:: KOKKOS_INLINE_FUNCTION int team_rank() const noexcept ;

      Returns: the index ``i`` of the calling thread within the team with ``0 <= i < team_size()``

   .. cpp:function:: KOKKOS_INLINE_FUNCTION int team_size() const noexcept ;

      Returns: the number of threads associated with the team.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION int league_rank() const noexcept ;

      Returns: the index ``i`` of the calling team within the league with ``0 <= i < league_size()``

   .. cpp:function:: KOKKOS_INLINE_FUNCTION int league_size() const noexcept ;

      Returns: the number of teams/workitems launched in the kernel.


   .. rubric:: Scratch Space Control

   .. cpp:function:: KOKKOS_INLINE_FUNCTION const scratch_memory_space & team_shmem() const ;

      Equivalent to calling ``team_scratch(0)``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION const scratch_memory_space & team_scratch(int level) const ;

      This function returns a scratch memory handle shared by all threads in a team, which allows access to scratch memory.
      This handle can be given as the first argument to a ``Kokkos::View`` to make it use scratch memory.
      The handle is automatically incremented by the size of the ``Kokkos::View`` allocation.

      - ``level``: The level of requested scratch memory is either ``0`` or ``1``.

      - Returns: a scratch memory handle to the team shared scratch memory specified by level.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION const scratch_memory_space & thread_scratch(int level) const ;

      This function returns a scratch memory handle specific to the calling thread,
      which allows access to its private scratch memory. This handle can be given as the
      first argument to a ``Kokkos::View`` to make it use scratch memory.

      - ``level``: The level of requested scratch memory is either ``0`` or ``1``.

      - Returns: a scratch memory handle to the thread scratch memory specified by level.


   .. rubric:: Team Collective Operations

   The following functions must be called collectively by all members of a team.
   These calls must be lexically the same call, i.e. it is not legal to have some members of a team call
   a collective in one branch and the others in another branch of the code (see example).

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void team_barrier() const noexcept ;

      All members of the team wait at the barrier, until the whole team arrived. This also issues a memory fence.

   .. cpp:function:: template<typename T> KOKKOS_INLINE_FUNCTION void team_broadcast( T & value , const int source_team_rank ) const noexcept;

      After this call ``var`` contains for every member of the team the value of ``var`` from the thread for which ``team_rank() == source_team_rank``.

      - ``var``: a variable of type ``T`` which gets overwritten by the value of ``var`` from the source rank.

      - ``source_team_rank``: identifies the broadcasting member of the team.

   .. cpp:function:: template<class Closure, typename T> KOKKOS_INLINE_FUNCTION void team_broadcast( Closure const & f , T & value , const int source_team_rank) const noexcept;

      After this call ``var`` contains for every member of the team the value of ``var`` from the thread for which ``team_rank() == source_team_rank`` after applying ``f``.

      - ``f``: a function object with an ``void operator() ( T & )`` which is applied to ``var`` before broadcasting it.

      - ``var``: a variable of type ``T`` which gets overwritten by the value of ``f(var)`` from the source rank.

      - ``source_team_rank``: identifies the broadcasting member of the team.

   .. cpp:function:: template< typename ReducerType> KOKKOS_INLINE_FUNCTION void team_reduce( ReducerType const & reducer ) const noexcept;

      Performs a reduction across all members of the team as specified by ``reducer``. ``ReducerType`` must meet the concept of ``Kokkos::Reducer``.

   .. cpp:function:: template< typename T > KOKKOS_INLINE_FUNCTION T team_scan( T const & value , T * const global = 0 ) const noexcept;

      Performs an exclusive scan over the ``var`` provided by the team members. Let ``t = team_rank()`` and ``VALUES[t]`` the value of ``var`` from thread ``t``.

      - Returns: ``VALUES[0]`` + ``VALUES[1]`` + ``...`` + ``VALUES[t-1]`` or zero for ``t==0``.

      - ``global`` if provided will be set to ``VALUES[0]`` + ``VALUES[1]`` + ``...`` + ``VALUES[team_size()-1]``,
	must be the same pointer for every team member.

Examples
--------

.. code-block:: cpp

    typedef TeamPolicy<...> policy_type;
    parallel_for(policy_type(N,TEAM_SIZE).set_scratch_size(PerTeam(0,4096)),
                KOKKOS_LAMBDA (const typename policy_type::member_type& team_handle) {
        int ts = team_handle.team_size(); // returns TEAM_SIZE
        int tid = team_handle.team_rank(); // returns a number between 0 and TEAM_SIZE
        int ls = team_handle.league_size(); // returns N
        int lid = team_handle.league_rank(); // returns a number between 0 and N

        int value = tid * 5;
        team_handle.team_broadcast(value, 3);
        // value==15 on every thread
        value += tid;
        team_handle.team_broadcast([&] (int & var) { var*=2 }, value, 2);
        // value==34 on every thread
        int global;
        int scan = team_handle.team_scan(tid+1, &global);
        // scan == tid*(tid+1)/2 on every thread
        // global == ts*(ts-1)/2 on every thread
        // The view does not require the Unmanaged trait, since team_handle.team_scratch(0) decays to a pointer,
        // but it is added for clarity here. The View constructor that creates an unmanaged view is triggered by the pointer passed.
        Kokkos::View<int*, policy_type::execution_space::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        a(team_handle.team_scratch(0), 1024);

    });
