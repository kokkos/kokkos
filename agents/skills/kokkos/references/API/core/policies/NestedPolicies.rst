``NestedPolicies``
==================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

List
----

``Kokkos::PerTeam``
~~~~~~~~~~~~~~~~~~~

``Kokkos::PerThread``
~~~~~~~~~~~~~~~~~~~~~

``Kokkos::TeamThreadRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::TeamThreadMDRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::TeamVectorRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::TeamVectorMDRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::ThreadVectorRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::ThreadVectorMDRange``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments are described `here <../Execution-Policies.rst#common-arguments-for-all-execution-policies>`_

Usage
~~~~~

.. code-block:: cpp

    parallel_for(TeamThreadRange(team,begin,end), [=] (int i) {});
    parallel_for(ThreadVectorRange(team,begin,end), [=] (int i) {});
    single(PerTeam(team), [=] () {});
    single(PerThread(team), [=] () {});

Nested policies can be used for nested parallel patterns. In contrast to global policies, the public interface for nested policies is implemented as functions, in order to enable implicit templating on the execution space type via the team handle.

Synopsis
~~~~~~~~

.. code-block:: cpp

    Impl::TeamThreadRangeBoundariesStruct TeamThreadRange(TeamMemberType team, IndexType count);
    Impl::TeamThreadRangeBoundariesStruct TeamThreadRange(TeamMemberType team, IndexType begin, IndexType end);
    Impl::ThreadVectorRangeBoundariesStruct ThreadVectorRange(TeamMemberType team, IndexType count);
    Impl::ThreadVectorRangeBoundariesStruct ThreadVectorRange(TeamMemberType team, IndexType begin, IndexType end);
    Impl::ThreadSingleStruct PerTeam(TeamMemberType team);
    Impl::VectorSingleStruct PerThread(TeamMemberType team);

Description
~~~~~~~~~~~

.. cpp:function:: Impl::TeamThreadRangeBoundariesStruct TeamThreadRange(TeamMemberType team, IndexType count);

    Splits the index range ``0`` to ``count-1`` over the threads of the team. This call is potentially a synchronization point for the team, and thus must meet the requirements of ``team_barrier``.
        - ``team``: object meeting the requirements of TeamHandle
        - ``count``: index range length.

.. cpp:function:: Impl::TeamThreadRangeBoundariesStruct TeamThreadRange(TeamMemberType team, IndexType begin, IndexType end);

    Splits the index range ``begin`` to ``end-1`` over the threads of the team. This call is potentially a synchronization point for the team, and thus must meet the requirements of ``team_barrier``.
        - ``team``: object meeting the requirements of TeamHandle
        - ``begin``: start index.
        - ``end``: end index.

.. cpp:function:: Impl::ThreadVectorRangeBoundariesStruct ThreadVectorRange(TeamMemberType team, IndexType count);

    Splits the index range ``0`` to ``count-1`` over the vector lanes of the calling thread. It is not legal to call this function inside of a vector level loop.
        - ``team``: object meeting the requirements of TeamHandle
        - ``count``: index range length.

.. cpp:function:: Impl::ThreadVectorRangeBoundariesStruct ThreadVectorRange(TeamMemberType team, IndexType begin, IndexType end);

    Splits the index range ``begin`` to ``end-1`` over the vector lanes of the calling thread. It is not legal to call this function inside of a vector level loop.
        - ``team``: object meeting the requirements of TeamHandle
        - ``begin``: start index.
        - ``end``: end index.

.. cpp:function:: Impl::ThreadSingleStruct PerTeam(TeamMemberType team);

    When used in conjunction with the ``single`` pattern restricts execution to a single vector lane in the calling team. While not a synchronization event, this call must be encountered by the entire team, and thus meet the calling requirements of ``team_barrier``.
        - ``team``: object meeting the requirements of TeamHandle

.. cpp:function:: Impl::VectorSingleStruct PerThread(TeamMemberType team);

    When used in conjunction with the ``single`` pattern restricts execution to a single vector lane in the calling thread. It is not legal to call this function inside of a vector level loop.
        - ``team``: object meeting the requirements of TeamHandle

Examples
~~~~~~~~

.. code-block:: cpp

    typedef TeamPolicy<>::member_type team_handle;
    parallel_for(TeamPolicy<>(N,AUTO,4), KOKKOS_LAMBDA (const team_handle& team) {
        int n = team.league_rank();
        parallel_for(TeamThreadRange(team,M), [&] (const int& i) {
            int thread_sum;
            parallel_reduce(ThreadVectorRange(team,K), [&] (const int& j, int& lsum) {
                //...
            },thread_sum);
            single(PerThread(team), [&] () {
                A(n,i) += thread_sum;
            });
        });
        team.team_barrier();
        int team_sum;
        parallel_reduce(TeamThreadRange(team,M), [&] (const int& i, int& lsum) {
            lsum += A(n,i);
        },team_sum);
        single(PerTeam(team),[&] () {
            A_rowsum(n) += team_sum;
        });
    });
