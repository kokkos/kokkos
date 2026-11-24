``TeamThreadMDRange``
=====================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Description
-----------

TeamThreadMDRange is a `nested execution policy <./NestedPolicies.rst>`_  used inside of hierarchical parallelism.


Interface
---------

.. cpp:class:: template <class Rank, typename TeamHandle> TeamThreadMDRange

   .. rubric:: Constructor

   .. cpp:function:: TeamThreadMDRange(team, extent_1, extent_2, ...);

      Splits the index range ``0`` to ``extent`` over the threads of the team,
      where ``extent`` is the backend-dependent rank that will be threaded

      :param team: TeamHandle to the calling team execution context

      :param extent_1, extent_2, ...: index range lengths of each rank


      * **Requirements**

	* ``TeamHandle`` is a type that models `TeamHandle <./TeamHandleConcept.rst>`_

	* ``extent_1, extent_2, ...`` are ints

	* Every member thread of ``team`` must call the operation in the same branch,
	  i.e. it is not legal to have some threads call this function in one branch,
	  and the other threads of ``team`` call it in another branch

	* ``extent_i`` is such that ``i >= 2 && i <= 8`` is true.
	  For example:

	  .. code-block:: cpp

	     TeamThreadMDRange(team, 4);               // NOT OK, violates i>=2

	     TeamThreadMDRange(team, 4,5);             // OK
	     TeamThreadMDRange(team, 4,5,6);           // OK
	     TeamThreadMDRange(team, 4,5,6,2,3,4,5,6); // OK, max num of extents allowed

Restrictions
------------

Note that when used in `parallel_reduce <../parallel-dispatch/parallel_reduce.rst>`_, the reduction is limited to a sum.

Examples
--------

.. code-block:: cpp

   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N,AUTO),
     KOKKOS_LAMBDA (TeamHandle const& team) {

       int leagueRank = team.league_rank();

       auto range = TeamThreadMDRange<Rank<4>, TeamHandle>(team, n0, n1, n2, n3);

       parallel_for(range, [=](int i0, int i1, int i2, int i3) {
         A(leagueRank, i0, i1, i2, i3) = B(leagueRank, i1) + C(i1, i2, i3);
       });
       team.team_barrier();

       int teamSum = 0;
       parallel_reduce(range,
         [=](int i0, int i1, int i2, int i3, int& threadSum) {
           threadSum += D(leagueRank, i0, i1, i2, i3);
         }, teamSum
       );
       single(PerTeam(team), [&leagueSum, teamSum]() { leagueSum += teamSum; });
       A_rowSum[leagueRank] = leagueSum;
   });
