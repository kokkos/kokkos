``ThreadVectorMDRange``
=======================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Description
-----------

ThreadVectorMDRange is a `nested execution policy <./NestedPolicies.rst>`_  used inside of hierarchical parallelism.

Interface
---------

.. cpp:class:: template <class Rank, typename TeamHandle> ThreadVectorMDRange

   .. rubric:: Constructor

   .. cpp:function:: ThreadVectorMDRange(team, extent_1, extent_2, ...);

      Splits the index range ``0`` to ``extent`` over the vector lanes of the calling thread,
      where ``extent`` is the backend-dependent rank that will be vectorized

      :param team: TeamHandle to the calling team execution context

      :param extent_1, extent_2, ...: index range lengths of each rank

      * **Requirements**

	* ``TeamHandle`` is a type that models `TeamHandle <./TeamHandleConcept.rst>`_

	* ``extent_1, extent_2, ...`` are ints

	* ``extent_i`` is such that ``i >= 2 && i <= 8`` is true.
	  For example:

	  .. code-block:: cpp

	     ThreadVectorMDRange(team, 4);               // NOT OK, violates i>=2

	     ThreadVectorMDRange(team, 4,5);             // OK
	     ThreadVectorMDRange(team, 4,5,6);           // OK
	     ThreadVectorMDRange(team, 4,5,6,2,3,4,5,6); // OK, max num of extents allowed

	* The constructor can not be called inside a parallel operation dispatched using a
	  ``TeamVectorRange`` policy, ``TeamVectorRange`` policy, ``TeamVectorMDRange`` policy
	  or ``ThreadVectorMDRange`` policy.

Restrictions
------------

Note that when used in `parallel_reduce <../parallel-dispatch/parallel_reduce.rst>`_, the reduction is limited to a sum.

Examples
--------

.. code-block:: cpp

   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N, Kokkos::AUTO),
     KOKKOS_LAMBDA(TeamHandle const& team) {
       int leagueRank = team.league_rank();

       auto teamThreadRange = TeamThreadRange(team, n0);
       auto threadVectorMDRange =
           ThreadVectorMDRange<Rank<3>, TeamHandle>(
               team, n1, n2, n3);

       parallel_for(teamThreadRange, [=](int i0) {
         parallel_for(threadVectorMDRange, [=](int i1, int i2, int i3) {
           A(leagueRank, i0, i1, i2, i3) += B(leagueRank, i1) + C(i1, i2, i3);
         });
       });
       team.team_barrier();

       int teamSum = 0;
       parallel_for(teamThreadRange, [=, &teamSum](int const& i0) {
         int threadSum = 0;
         parallel_reduce(threadVectorMDRange,
           [=](int i1, int i2, int i3, int& vectorSum) {
             vectorSum += D(leagueRank, i0, i1, i2, i3);
           }, threadSum
         );

         teamSum += threadSum;
       });
   });
