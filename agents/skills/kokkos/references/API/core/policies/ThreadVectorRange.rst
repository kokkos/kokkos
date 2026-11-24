``ThreadVectorRange``
=====================

Header File: ``<Kokkos_Core.hpp>``

Usage: 

.. code-block:: cpp

   parallel_for(ThreadVectorRange(team,range), [=] (int i) {...});
   parallel_reduce(ThreadVectorRange(team,begin,end), 
     [=] (int i, double& lsum) {...},sum);
   parallel_scan(ThreadVectorRange(team,range), 
     [=] (int i, double& lsum, bool final) {...});

ThreadVectorRange is a `nested execution policy <NestedPolicies.rst>`__ used inside hierarchical parallelism. 
In contrast to global policies, the public interface for nested policies is implemented 
as functions, in order to enable implicit templating on the execution space type via 
the team handle.

Synopsis 
--------

.. code-block:: cpp

   template<class TeamMemberType, class iType>
   /* implementation defined */ ThreadVectorRange(TeamMemberType team, iType count);
   template<class TeamMemberType, class iType1, class iType2>
   /* implementation defined */ ThreadVectorRange(TeamMemberType team, iType1 begin, iType2 end);


Description
-----------

.. code-block:: cpp

   template<class TeamMemberType, class iType>
   /* Implementation defined */ ThreadVectorRange(TeamMemberType team, iType count);
   

Splits the index range ``0`` to ``count-1`` over the vector lanes of the calling thread.
   
*  **Arguments**:

   * ``team``: a handle to the calling team execution context.

   * ``count``: index range length. 

*  **Returns**:

   * Implementation defined type.

*  **Requirements**

   * ``TeamMemberType`` is a type that models `TeamHandle <TeamHandleConcept>`__

   * ``std::is_integral<iType>::value`` is true.

   * ``count >= 0`` is true;

   * This function can not be called inside a parallel operation dispatched using a `TeamVectorRange <TeamVectorRange>`__ policy or ``ThreadVectorRange`` policy.


.. code-block:: cpp

   template<class TeamMemberType, class iType1, class iType2>
   /* Implementation defined */ ThreadVectorRange(TeamMemberType team, iType1 begin, iType2 end);


Splits the index range ``begin`` to ``end-1`` over the vector lanes of the calling thread. 

*  **Arguments**

   * ``team``: a handle to the calling team execution context.

   * ``begin``: index range begin. 

   * ``end``: index range end.

*  **Returns**

   * Implementation defined type.

* **Requirements**:

  * ``TeamMemberType`` is a type that models `TeamHandle <TeamHandleConcept.rst>`__   
  
  * ``std::is_integral<iType1>::value`` is true.

  * ``std::is_integral<iType2>::value`` is true.

  * ``end >= begin`` is true;

  * This function can not be called inside a parallel operation dispatched using a `TeamVectorRange <TeamVectorRange.rst>`__ policy or ``ThreadVectorRange`` policy.

  
Examples
--------

.. code-block:: cpp

    typedef TeamPolicy<>::member_type team_handle;
    parallel_for(TeamPolicy<>(N,AUTO,4), KOKKOS_LAMBDA (const team_handle& team) {
     int n = team.league_rank();
     parallel_for(TeamThreadRange(team,M), [&] (const int i) {
       parallel_for(ThreadVectorRange(team,K), [&] (const int j) {
         A(n,i,j) = B(n,i) + j;
       });
     });
     team.team_barrier();
     int team_sum;
     parallel_reduce(TeamThreadRange(team,M), [&] (const int& i, int& threadsum) {
       int tsum = 0;
       parallel_reduce(ThreadVectorRange(team,K), [&] (const int& j, int& lsum) {
         lsum += A(n,i,j);
       },tsum);
       single(PerThread(team),[&] () {
         threadsum += tsum;
       });
     },team_sum);
       
       lsum += A(n,i);
     },team_sum);
     single(PerTeam(team),[&] () {
       A_rowsum(n) += team_sum;
     });
    });
