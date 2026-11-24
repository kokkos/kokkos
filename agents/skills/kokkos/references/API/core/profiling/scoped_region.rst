``Profiling::ScopedRegion``
===========================

.. role:: cpp(code)
   :language: cpp

Defined in header ``<Kokkos_Profiling_ScopedRegion.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::Profiling::ScopedRegion region("label");  // (since 4.1)



The class ``ScopedRegion`` is a `RAII
<https://en.cppreference.com/w/cpp/language/raii>`_ wrapper that "pushes" a
user-defined profiling region when an object is created and properly "pops"
that region upon destruction when the scope is exited. This is useful in
particular to profile code that has non-trivial control flow (e.g.  early
return).

The ``ScopedRegion`` class is non-copyable.

.. cpp:Function:: ScopedRegion(std::string const& regionName);

   Starts a user-defined region with provided label.
   Calls ``Profiling::pushRegion(regionName)``

.. cpp:Function:: ~ScopedRegion();

   Ends the region.
   Calls ``Profiling::popRegion()``

Example
-------

.. code-block:: cpp

   #include <Kokkos_Profiling_ScopedRegion.hpp>

   void do_work_v1() {
     Kokkos::Profiling::pushRegion("MyApp::do_work");
     // <code>
     if (cond) {
       Kokkos::Profiling::popRegion();  // must remember to pop here as well
       return;
     }
     // <more code>
     Kokkos::Profiling::popRegion();
   }

   void do_work_v2() {
     Kokkos::Profiling::ScopedRegion region("MyApp::do_work");
     // <code>
     if (cond) return;
     // <more code>
   }



**See also**

`ProfilingSection <profiling_section.rst>`_: Implements a scope-based section ownership wrapper
