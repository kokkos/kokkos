``is_initialized`` and ``is_finalized``
=======================================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

  if(Kokkos::is_initialized()) {
    // do work
  }
  if(!Kokkos::is_initialized() && Kokkos::is_finalized()) {
    // may initialize Kokkos
  }

The functions :cpp:func:`is_initialized` and :cpp:func:`is_finalized` allow
applications to query the current state of the Kokkos execution environment.
Because Kokkos follows a strict linear lifecycle, these functions are often
used to ensure that initialization or finalization occurs exactly once.

Interface
---------

.. cpp:function:: bool is_initialized() noexcept

   :return: ``true`` if the Kokkos execution environment is currently active
     and usable.

.. cpp:function:: bool is_finalized() noexcept

   :return: ``true`` if the Kokkos execution environment has been shut down via
     :cpp:func:`finalize`.


Notes
-----
.. note:: **The Lifecycle State Machine**

  Kokkos moves through three distinct states. Note that once Kokkos is finalized,
  it **cannot** be re-initialized within the same process execution.
  
  .. list-table::
     :widths: 25 20 20 35
     :header-rows: 1
     
     * - Program Phase
       - ``is_initialized()``
       - ``is_finalized()``
       - Description
     * - Pre-Initialization
       - ``false``
       - ``false``
       - Kokkos is not yet active.
     * - Active
       - ``true``
       - ``false``
       - ``initialize()`` has been called. Kernels can be launched.
     * - Post-Finalization
       - ``false``
       - ``true``
       - ``finalize()`` has been called. Kokkos is no longer usable.

.. caution::

   **Comparison with MPI:** Users familiar with MPI should note a key
   difference. In MPI, ``MPI_Initialized`` returns ``true`` even after
   ``MPI_Finalize`` is called. In Kokkos, ``is_initialized()`` returns
   ``false`` after finalization.

   To check if ``Kokkos::initialize()`` was ever called, use the logic:

   .. code-block:: cpp
      
      if (Kokkos::is_initialized() || Kokkos::is_finalized()) { ... }

Example
-------

.. code-block:: cpp

  #include <Kokkos_Core.hpp>
  #include <cstdio>
  
  int main(int argc, char* argv[]) {
    printf("before initialize: initialized=%d finalized=%d\n",
      Kokkos::is_initialized(), Kokkos::is_finalized());

    Kokkos::initialize();

    printf("kokkos active:     initialized=%d finalized=%d\n",
      Kokkos::is_initialized(), Kokkos::is_finalized());

    Kokkos::finalize();

    printf("after finalize:    initialized=%d finalized=%d\n",
      Kokkos::is_initialized(), Kokkos::is_finalized());
  }


Output:

.. code-block::

    before initialize: initialized=0 finalized=0
    kokkos active:     initialized=1 finalized=0
    after finalize:    initialized=0 finalized=1


See also
--------

.. seealso::

  :doc:`initialize`
    Start the Kokkos execution environment.
  :doc:`finalize`
    Terminate the Kokkos execution environment.
