``atomic_load``
===============

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    auto current = atomic_load(&obj);

Atomically obtains the current value of ``obj``.

Description
-----------

.. cpp:function:: template<class T> T atomic_load(T* ptr);

   Atomically reads the content of ``*ptr`` and returns it.

   ``{ T val = *ptr; return val; }``

   :param ptr: address of the object whose current value is to be returned
   :returns: the value that is held by the object pointed to by ``ptr``

See also
--------
* `atomic_store <atomic_store.rst>`_: atomically replaces the value of the referenced object with a non-atomic argument
* `atomic_exchange <atomic_exchange.rst>`_: atomically replaces the value of the referenced object and obtains the value held previously
