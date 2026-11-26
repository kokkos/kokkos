``atomic_store``
================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_store(&obj, desired);

Atomically replaces the current value of ``obj`` with ``desired``.

Description
-----------

.. cpp:function:: template<class T> void atomic_store(T* ptr, std::type_identity_t<T> val);

   Atomically writes ``val`` into ``*ptr``.

   ``{ *ptr = val; }``

   :param ptr: address of the object whose value is to be replaced
   :param val: the value to store in the referenced object
   :returns: (nothing)


See also
--------
* `atomic_load <atomic_load.rst>`_: atomically obtains the value of the referenced object
* `atomic_exchange <atomic_exchange.rst>`_: atomically replaces the value of the referenced object and obtains the value held previously
