``atomic_exchange``
===================

.. role:: cpp(code)
   :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   auto old = atomic_exchange(&obj, desired);

Atomically replaces the value of ``obj`` with ``desired`` and returns the value before the call.

Description
-----------

.. cpp:function:: template<class T> T atomic_exchange(T* ptr, std::type_identity_t<T> val);

   Atomically writes ``val`` into ``*ptr`` and returns the original value of ``*ptr``.

   ``{ auto old = *ptr; *ptr = val; return old; }``

   :param ptr: address of the object to modify
   :param val: the value to store in the referenced object
   :returns: the value held previously by the object pointed to by ``ptr``


See also
--------
* `atomic_load <atomic_load.rst>`_: atomically obtains the value of the referenced object
* `atomic_store <atomic_store.rst>`_: atomically replaces the value of the referenced object with a non-atomic argument
* `atomic_compare_exchange <atomic_compare_exchange.rst>`_: atomically compares the value of the referenced object with non-atomic argument and performs atomic exchange if equal or atomic load if not
