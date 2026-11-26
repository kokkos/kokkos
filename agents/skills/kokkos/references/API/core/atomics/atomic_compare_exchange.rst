``atomic_compare_exchange``
===========================

.. role:: cpp(code)
   :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   auto old = atomic_compare_exchange(&obj, expected, desired);

Atomically compares the current value of ``obj`` with ``expected``,
replaces its value with ``desired`` if equal, and
always returns the previously stored value at the address ``&obj`` regardless of whether the exchange has happened or not.

Description
-----------

.. cpp:function:: template<class T> T atomic_compare_exchange(T* ptr, std::type_identity_t<T> expected, std::type_identity_t<T> desired);

   Atomically compares ``*ptr`` with ``expected``, and if those are bitwise-equal, replaces the former with ``desired``, and always returns the actual value that was pointed to by ``ptr`` before the call.

   ``{ old = *ptr; if (old == expected) *ptr = desired; return old; }``

   :param ptr: address of the object to test and to modify
   :param expected: value expected to be found in the object
   :param desired: the value to store in the object if as expected
   :returns: the value held previously by the object pointed to by ``ptr``


See also
--------
* `atomic_exchange <atomic_exchange.rst>`_: atomically replaces the value of the referenced object and obtains the value held previously
