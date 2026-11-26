``atomic_compare_exchange_strong``
==================================

.. warning::
   Deprecated since Kokkos 4.5,
   use `atomic_compare_exchange <atomic_compare_exchange.rst>`_ instead.

.. role:: cpp(code)
   :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   bool was_exchanged = atomic_compare_exchange_strong(&obj, expected, desired);

Atomically compares the current value of ``obj`` with ``expected``
and replaces its value with ``desired`` if equal.
The function returns ``true`` if the exchange has happened, ``false`` otherwise.

Description
-----------

.. cpp:function:: template<class T> bool atomic_compare_exchange_strong(T* ptr, std::type_identity_t<T> expected, std::type_identity_t<T> desired);

   Atomically compares ``*ptr`` with ``expected``, and if those are bitwise-equal, replaces the former with ``desired``.
   If ``desired`` is written into ``*ptr`` then ``true`` is returned.

   ``if (*ptr == expected) { *ptr = desired; return true; } else return false;``

   :param ptr: address of the object to test and to modify
   :param expected: value expected to be found in the object
   :param desired: the value to store in the object if as expected
   :returns: the result of the comparison, ``true`` if ``*ptr`` was equal to ``expected``, ``false`` otherwise

   .. deprecated:: 4.5
      Prefer :cpp:expr:`expected == atomic_compare_exchange(&obj, expected, desired)`
