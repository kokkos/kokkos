``atomic_assign``
=================

.. warning::
   Deprecated since Kokkos 4.5,
   use :cpp:func:`atomic_store` instead.

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_assign(&obj, desired);

Atomically replaces the current value of ``obj`` with ``desired``.

Description
-----------

.. cpp:function:: template<class T> void atomic_assign(T* ptr, std::type_identity_t<T> val);

   Atomically writes ``val`` into ``*ptr``.

   ``{ *ptr = val; }``

   :param ptr: address of the object whose value is to be replaced
   :param val: the value to store in the referenced object
   :returns: (nothing)

   .. deprecated:: 4.5
      Use :cpp:func:`atomic_store` instead.
