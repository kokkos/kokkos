``atomic_[op]``
===============

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_[op](ptr_to_value,update_value);

Atomically updates the ``value`` at the address given by ``ptr_to_value`` with ``update_value`` according to the relevant operation.

Description
-----------

.. cpp:function:: template<class T> void atomic_add(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value += value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be added.

.. cpp:function:: template<class T> void atomic_and(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value &= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T> void atomic_dec(T* ptr_to_value);

   Atomically executes ``(*ptr_to_value)--`` or calls ``atomic_fetch_sub(ptr_to_value, T(-1))``.

   * ``ptr_to_value``: address of the to be updated value.

.. cpp:function:: template<class T> void atomic_div(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value /= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to divide with.

.. cpp:function:: template<class T> void atomic_decrement(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)--`` or calls ``atomic_fetch_sub(ptr_to_value, T(-1))``.

   * ``ptr_to_value``: address of the to be updated value.

   .. deprecated:: 4.5
      Use :cpp:func:`atomic_dec` instead.

.. cpp:function:: template<class T> void atomic_inc(T* ptr_to_value);

   Atomically executes ``(*ptr_to_value)++`` or calls ``atomic_fetch_add(ptr_to_value, T(1))``.

   * ``ptr_to_value``: address of the to be updated value.

.. cpp:function:: template<class T> void atomic_increment(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)++`` or calls ``atomic_fetch_add(ptr_to_value, T(1))``.

   * ``ptr_to_value``: address of the to be updated value.

   .. deprecated:: 4.5
      Use :cpp:func:`atomic_inc` instead.

.. cpp:function:: template<class T> void atomic_lshift(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``*ptr_to_value << shift;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``shift``: value by which to shift the original variable.

.. cpp:function:: template<class T> void atomic_max(T* const ptr_to_value, const T value);

   Atomically executes ``if (value > *ptr_to_value) *ptr_to_value = value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the maximum with.

.. cpp:function:: template<class T> void atomic_min(T* const ptr_to_value, const T value);

   Atomically executes ``if (value < *ptr_to_value) *ptr_to_value = value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the minimum with.

.. cpp:function:: template<class T> void atomic_mod(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value %= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be used as modulus.

.. cpp:function:: template<class T> void atomic_mul(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value *= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be multiplied.

.. cpp:function:: template<class T> void atomic_nand(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value = ~(*ptr_to_value & value)``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T> void atomic_or(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value |= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T> void atomic_rshift(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``*ptr_to_value >> shift;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``shift``: value by which to shift the original variable.

.. cpp:function:: template<class T> void atomic_sub(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value -= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be subtracted.

.. cpp:function:: template<class T> void atomic_xor(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value ^= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.
