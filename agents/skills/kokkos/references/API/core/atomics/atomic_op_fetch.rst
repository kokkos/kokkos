``atomic_[op]_fetch``
=====================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    new_value =  atomic_[op]_fetch(ptr_to_value,update_value);

Atomically updates the variable at the address given by ``ptr_to_value`` with ``update_value`` according to the relevant operation, and returns the updated value found at that address.

Description
-----------

.. cpp:function:: template<class T> T atomic_add_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value += value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be added.

.. cpp:function:: template<class T> T atomic_and_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value &= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T>  T atomic_dec_fetch(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)--; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the value to be updated

.. cpp:function:: template<class T> T atomic_div_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value /= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value by which to divide the original value..

.. cpp:function:: template<class T>  T atomic_inc_fetch(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)++; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the value to be updated

.. cpp:function:: template<class T> T atomic_lshift_fetch(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``*ptr_to_value << shift; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``shift``: value by which to shift the original variable.

.. cpp:function:: template<class T> T atomic_max_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value = max(*ptr_to_value, value); return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the maximum with.

.. cpp:function:: template<class T> T atomic_min_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value = min(*ptr_to_value, value); return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the minimum with.

.. cpp:function:: template<class T> T atomic_mul_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value *= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value by which to multiply the original value.

.. cpp:function:: template<class T> T atomic_mod_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value %= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be used as modulus.

.. cpp:function:: template<class T> T atomic_nand_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value = ~(*ptr_to_value & val); return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T> T atomic_or_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value |= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cpp:function:: template<class T> T atomic_rshift_fetch(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``*ptr_to_value >> shift; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``shift``: value by which to shift the original variable.

.. cpp:function:: template<class T> T atomic_sub_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value -= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be subtracted.

.. cpp:function:: template<class T> T atomic_xor_fetch(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value ^= value; return *ptr_to_value;``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.
