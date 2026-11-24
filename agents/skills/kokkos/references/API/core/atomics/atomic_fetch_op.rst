``atomic_fetch_[op]``
=====================

.. role:: cpp(code)
   :language: cpp

Header File: <Kokkos_Core.hpp>

Usage
-----

.. code-block:: cpp

   old_value =  atomic_fetch_[op](ptr_to_value, update_value);

Atomically updates the variable at the address given by ``ptr_to_value`` with ``update_value``
according to the relevant operation ``op``, and returns the previous value found at that address.

Description
-----------

.. cpp:function:: template<class T> T atomic_fetch_add(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value += value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value to be added

.. cpp:function:: template<class T> T atomic_fetch_and(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value &= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value with which to combine the original value.

.. cpp:function:: template<class T>  T atomic_fetch_dec(T* const ptr_to_value);

   Atomically executes ``tmp = *ptr_to_value; (*ptr_to_value)--; return tmp;``.

   :param ptr_to_value: address of the value to be updated

.. cpp:function:: template<class T>  T atomic_fetch_div(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value /= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value by which to divide the original value..

.. cpp:function:: template<class T>  T atomic_fetch_inc(T* const ptr_to_value);

   Atomically executes ``tmp = *ptr_to_value; (*ptr_to_value)++; return tmp;``.

   :param ptr_to_value: address of the value to be updated

.. cpp:function:: template<class T> T atomic_fetch_lshift(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value << shift; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param shift: value by which to shift the original variable.

.. cpp:function:: template<class T> T atomic_fetch_max(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value = max(*ptr_to_value, value); return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value which to take the maximum with.

.. cpp:function:: template<class T> T atomic_fetch_min(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value = min(*ptr_to_value, value); return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value which to take the minimum with.

.. cpp:function:: template<class T> T atomic_fetch_mul(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value *= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value by which to multiply the original value.

.. cpp:function:: template<class T> T atomic_fetch_mod(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value %= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value to be used as modulus.

.. cpp:function:: template<class T> T atomic_fetch_nand(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value = ~(*ptr_to_value & val); return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value with which to combine the original value.

.. cpp:function:: template<class T> T atomic_fetch_or(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value |= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value with which to combine the original value.

.. cpp:function:: template<class T> T atomic_fetch_rshift(T* const ptr_to_value, const unsigned shift);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value >> shift; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param shift: value by which to shift the original variable.

.. cpp:function:: template<class T> T atomic_fetch_sub(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value -= value``.

   :param ptr_to_value: address of the value to be updated
   :param value: value to be subtracted..

.. cpp:function:: template<class T> T atomic_fetch_xor(T* const ptr_to_value, const T value);

   Atomically executes ``tmp = *ptr_to_value; *ptr_to_value ^= value; return tmp;``.

   :param ptr_to_value: address of the value to be updated
   :param value: value with which to combine the original value.
