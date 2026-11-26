.. _kokkos_if_on_host_device_macros:

=================================================
``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE``
=================================================

.. role:: cpp(code)
   :language: cpp

Overview
========
The ``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` macros are a pair of
function-like macros introduced in Kokkos 3.6 that enable **portable
conditional compilation** within a **single ``KOKKOS_FUNCTION`` body**. They
allow you to select which code is compiled and executed based on whether the
function is running on the host (CPU) or a device (GPU, etc.). These macros
provide an alternative to non-portable preprocessor idioms like #ifdef
``__CUDA_ARCH__``.

Motivation
==========
Traditional preprocessor directives like ``#ifdef __CUDA_ARCH__`` rely on a
split compilation model, where host and device code are compiled in separate
passes.  While this model is supported by some compilers (like ``nvcc``), it is
not universally portable.  Other modern compilers for GPU-accelerated code,
such as those that support OpenACC, use a unified compilation
approach where both host and device code are compiled in a single pass. As a
result, code written with backend-specific macros is not portable across
different compilers and programming models

The ``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` macros solve this
portability problem by allowing the compiler to conditionally compile code
within a single pass, enabling a single code base to be used with a wider range
of compilers and backends.

Usage
=====
These macros are designed to be used within a function decorated with
``KOKKOS_FUNCTION``. They accept a single argument, which is a block of code
enclosed in double parentheses. The code inside the macro's parentheses will
only be compiled and executed on the specified architecture


Signature
---------

.. code-block:: cpp

    KOKKOS_IF_ON_HOST(( /* code to be compiled on the host */ ))
    KOKKOS_IF_ON_DEVICE(( /* code to be compiled on the device */ ))


Example: Host/Device Overloading
--------------------------------

A common use case is to provide different implementations of a function for
host and device execution.

.. code-block:: cpp

    struct MyS { int i; };

    KOKKOS_FUNCTION MyS MakeStruct() {
      // This return statement is only compiled for the host target.
      KOKKOS_IF_ON_HOST((
        return MyS{0};
      ))

      // This return statement is only compiled for the device target.
      KOKKOS_IF_ON_DEVICE((
        return MyS{1};
      ))
    }

Important Considerations
========================

Scope
-----

Each ``KOKKOS_IF_ON_*`` macro introduces a new scope. Any variables declared
within the macro's parentheses are local to that scope and will not be
accessible outside of it.

.. code-block:: cpp

    KOKKOS_IF_ON_HOST((
      int x = 0; // 'x' is only visible within this scope
      std::cout << x << '\n';
    )) // The scope of 'x' ends here.

``constexpr`` Context
---------------------

These macros cannot be used in a context that requires a ``constexpr``
(constant expression).

Best Practices
--------------

**Avoid using these macros whenever possible.**

``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` should be considered a **last
resort** for code differentiation. The primary goal of Kokkos is to achieve
high-performance portability through a unified code base. Relying on these
macros can hinder this goal by introducing host/device-specific logic.

Before using these macros, consider alternative approaches like **partial
template specialization** on execution spaces or using Kokkos's built-in
functionalities, which are designed to be portable across all backends. Using
these macros should be limited to situations where a fundamental difference
between host and device APIs necessitates separate code paths, such as for I/O
operations or specific backend intrinsics.
