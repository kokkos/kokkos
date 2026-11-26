``ExecutionPolicy``
===================

.. role::cpp(code)
    :language: cpp

The concept of an ``ExecutionPolicy`` is the fundamental abstraction to represent "how" the execution of a Kokkos parallel pattern takes place.  This page talks practically about how to *use* the common features of execution policies in Kokkos; for a more formal and theoretical treatment, see `this document <../KokkosConcepts.rst>`_.

    *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

What is an ``ExecutionPolicy``?
-------------------------------

The dominant parallel dispatch mechanism in Kokkos, described `elsewhere in the programming guide <../../../ProgrammingGuide/ParallelDispatch.rst>`_, involves a ``parallel_pattern`` (e.g., something like `Kokkos::parallel_for <../parallel-dispatch/parallel_for.rst>`_ or `Kokkos::parallel_reduce <../parallel-dispatch/parallel_reduce.rst>`_), an ``ExecutionPolicy``, and a ``Functor``.  In a hand-wavy sense:

.. code-block:: cpp
        
    parallel_pattern(
        ExecutionPolicy(),
        Functor()
    );

The most basic ("beginner") case is actually a shortcut:

.. code-block:: cpp

    Kokkos::parallel_for(
        42,
        KOKKOS_LAMBDA (int n) { /* ... */ }
    );

is a "shortcut" for

.. code-block:: cpp

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            Kokkos::DefaultExecutionSpace(), 0, 42
        ),
        KOKKOS_LAMBDA(int n) { /* ... */ }
    );

In this example, ``Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>`` is the ``ExecutionPolicy`` type.

Functionality
~~~~~~~~~~~~~

All ``ExecutionPolicy`` types provide a nested type named ``index_type``.
