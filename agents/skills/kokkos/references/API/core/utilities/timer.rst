``Timer``
=========

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Timer.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::Timer timer;
    double time = timer.seconds();
    timer.reset();

Interface
---------

.. cpp:class:: Timer

   A high-resolution timer class for measuring elapsed time.

   .. note::

        This class is intended for "quick and dirty" timing as well as situations where
        timing is meant to be "always on". For serious performance profiling, it is
        recommended to use the **Kokkos Tools** API. Kokkos Tools provides the
        flexibility to enable or disable profiling at runtime without modifying
        your application, avoiding the need to clutter your code with explicit
        timer objects.

   .. cpp:function:: Timer()

      Constructs a new Timer instance and immediately starts the clock.

      The timer is initialized with the current time, marking the beginning of
      the measurement period.

   .. cpp:function:: double seconds() const

      :returns: The number of seconds that have elapsed since the timer was
        started or last reset.

   .. cpp:function:: void reset()

      Resets the timer, setting the start time to the current time.

      This function effectively restarts the measurement period without
      creating a new Timer object.

   .. cpp:function:: Timer(Timer const&&) = delete
   .. cpp:function:: Timer(Timer&&) = delete

      The Timer class is neither copy-constructible nor copy-assignable.


Example
-------

.. code-block:: cpp

    Timer timer;
    // ...
    double time1 = timer.seconds();
    timer.reset();
    // ...
    double time2 = timer.seconds();
