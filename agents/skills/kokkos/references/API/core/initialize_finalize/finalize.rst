``finalize``
============

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::finalize();

Terminates the Kokkos execution environment.
This functions cleans up all Kokkos states and released the associated
resources.
Once this function is called, no Kokkos API functions (not even
:cpp:func:`initialize`) may be called, except for :cpp:func:`is_initialized` or
:cpp:func:`is_finalized`.
The user must ensure that all Kokkos objects (e.g. :cpp:class:`View`) are destroyed
before ``finalize`` gets called.

Programs are ill-formed if they do not call this function after calling
:cpp:func:`initialize`, before program termination.

Interface
---------

.. cpp:function:: void finalize();

   :preconditions:
     * :cpp:func:`is_initialized` returns ``true``
     * :cpp:func:`is_finalized` returns ``false``

Examples
--------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        {  // scope to ensure that my_view destructor is called before Kokkos::finalize
            Kokkos::View<double*> my_view("my_view", 10);
        }  // scope of my_view ends here
        Kokkos::finalize();
    }

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cstdlib>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        std::atexit(Kokkos::finalize); // register to be called on program termination
        Kokkos::View<double*> my_view("my_view", 10);
    } // my_view is properly destructed before Kokkos::finalize


See also
--------

.. seealso::

  :doc:`ScopeGuard`
    A RAII-based approach to ensure initialization and finalization are handled
    correctly.
  :doc:`push_finalize_hook`
    Register a function to be called on finalize() invocation.
  :doc:`is_initialized_or_finalized`
    Query the current state of the Kokkos execution environment.
