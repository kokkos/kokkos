``initialize``
==============

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage 
-----

.. code-block:: cpp

    Kokkos::initialize(argc, argv);
    Kokkos::initialize(Kokkos::InitializationSettings()
                           .set_disable_warnings(true)
                           .set_num_threads(8)
                           .set_map_device_id_by("random"));
    Kokkos::initialize();

Initializes the Kokkos execution environment.
This function must be called before any other Kokkos API functions or
constructors.  There are a small number of exceptions, such as
:cpp:func:`is_initialized` or :cpp:func:`is_finalized`.
Kokkos can be initialized at most once; subsequent calls are erroneous.

The function has two overloads.
The first one takes the same two parameters as ``main()`` corresponding to
the command line arguments passed to the program from the environment in which
the program is run.  Kokkos parses the arguments for the flags that it
recognizes.  Whenever a Kokkos flag is seen, it is removed from ``argv``, and
``argc`` is decremented.
The second one takes a :cpp:class:`InitializationSettings` class object
which allows for programmatic control of arguments.

Interface
---------

.. cpp:function:: void initialize(int& argc, char* argv[]);
.. cpp:function:: void initialize(const InitializationSettings& settings = {});

   :param argc: Non-negative value, representing the number of command line
     arguments passed to the program.

   :param argv: Pointer to the first element of an array of ``argc + 1``
     pointers, of which the last one is null and the previous, if any, point to
     null-terminated multibyte strings that represent the arguments passed to
     the program.

   :param settings: ``class`` object that contains settings to control the
     initialization of Kokkos.

   :preconditions:
     * :cpp:func:`is_initialized` returns ``false``
     * :cpp:func:`is_finalized` returns ``false``


Note
----
.. important::

   ``Kokkos::initialize`` generally should be called after ``MPI_Init`` when
   Kokkos is initialized within an MPI context.

Example
-------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        {  // scope to ensure that my_view destructor is called before Kokkos::finalize
            Kokkos::View<double*> my_view("my_view", 10);
        }  // scope of my_view ends here
        Kokkos::finalize();
    }    

See also
--------

.. seealso::

  :doc:`finalize`
    Terminate the Kokkos execution environment.
  :doc:`ScopeGuard`
    A RAII-based approach to ensure initialization and finalization are handled
    correctly.
  :doc:`is_initialized_or_finalized`
    Query the current state of the Kokkos execution environment.
