``push_finalize_hook``
======================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::push_finalize_hook(func);

Registers the callable object ``func`` to be called when the Kokkos execution
environment is terminated.

The functions registered via ``push_finalize_hook()`` will be called in reverse
order when entering :cpp:func:`finalize`, before releasing acquired resources
and finalizing all backends.

If a function exits via a thrown exception, ``std::terminate`` is called.

Interface
---------

.. cpp:Function:: void push_finalize_hook(std::function<void()> func);

   Register the function object ``func`` to be called when entering
   :cpp:func:`finalize`




Example
-------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <iostream>

    void my_hook() {
      std::cout << "Cruel world!\n";
    }

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        Kokkos::push_finalize_hook(my_hook);
        Kokkos::push_finalize_hook([]{ std::cout << "Goodbye\n"; });
        std::cout << "Calling Kokkos::finalize() ...\n";
        Kokkos::finalize();
    }


Output:

.. code-block::

    Calling Kokkos::finalize() ...
    Goodbye
    Cruel world!


See also
--------

.. seealso::

  :doc:`finalize`
    Terminate the Kokkos execution environment
