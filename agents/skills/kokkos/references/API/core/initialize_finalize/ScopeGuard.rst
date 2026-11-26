``ScopeGuard``
==============

.. role:: cpp(code)
   :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::ScopeGuard guard(Kokkos::InitializationSettings()
                                .set_map_device_id_by("random")
                                .set_num_threads(1));
    Kokkos::ScopeGuard guard;


``ScopeGuard`` is a class to initialize and finalize Kokkos using `RAII
<https://en.cppreference.com/w/cpp/language/raii>`_.
It calls :cpp:func:`initialize` with the provided arguments in the constructor
and :cpp:func:`finalize` in the destructor.

Interface
---------

.. cpp:class:: ScopeGuard

    A class calling :cpp:func:`initialize` at the start of its lifetime and
    :cpp:func:`finalize` at the end of its lifetime.

    .. cpp:function:: template <class... Args> ScopeGuard(Args&&... args);

        :param args: arguments to pass to :cpp:func:`initialize`

	Possible implementation:

	.. code-block:: cpp

	   template <class... Args> ScopeGuard(Args&&... args){
             initialize(std::forward<Args>(args)...);
           }

    .. cpp:function:: ~ScopeGuard();

       Possible implementation:

       .. code-block:: cpp

	  ~ScopeGuard() { finalize(); }

    .. cpp:function:: ScopeGuard(ScopeGuard const&) = delete;

       Copy constructor

    .. cpp:function:: ScopeGuard(ScopeGuard&&) = delete;

       Move constructor

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard const&) = delete;

       Copy assignment operator

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard&&) = delete;

       Move assignment operator

Notes
-----

.. caution::

  Using ``ScopeGuard`` is mutually exclusive with calling
  :cpp:func:`initialize` and :cpp:func:`finalize` directly.
  Furthermore, only a single ``ScopeGuard`` object can be created during the
  lifetime of the program, and most Kokkos functionality can only be used
  during the lifetime of that object.

  .. code-block:: cpp

     Kokkos::ScopeGuard(argc, argv);  // Temporary object get destroyed immediately and
     //                ^                 the Kokkos execution environment is finalized with it
     //                Forgot to define a named variable
     Kokkos::View<int> v("v");  // ERROR Kokkos finalized

.. note::

  ``ScopeGuard`` unconditionally forwards the provided
  arguments to :cpp:func:`initialize`, which means they have the same
  preconditions.  Until version 3.7, ``ScopeGuard`` was calling
  :cpp:func:`initialize` in its constructor if and only if :cpp:func:`is_initialized` would return
  ``false``, and it was calling :cpp:func:`finalize` in its destructor if and only if it
  called :cpp:func:`initialize` in its constructor.

  We dropped support for the old behavior.  If you think you really need it, you may do:

  .. code-block:: cpp

    auto guard = Kokkos::is_initialized()
                     ? std::make_optional<Kokkos::ScopeGuard>()
                     : std::nullopt;

Example
-------

.. code-block:: cpp

    int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);
        Kokkos::View<double*> my_view("my_view", 10);
        // my_view destructor called before Kokkos::finalize
        // ScopeGuard destructor called, calls Kokkos::finalize
    }


See also
--------

.. seealso::

  :doc:`initialize`
    Start the Kokkos execution environment.
  :doc:`finalize`
    Terminate the Kokkos execution environment.
