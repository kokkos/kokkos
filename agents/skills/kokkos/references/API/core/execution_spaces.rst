.. _api-execution-spaces:

Execution Spaces
================

.. role:: cpp(code)
    :language: cpp

.. _ExecutionSpaceConceptType: #kokkos-executionspaceconcept

.. |ExecutionSpaceConceptType| replace:: :cpp:func:`ExecutionSpace` type

.. _DocExecutionSpaceConcept: #kokkos-executionspaceconcept

.. |DocExecutionSpaceConcept| replace:: the documentation on the :cpp:func:`ExecutionSpace` concept

.. _Experimental: utilities/experimental.md#experimentalnamespace

.. |Experimental| replace:: Experimental

.. _KokkosConcepts: KokkosConcepts.rst

.. |KokkosConcepts| replace:: this document

.. _ExecutionSpaceS: #kokkos-executionspaceconcept

.. |ExecutionSpaceS| replace:: :cpp:func:`ExecutionSpace` s

.. _MemorySpace: memory_spaces.rst#kokkos-memoryspaceconcept

.. |MemorySpace| replace:: :cpp:func:`MemorySpace`

.. _KokkosSpaceAccessibility: SpaceAccessibility.rst

.. |KokkosSpaceAccessibility| replace:: :cpp:func:`Kokkos::SpaceAccessibility`

.. _KokkosTeamPolicy: policies/TeamPolicy.rst

.. |KokkosTeamPolicy| replace:: :cpp:func:`Kokkos::TeamPolicy`

.. _ExecutionSpaceConcept: #kokkos-executionspaceconcept

.. |ExecutionSpaceConcept| replace:: :cpp:func:`ExecutionSpaceConcept`

``Kokkos::Cuda``
----------------

``Kokkos::Cuda`` is an |ExecutionSpaceConceptType|_ representing execution on a Cuda device.
Except in rare instances, it should not be used directly, but instead should be used generically as an execution space. For details, see |DocExecutionSpaceConcept|_.


``Kokkos::HIP``
---------------

``Kokkos::HIP`` :sup:`promoted from` |Experimental|_ :sup:`since 4.0` is an |ExecutionSpaceConceptType|_ representing
execution on a device supported by HIP. Except in rare instances, it should not be used directly,
but instead should be used generically as an execution space. For details, see |DocExecutionSpaceConcept|_.

``Kokkos::SYCL``
------------------------------

``Kokkos::SYCL`` :sup:`promoted from` |Experimental|_ :sup:`since 4.5` is an |ExecutionSpaceConceptType|_ representing execution on a device supported by SYCL.

If the SYCL backend is enabled and no GPU architecture is specified, Kokkos will use Just-In-Time compilation without any restriction to a particular SYCL device type.
Thus, this is the only option to target a CPU with the SYCL backend (which is experimental, untested, and not optimized for).

``Kokkos::HPX``
---------------

``Kokkos::HPX`` is an |ExecutionSpaceConceptType|_ representing execution with the HPX runtime system.
Except in rare instances, it should not be used directly, but instead should be used generically as an execution space.
For details, see |DocExecutionSpaceConcept|_.

``Kokkos::OpenMP``
------------------

``Kokkos::OpenMP`` is an |ExecutionSpaceConceptType|_ representing execution with the OpenMP runtime system.
Except in rare instances, it should not be used directly, but instead should be used
generically as an execution space. For details, see |DocExecutionSpaceConcept|_.

``Kokkos::Threads``
-------------------

``Kokkos::Threads`` is an |ExecutionSpaceConceptType|_ representing parallel execution with std::threads.
Except in rare instances, it should not be used directly, but instead should be used
generically as an execution space. For details, see |DocExecutionSpaceConcept|_.

``Kokkos::Serial``
------------------

``Kokkos::Serial`` is an |ExecutionSpaceConceptType|_ representing serial execution the CPU.
Except in rare instances, it should not be used directly, but instead should be
used generically as an execution space. For details, see |DocExecutionSpaceConcept|_.

``Kokkos::ExecutionSpaceConcept``
---------------------------------

The concept of an ``ExecutionSpace`` is the fundamental abstraction to represent the "where" and the "how"
that execution takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept*
of an ``ExecutionSpace`` rather than any specific instance.  This page talks practically about how to *use*
the common features of execution spaces in Kokkos; for a more formal and theoretical treatment, see |KokkosConcepts|_.

  *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates
  in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself,
  which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just
  means "what you're allowed to do with a type that is a template parameter in certain places".

Aliases based on configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Kokkos::DefaultExecutionSpace``
---------------------------------

``Kokkos::DefaultExecutionSpace`` is an alias of |ExecutionSpaceConceptType|_ pointing to an ``ExecutionSpace``
based on the current configuration of Kokkos. It is set to the highest available in the hierarchy ``device,host-parallel,host-serial``.
It also serves as default for optionally specified template parameters of |ExecutionSpaceConceptType|_.

``Kokkos::DefaultHostExecutionSpace``
-------------------------------------

``Kokkos::DefaultHostExecutionSpace`` is an alias of |ExecutionSpaceConceptType|_ pointing to an ``ExecutionSpace`` based
on the current configuration of Kokkos. It is set to the highest available in the hierarchy ``host-parallel,host-serial``.

Very Simplest Use: Not at all?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When first starting to use Kokkos, the (surprising) answer to where you'll see |ExecutionSpaceS|_ used explicitly is "nowhere".
Many of the first things most users learn are "shortcuts" for "do this thing using the default execution space",
which is a type alias (a.k.a., ``typedef``) named ``Kokkos::DefaultExecutionSpace`` defined based on build system flags. For instance,

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

Being more generic
~~~~~~~~~~~~~~~~~~

For more intermediate and advanced users, however, it is often good practice to write code that is
explicitly generic over the execution space, so that calling code can pass in a non-default execution space if needed.
For instance, if the simple version of your function is

.. code-block:: cpp

    void my_function(Kokkos::View<double*> data, double scale) {
        Kokkos::parallel_for(
            data.extent(0),
            KOKKOS_LAMBDA (int n) {
                data(n) *= scale;
            }
        );
    }

then a more advanced, more flexible version of your function might look like:

.. code-block:: cpp

    template <class ExecSpace, class ViewType>
    void my_function(
    ExecSpace ex,
    ViewType data,
    double scale
    ) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, typename ViewType::memory_space>::assignable,
        "Incompatible ViewType and ExecutionSpace"
    );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(ex, 0, data.extent(0)),
        KOKKOS_LAMBDA (int n) {
        data(n) *= scale;
        }
    );
    }

More advanced users may also prefer the more explicit form simply to avoid the additional mental
exercise of translating "shortcuts" when reading the code later. Being explicit about *where* and *how*
Kokkos parallel patterns are executing tends to reduce bugs, even if it is more verbose.

Functionality
~~~~~~~~~~~~~

All ``ExecutionSpace`` types expose a common set of functionality. In generic code that uses Kokkos (which is pretty much all user code),
you should never use any part of an execution space type that isn't common to all execution
space types (otherwise, you risk losing portability of your code). There are a few expressions guaranteed to be
valid for any ``ExecutionSpace`` type. Given a type ``Ex`` that is an ``ExecutionSpace`` type,
and an instance of that type ``ex``, Kokkos guarantees the following expressions will provide the specified functionality:

.. code-block:: cpp

    ex.name();

*Returns:* a value convertible to ``const char*`` that is guaranteed to be unique to a given ``ExecutionSpace`` instance type.
*Note:* the pointer returned by this function may not be accessible from the ``ExecutionSpace`` itself (for instance, on a device); use with caution.

.. code-block:: cpp

    ex.fence(str);

*Effects:* Upon return, all parallel patterns and deep_copy calls executed on the instance ``ex`` are guaranteed to have completed, and their effects are guaranteed visible to the calling thread. The optiopnal ``str`` argument allows customizing the event reported to Kokkos Tools.
*Returns:* Nothing.
*Note:* This *cannot* be called from within a parallel pattern.  Doing so will lead to unspecified effects (i.e., it might work, but only for some execution spaces, so be extra careful not to do it).

.. code-block:: cpp

    ex.print_configuration(ostr);
    ex.print_configuration(ostr, detail);

where ``ostr`` is a ``std::ostream`` (like ``std::cout``, for instance) and ``detail`` is a boolean indicating whether a detailed description should be printed.

*Effects:* Outputs the configuration of ``ex`` to the given ``std::ostream``.
*Returns:* Nothing.
*Note:* This *cannot* be called from within a parallel pattern.

Additionally, the following type aliases (a.k.a. ``typedef`` s) will be defined by all execution space types:

* ``Ex::memory_space``: the default |MemorySpace|_ to use when executing with ``Ex``. Kokkos guarantees that ``Kokkos::SpaceAccessibility<Ex, Ex::memory_space>::accessible`` will be ``true`` (see |KokkosSpaceAccessibility|_)

* ``Ex::array_layout``: the default ``ArrayLayout`` recommended for use with ``View`` types accessed from ``Ex``.

* ``Ex::scratch_memory_space``: the ``ScratchMemorySpace`` that parallel patterns will use for allocation of scratch memory (for instance, as requested by a |KokkosTeamPolicy|_). Only unmanaged Views can be created using this memory space.

Default Constructibility, Copy Constructibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the above functionality, all ``ExecutionSpace`` types in Kokkos are default
constructible (you can construct them as ``Ex ex()``) and copy constructible (you can construct them as ``Ex ex2(ex1)``).
All default constructible instances of an ``ExecutionSpace`` type are guaranteed to have equivalent behavior,
and all copy constructed instances are guaranteed to have equivalent behavior to the instance they were copied from.

Backend Detection
^^^^^^^^^^^^^^^^^

A Kokkos installation always has at least one CPU backend (``Serial``, ``OpenMP``, or ``Threads``) and may have up to one GPU backend (``Cuda``, ``HIP``, or ``SYCL``). Test whether a backend is enabled with the preprocessor macro ``KOKKOS_ENABLE_<BACKEND>``, where ``<BACKEND>`` is the all-uppercase backend name (e.g. ``KOKKOS_ENABLE_CUDA``).

Detection
^^^^^^^^^

Kokkos provides the convenience type trait ``Kokkos::is_execution_space<T>`` which has a ``value`` compile-time
accessible value (usable as ``Kokkos::is_execution_space<T>::value``) that is ``true`` if and only if a type ``T``
meets the requirements of the ``ExecutionSpace`` concept. Any ``ExecutionSpace`` type ``T`` will also
have the expression ``Kokkos::is_space<T>::value`` evaluate to ``true`` as a compile-time constant.

Synopsis
~~~~~~~~

.. code-block:: cpp

    // This is not an actual class, it just describes the concept in shorthand
    class ExecutionSpaceConcept {
    public:
        typedef ExecutionSpaceConcept execution_space;
        typedef ... memory_space;
        typedef Device<execution_space, memory_space> device_type;
        typedef ... scratch_memory_space;
        typedef ... array_layout;
        typedef ... size_type;

        ExecutionSpaceConcept();
        ExecutionSpaceConcept(const ExecutionSpaceConcept& src);

        const char* name() const;
        void print_configuration(std::ostream ostr&) const;
        void print_configuration(std::ostream ostr&, bool details) const;

        int concurrency() const;

        void fence(const std::string&) const;
        void fence() const;

        friend bool operator==(const execution_space& lhs, const execution_space& rhs);
    };

    template<class MS>
    struct is_execution_space {
    enum { value = false };
    };

    template<>
    struct is_execution_space<ExecutionSpaceConcept> {
    enum { value = true };
    };

Typedefs
~~~~~~~~

* ``execution_space``: The self type;

* ``memory_space``: The default |MemorySpace|_ to use when executing with |ExecutionSpaceConcept|_. Kokkos guarantees that ``Kokkos::SpaceAccessibility<Ex, Ex::memory_space>::accessible`` will be ``true`` (see |KokkosSpaceAccessibility|_)

* ``device_type``: ``DeviceType<execution_space,memory_space>``.

* ``array_layout``: The default ``ArrayLayout`` recommended for use with ``View`` types accessed from |ExecutionSpaceConcept|_.

* ``scratch_memory_space``: The ``ScratchMemorySpace`` that parallel patterns will use for allocation of scratch memory (for instance, as requested by a |KokkosTeamPolicy|_). Only unmanaged Views can be created using this memory space.

* ``size_type``: The default integer type associated with this space. Signed or unsigned, 32 or 64 bit integer type, used as preferred type for indexing.

Constructors
~~~~~~~~~~~~

* ``ExecutionSpaceConcept()``: Default constructor.

* ``ExecutionSpaceConcept(const ExecutionSpaceConcept& src)``: Copy constructor.

Functions
~~~~~~~~~

* ``const char* name() const;``: *Returns* the label of the execution space instance.

* ``int concurrency() const;`` *Returns* the maximum amount of concurrently executing work items in a parallel setting, i.e. the maximum number of threads simultaneously utilized by an execution space instance. On CPU this is the number of threads. On GPU this is the maximum number of threads that can be active over the whole device, assuming occupancy is not limited by registers or other resources.

* ``void fence(const std::string& label = unspecified-default-value) const;`` *Effects:* Upon return, all parallel patterns executed on the instance |ExecutionSpaceConcept|_ are guaranteed to have completed, and their effects are guaranteed visible to the calling thread. *Note:* This *cannot* be called from within a parallel pattern. Doing so will lead to unspecified effects (i.e., it might work, but only for some execution spaces, so be extra careful not to do it). The optional ``label`` argument allows customizing the event reported to Kokkos Tools.

* ``void print_configuration(std::ostream ostr) const;``: *Effects:* Outputs the configuration of ``ex`` to the given ``std::ostream``. *Note:* This *cannot* be called from within a parallel pattern.

Non Member Facilities
~~~~~~~~~~~~~~~~~~~~~

* ``template<class MS> struct is_execution_space;``: typetrait to check whether a class is a execution space.

* ``template<class S1, class S2> struct SpaceAccessibility;``: typetraits to check whether two spaces are compatible (assignable, deep_copy-able, accessible). (see |KokkosSpaceAccessibility|_)

* ``bool operator==(const execution_space& lhs, const execution_space& rhs)``: tests whether the two space instances (of the same type) are identical.
