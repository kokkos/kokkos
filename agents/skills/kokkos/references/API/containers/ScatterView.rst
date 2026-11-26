``ScatterView``
===============

.. role:: cpp(code)
	:language: cpp

Header File: ``<Kokkos_ScatterView.hpp>``

.. warning::

   ``ScatterView`` is still in the namespace ``Kokkos::Experimental``


.. _parallelReduce: ../core/parallel-dispatch/parallel_reduce.rst

.. |parallelReduce| replace:: :cpp:func:`parallel_reduce`

.. _View: ../core/view/view.rst

.. |View| replace:: ``View``

.. |reset| replace:: ``reset()``

.. |access| replace:: ``access()``

.. |contribute| replace:: ``contribute()``

.. |create_scatter_view| replace:: ``create_scatter_view()``

Usage
-----

A Kokkos ScatterView serves as an interface for a standard Kokkos::|View|_ implementing a scatter-add pattern either via atomics or data replication.

Construction of a ScatterView can be expensive, so you should try to reuse the same one if possible, in which case, you should call |reset|_ between uses.

ScatterView can not be addressed directly: each thread inside a parallel region needs to make a call to |access|_ and access the underlying View through the return value of |access|_.

Following the parallel region, a call to the free function Kokkos::Experimental::|contribute|_ should be made to perform the final reduction.

Interface
---------
.. code-block:: cpp

    template <typename DataType [, typename Layout [, typename ExecSpace [, typename Operation [, typename Duplication [, typename Contribution]]]]]>
    class ScatterView

Parameters
~~~~~~~~~~
Template parameters other than ``DataType`` are optional, but if one is specified, preceding ones must also be specified.
That means for example that ``Operation`` can be omitted but if it is specified, ``Layout`` and ``ExecSpace`` must also be specified.

* ``DataType``, ``Layout`` and ``ExecSpace`` need to be the same types as the one from the Kokkos::View this ScatterView is interfacing.

* ``Operation``:
  Can take the values:

  - ``Kokkos::Experimental::ScatterSum``: performs a Sum. It is the default value.

  - ``Kokkos::Experimental::ScatterProd``: performs a Multiplication.

  - ``Kokkos::Experimental::ScatterMin``: takes the min.

  - ``Kokkos::Experimental::ScatterMax``: takes the max.

* ``Duplication``:
  Whether to duplicate the grid or not; defaults to ``Kokkos::Experimental::ScatterDuplicated``, other option is ``Kokkos::Experimental::ScatterNonDuplicated``.

* ``Contribution``:
  Whether to contribute to use atomics; defaults to ``Kokkos::Experimental::ScatterAtomics``, other option is ``Kokkoss::Experimental::ScatterNonAtomic``.

Creating a ScatterView with non default ``Operation``, ``Duplication`` or ``Contribution`` using this interface can become complicated, because you need to specify the exact type for ``DataType``, ``Layout`` and ``ExecSpace``. This is why it is advised that you instead use the function Kokkos::Experimental::|create_scatter_view|_.

Description
-----------

.. cpp:class:: template <typename DataType, typename Layout, typename ExecSpace, typename Op, typename Duplication, typename Contribution> ScatterView

    .. rubric:: Public Member Variables

    .. cpp:type:: original_view_type

        Type of View passed to ScatterView constructor.

    .. cpp:type:: original_value_type

        Value type of the original_view_type.

    .. cpp:type:: original_reference_type

        Reference type of the original_view_type.

    .. cpp:type:: data_type_info

        DuplicatedDataType, a newly created DataType that has a new runtime dimension which becomes the largest-stride dimension, from the given View DataType.

    .. cpp:type:: internal_data_type

        Value type of data_type_info.

    .. cpp:type:: internal_view_type

        A View type created from the internal_data_type.

    .. rubric:: Constructors

    .. cpp:function:: ScatterView()

        The default constructor. Default constructs members.

    .. cpp:function:: ScatterView(View<RT, RP...> const&)

        Constructor from a ``Kokkos::View``. ``internal_view`` member is copy constructed from this input view.

    .. cpp:function:: ScatterView(std::string const& name, Dims ... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.

    .. cpp:function:: ScatterView(ALLOC_PROP const& arg_prop, Dims... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.
        This constructor allows passing an object created by ``Kokkos::view_alloc`` as first argument, e.g., for specifying an execution space via
        ``Kokkos::view_alloc(exec_space, "label")``.

    .. rubric:: Public Methods

    .. cpp:function:: constexpr bool is_allocated() const

        :return: true if the ``internal_view`` points to a valid memory location. This function works for both managed and unmanaged views. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

    .. _access:

    .. cpp:function:: access() const

       use within a kernel to return a ``ScatterAccess`` member; this member accumulates a given thread's contribution to the reduction.

    .. cpp:function:: subview() const

        :return: a subview of a ``ScatterView``

    .. cpp:function:: contribute_into(View<DT, RP...> const& dest) const

       contribute ``ScatterView`` array's results into the input View ``dest``

    .. _reset:

    .. cpp:function:: reset()

       performs reset on destination array

    .. cpp:function:: reset_except(View<DT, RP...> const& view)

       tbd

    .. cpp:function:: resize(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with copying old data to new data at the corresponding indices

    .. cpp:function:: realloc(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with discarding old data


    .. rubric:: *Private* Members

    :member: typedef original_view_type internal_view_type;
    :member: internal_view_type internal_view;


.. rubric:: Free Functions

.. _create_scatter_view:

.. cpp:function:: template <typename Operation, typename Duplication, typename Contribution> create_scatter_view(const View<DT1, VP...>& view)

   create a new ScatterView interfacing the View ``view``.
   Default value for ``Operation`` is ``Kokkos::Experimental::ScatterSum``, ``Duplication`` and ``Contribution`` are chosen to make the ScatterView as efficient as possible when running on its ``ExecSpace``.

.. _contribute:

.. cpp:function:: contribute(View<DT1, VP...>& dest, Kokkos::Experimental::ScatterView<DT2, LY, ES, OP, CT, DP> const& src)

   convenience function to perform final reduction of ScatterView
   results into a resultant View; may be called following |parallelReduce|_.


Example
-------

.. code-block:: cpp


    #include <Kokkos_Core.hpp>
    #include <Kokkos_ScatterView.hpp>

    KOKKOS_INLINE_FUNCTION int foo(int i) { return i; }
    KOKKOS_INLINE_FUNCTION double bar(int i) { return i*i; }

    int main (int argc, char* argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);

        Kokkos::View<double*> results("results", 1);
        auto scatter = Kokkos::Experimental::create_scatter_view(results);
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(int input_i) {
            auto access = scatter.access();
            auto result_i = foo(input_i);
            auto contribution = bar(input_i);
            access(result_i) += contribution;
        });
        Kokkos::Experimental::contribute(results, scatter);
    }
