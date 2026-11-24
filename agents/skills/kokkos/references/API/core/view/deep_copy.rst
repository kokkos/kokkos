``deep_copy``
=============

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::deep_copy(exec_space, dest, src);
   Kokkos::deep_copy(dest, src);

Copies data from ``src`` to ``dest``, where ``src`` and ``dest``
can be `Kokkos::Views <view.rst>`_ or scalars under certain circumstances.

If the provided execution space can access ``src`` and ``dest`` (for
the three-argument overload) or the execution space corresponding to
the source or destination's memory space can access both views,
deep_copy() can convert value types and different layouts by assigning
entry by entry. Otherwise, value types and layouts have to be
identical and contiguous as a memcopy is performed.

Interface
---------

.. cpp:function:: template <class ExecSpace, class ViewDest, class ViewSrc> void Kokkos::deep_copy(const ExecSpace& exec_space, const ViewDest& dest, const ViewSrc& src);

.. cpp:function:: template <class ExecSpace, class ViewDest> void Kokkos::deep_copy(const ExecSpace& exec_space, const ViewDest& dest, const typename ViewDest::value_type& src);

.. cpp:function:: template <class ExecSpace, class ViewSrc> void Kokkos::deep_copy(const ExecSpace& exec_space, ViewSrc::value_type& dest, const ViewSrc& src);

.. cpp:function:: template <class ViewDest, class ViewSrc> void Kokkos::deep_copy(const ViewDest& dest, const ViewSrc& src);

.. cpp:function:: template <class ViewDest> void Kokkos::deep_copy(const ViewDest& dest, const typename ViewDest::value_type& src);

.. cpp:function:: template <class ViewSrc> void Kokkos::deep_copy(ViewSrc::value_type& dest, const ViewSrc& src);

Parameters
~~~~~~~~~~

* ExecSpace: An `ExecutionSpace <../execution_spaces.rst>`_

* ViewDest: A `view-like type <view_like.rst>`_ with a non-const ``value_type``

* ViewSrc: A `view-like type <view_like.rst>`_.

Requirements
~~~~~~~~~~~~

* If ``src`` and ``dest`` are `Kokkos::View <view.rst>`_ s, then all the following are true:

  - ``src.rank() == dest.rank()``

  - For all ``k`` in ``[0, dest.rank)`` ``dest.extent(k) == src.extent(k)`` (or the same as ``dest.rank()``)

  - One of the following set of conditions must be true:

    - Either ``std::is_assignable_v<ViewSrc::value_type, DestView::value_type>`` and there exists an `ExecutionSpace <../execution_spaces.rst>`_ ``copy_space`` (either given or defaulted) such that both ``SpaceAccessibility<copy_space, ViewDest::memory_space>::accessible == true`` and ``SpaceAccessibility<copy_space,ViewSrc::memory_space>::accessible == true``.

    - Or ``src.span_is_contiguous() && dest.span_is_contiguous()``, ``std::is_same_v<ViewDest::array_layout,ViewSrc::array_layout>``, and ``std::is_same_v<DestView::value_type, SrcView::non_const_value_type>``.

* If ``src`` is a `Kokkos::View <view.rst>`_ and ``dest`` is a scalar, then ``src.rank == 0`` is true.

Semantics
---------

* If no `ExecutionSpace <../execution_spaces.rst>`_ argument is provided, all outstanding operations (kernels, copy operation) in any execution spaces will be finished before the copy is executed, and the copy operation is finished before the call returns.

* If an `ExecutionSpace <../execution_spaces.rst>`_ argument ``exec_space`` is provided the call is potentially asynchronous—i.e., the call returns before the copy operation is executed. In that case the copy operation will occur only after any already submitted work to ``exec_space`` is finished, and the copy operation will be finished before any work submitted to ``exec_space`` after the ``deep_copy`` call returns is executed. Note: the copy operation is only synchronous with respect to work in the specific execution space instance, but not necessarily with work in other instances of the same type. This behaves analogous to issuing a ``cudaMemcpyAsync`` into a specific CUDA stream, without any additional synchronization.

Examples
--------

Some things you can and cannot do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cstdio>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        {
            int N = argc > 1 ? atoi(argv[1]) : 12;
            if (N < 6) N = 12;

            // Contiguous Device View
            Kokkos::View<int**, Kokkos::LayoutLeft> d_a("A", N, 10);
            // Deep Copy Scalar into every element of a view
            Kokkos::deep_copy(d_a, 3);

            // Non Contiguous Device View
            auto d_a_2 = Kokkos::subview(d_a, 2, Kokkos::ALL);
            // Deep Copy Scalar into every element of a non-contiguous view
            Kokkos::deep_copy(d_a_2, 5);
            // Non Contiguous Device View
            auto d_a_5 = Kokkos::subview(d_a, 5, Kokkos::ALL);
            // Deep Copy between two non-contiguous views with a common execution space
            Kokkos::deep_copy(d_a_2, d_a_5);

            // Contiguous Host View
            auto h_a = Kokkos::create_mirror_view(d_a);
            // Deep Copy contiguous views
            Kokkos::deep_copy(h_a, d_a);

            // Non Contiguous Host Views
            auto h_a_2 = Kokkos::subview(h_a, 2, Kokkos::ALL);
            // Deep Copy between two non-contiguous views with potentially no common
            // execution space This fails for example if you compile the code with Cuda
            // Kokkos::deep_copy(h_a_2, d_a_2);

            // A Scalar View
            auto d_a_2_5 = Kokkos::subview(d_a, 2, 5);
            int scalar;
            // Deep Copy Scalar View into a scalar
            Kokkos::deep_copy(scalar, d_a_2_5);
        }
        Kokkos::finalize();
    }

How to get layout incompatible views copied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To move data from a device view to an incompatible host view, the operation
needs to be done in two steps by using a temporary allocation:

.. code-block:: cpp

    #include<Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc,argv);
        {
            int N = argc>1?atoi(argv[1]):1000000;
            int R = argc>2?atoi(argv[2]):10;

            // Create two views with different Layouts
            Kokkos::View<int**[5], Kokkos::LayoutLeft> d_view("DeviceView",N,R);
            Kokkos::View<int**[5], Kokkos::LayoutRight, Kokkos::HostSpace> h_view("HostView",N,R);

            // This would fail for example in a CUDA or HIP build:
            // Kokkos::deep_copy(d_view,h_view);

            // To copy two views with incompatible layouts between devices we need a temporary
            auto h_view_tmp = Kokkos::create_mirror_view(d_view);

            // This inherits the Layout from d_view
            static_assert(std::is_same<decltype(h_view_tmp)::array_layout,
                                       Kokkos::LayoutLeft>::value);

            // This now works since h_view_tmp and h_view are both accessible
            // from HostSpace::execution_space
            Kokkos::deep_copy(h_view_tmp,h_view);

            // Now we can copy from h_view_tmp to d_view since they are Layout compatible
            // If we just compiled for OpenMP this is a no-op since h_view_tmp and d_view
            // would reference the same data.
            Kokkos::deep_copy(d_view,h_view_tmp);
        }
        Kokkos::finalize();
    }
