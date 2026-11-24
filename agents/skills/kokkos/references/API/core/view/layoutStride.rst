``LayoutStride``
================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: c++

    Kokkos::View<float***> full_mesh; // an entire mesh
    Kokkos::View<float**, Kokkos::LayoutStride> mesh_subcomponent;
    mesh_subcomponent = Kokkos::subview(full_mesh,Kokkos::ALL(), 0, Kokkos::ALL()); // take x and z components

Description
-----------

.. cpp:class:: LayoutStride

   When provided to a multidimensional View, lays out memory with an arbitrary stride.
   Most frequently encountered when taking a noncontiguous subview of some larger view.

   .. rubric:: Public Class Members

   .. cpp:member:: size_t dimension[ARRAY_LAYOUT_MAX_RANK];

      An array containing the size of each dimension of the Layout

   .. cpp:member:: size_t stride[ARRAY_LAYOUT_MAX_RANK];

      An array containing the stride for each dimension of the Layout

   .. cpp:member:: static constexpr bool is_extent_constructible = false;

      A boolean to allow detection that this class is extent constructible

   .. rubric:: Public Typedefs

   .. cpp:type:: array_layout

      A tag signifying that this models the Layout concept

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION explicit constexpr LayoutStride(size_t N0 = 0, size_t S0 = 0, \
			   size_t N1 = 0, \
                           size_t S1 = 0, size_t N2 = 0, size_t S2 = 0, \
                           size_t N3 = 0, size_t S3 = 0, size_t N4 = 0, \
                           size_t S4 = 0, size_t N5 = 0, size_t S5 = 0, \
                           size_t N6 = 0, size_t S6 = 0, size_t N7 = 0, size_t S7 = 0);

      Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout

   .. cpp:function:: LayoutStride(LayoutStride const&) = default;

      Default copy constructor, element-wise copies the other Layout

   .. cpp:function:: LayoutStride(LayoutStride&&) = default;

      Default move constructor, element-wise moves the other Layout

   .. rubric:: Assignment operators

   .. cpp:function:: LayoutStride& operator=(LayoutStride const&) = default;

      Default copy assignment, element-wise copies the other Layout

   .. cpp:function:: LayoutStride& operator=(LayoutStride&&) = default;

      Default move assignment, element-wise moves the other Layout

   .. rubric:: Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION static LayoutStride order_dimensions(int const rank, \
		   iTypeOrder const* const order, iTypeDimen const* const dimen);

      Calculates the strides given ordered dimensions

Example
-------

Creating a 3D unmanaged strided view around a ptr. (You can also just have a view allocate itself by providing a label)

.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc,argv);
        {
            // Some storage
            int* ptr = new int[80];
            // Creating a layout object
            Kokkos::LayoutStride layout(3,1,3,5,4,20);
            // Create a unmanaged view from a pointer and a layout
            Kokkos::View<int***, Kokkos::LayoutStride, Kokkos::HostSpace> a(ptr,layout);

            // Get strides
            int strides[8];
            a.stride(strides);

            // Print extents and strides
            printf("Extents: %d %d %d\n",a.extent(0),a.extent(1),a.extent(2));
            printf("Strides: %i %i %i\n",strides[0],strides[1],strides[2]);

            // delete storage
            delete [] ptr;
        }
        Kokkos::finalize();
    }

Output:

.. code-block::

    Extents: 3 3 4
    Strides: 1 5 20
