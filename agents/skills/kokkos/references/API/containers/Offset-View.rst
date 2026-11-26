
``OffsetView``
==============

Header file: ``<Kokkos_OffsetView.hpp>``

An ``OffsetView`` can be used when the indices of an array begin at something other than zero.

.. warning::

   The OffsetView is in the Experimental namespace.


Construction
------------

An OffsetView must have a label, and at least one dimension. Only runtime extents are supported, but otherwise the semantics of an OffsetView are similar to those of a View.

.. code-block:: cpp

   const size_t min0 = ...;
   const size_t max0 = ...;
   const size_t min1 = ...;
   const size_t max1 = ...;
   const size_t min2 = ...;
   const size_t max2 = ...;

   OffsetView<int***> a("someLabel", {min0, max0}, {min1, max1},{min2, max2});

Construction from a layout is also allowed.

.. code-block:: cpp

    OffsetView<int***> a("someLabel", LayoutLeft, {min0, min1, min2});

An OffsetView may also be created from a View that has the same underlying type. Since the View already has extents, the beginning indices must be passed to the constructor.

.. code-block:: cpp

   View<double**> b("somelabel", 10, 20);
   Array<int64_t, 2> begins = {-10, -20};
   OffsetView<double**> ov(b, begins);

The OffsetView ov has the same extents as b and must be indexed from [-10,-1] and [-20,-11].

A std::initializer_list may also be used instead of an Array.

.. code-block:: cpp

    OffsetView<double**> ov(b, {-10, -20});

Interface
---------

The beginning indices may be obtained as an array. The begin and end of iteration may be found for each rank.

.. code-block:: cpp

   OffsetView<int***> ov("someLabel", {-1,1}, {-2,2}, {-3,3});
    Array<int64_t, 3> a = ov.begins();

   const int64_t begin0 = ov.begin(0);
   const int64_t end0= ov.end(0);

Note that

.. code-block:: cpp

   OffsetView::end(const size_t i)

returns a value that is not a legal index:  It is exactly one more than the maximum allowable index for the given dimension i.

Subviews are supported, and the result of taking a subview of an OffsetView is another OffsetView. If ALL() is passed to the subview function, then the offsets for that rank are preserved, otherwise they are dropped.

.. code-block:: cpp

   OffsetView<Scalar***> sliceMe("offsetToSlice", {-10,20}, {-20,30}, {-30,40});
   auto offsetSubview = subview(sliceMe,0, Kokkos::ALL(), std::make_pair(-30, -21));

   ASSERT_EQ(offsetSubview.Rank, 2);
   ASSERT_EQ(offsetSubview.begin(0) , -20);
   ASSERT_EQ(offsetSubview.end(0) , 31);
   ASSERT_EQ(offsetSubview.begin(1) , 0);
   ASSERT_EQ(offsetSubview.end(1) , 9);

The following deep copies are also supported: from a constant value to an OffsetView; from a compatible OffsetView to another OffsetView; from a compatible View to an OffsetView; from a compatible OffsetView to a View.

A compatible View with the same label is obtained from the view() method.

.. code-block:: cpp

   OffsetView<int***> ov("someLabel", {-1,1}, {-2,2}, {-3,3});
   View<int***> v = ov.view();

A copy constructor and an assignment operator from a View to an OffsetView are also provided.

Equivalence operators "==" and "!=" are defined. Given an OffsetView and a View, they are equivalent in the same sense that two Views are equivalent. Similarly, two OffsetViews are equivalent in the same sense if their begins also match.

Mirrors are also supported.
