``subview``
===========

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    auto s = subview(view,std::pair<int,int>(5,191),Kokkos::ALL,1);

Creates a ``Kokkos::View`` representing a subset of another ``Kokkos::View``.


.. _KokkosAll: ../utilities/all.rst#kokkosall

.. |KokkosAll| replace:: :cpp:func:`Kokkos::ALL`

Description
-----------

.. cpp:function:: template<class ViewType, class ... Args> IMPL_DETAIL subview(const ViewType& v, Args ... args)

   Returns a new ``Kokkos::View`` ``s`` representing a subset of ``v`` specified by ``args...``. The return type of subview is an implementation detail and is determined by the types in ``Args...``.

   .. rubric:: Subset selection:

   * For every integer argument in ``args...`` the rank of the returned view is one smaller
     than the rank of ``v`` and the values referenced by ``s`` correspond to the values associated
     with using the integer argument in the corresponding position during indexing into ``v``.

   * Passing |KokkosAll|_ as the ``r``\ th argument is equivalent to passing ``pair<ptrdiff_t,ptrdiff_t>(0,v.extent(r))`` as the ``r``\ th argument.

   * If the ``r``\ th argument ``arg_r`` is the ``d``\ th range (\ ``std::pair``\ , ``Kokkos::pair`` or |KokkosAll|_ )
     in the argument list than ``s.extent(d) = arg_r.second-arg_r.first``\ , and dimension ``d`` of ``s``
     references the range ``[arg_r.first,arg_r.second)`` of dimension ``r`` of ``v``.

   .. rubric:: Restrictions:

   * ``sizeof...(args)`` is equal to ``ViewType::rank``.

   * Valid arguments are of type:

     - ``std::pair<iType,iType>`` with ``std::is_integral<iType>::value`` being true.

     - ``Kokkos::pair<iType,iType>`` with ``std::is_integral<iType>::value`` being true.

     - ``iType`` with ``std::is_integral<iType>::value`` being true.

     - ``std::remove_const_t< decltype(``\ |KokkosAll|_ ``)>``

   * If the ``r``\ th argument ``arg_r`` is of type ``std::pair<iType,iType>`` or ``Kokkos::pair<iType,iType>`` it must meet:

     - ``arg_r.first >= 0``

     - ``arg_r.second <= v.extent(r)``

     - ``arg_r.first <= arg_r.second``

   * If the ``r``\ th argument ``arg_r`` is an integral it must meet:

     - ``arg_r >= 0``

     - ``arg_r < v.extent(r)``

Examples
--------

.. code-block:: cpp

    Kokkos::View<double***[5]> a("A",N0,N1,N2);

    auto s  = Kokkos::subview(a,
                              std::pair<int,int>(3,15),
			      5,
			      Kokkos::ALL,
			      Kokkos::ALL);
    for(int i0 = 0; i0 < s.extent(0); i0++)
    for(int i1 = 0; i1 < s.extent(1); i1++)
    for(int i2 = 0; i2 < s.extent(2); i2++) {
        assert(s(i0,i1,i2) == a(i0+3,5,i1,i2));
    }

    auto s3415 = Kokkos::subview(a,3,4,1,5);
    assert(s3415() == a(3,4,1,5));
