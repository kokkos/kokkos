
``StaticCrsGraph`` [DEPRECATED]
===============================

Header file: ``<Kokkos_StaticCrsGraph.hpp>`` (deprecated since Kokkos 4.5)

``StaticCrsGraph`` was moved to KokkosKernels.

The StaticCrsGraph is a Compressed row storage array with the row map, the column indices and the non-zero entries stored in 3 different Kokkos::Views.  Appropriate types and functions are provided to simplify manipulation and access to CRS data on either a host or device.

Usage
-----

.. code-block:: cpp

    using StaticCrsGraphType = Kokkos::StaticCrsGraph<unsigned, Space, void, void, unsigned>;
    StaticCrsGraphType graph();

    const int begin = graph.row_map[0];
    const int end = graph.row_map[1];

    double sum = 0;
    for (int i = begin; i < end; i++) {
        Kokkos::View<unsigned, Space> v(graph.entries(i));
        for (int j = 0; j < v.extent(0); j++) {
            sum += v(j);
        }
    }
