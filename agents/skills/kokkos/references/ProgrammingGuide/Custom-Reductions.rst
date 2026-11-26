Custom Reductions
=================

As described previously Kokkos reductions perform a "sum" reduction by default. But while that is the most common type of reduction it is not the only one required by more complex applications. Kokkos provides the "Reducer" concept to accommodate custom reductions. 

A Reducer is a class which provides the necessary information to join (reduce) two values, knows how to initialize thread private variables and where to store the final result of a reduction. Depending on your situation, you may need to write more or less code specialization for a custom reduction. 

.. toctree::
    :maxdepth: 1

    Custom-Reductions-Built-In-Reducers

To perform a common operation, such as finding the minimum for an intrinsic C++ type, no custom code is required. Kokkos::complex is also supported with Built-in reducers without any custom additions. Click the heading for more detail.

.. toctree::
    :maxdepth: 1

    Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types

If your application requires a custom scalar types, the scalar type must be copy constructible and provide the necessary operators for the reduction (for example comparison operators are required for minmax-reductions). A specialization of the reduction_identity class is also required.  Click the heading for more detail. 

.. toctree::
    :maxdepth: 1

    Custom-Reductions-Custom-Reducers

For completely arbitrary reductions, you must provide an implementation of a Reducer class. Click the heading for more detail and an example.
