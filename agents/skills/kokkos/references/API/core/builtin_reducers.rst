Built-in Reducers
=================

`ReducerConcept <builtinreducers/ReducerConcept.rst>`__ provides the concept for Reducers.

Reducer objects used in conjunction with `parallel_reduce <parallel-dispatch/parallel_reduce.rst>`__

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Reducer
     - Description
    * - `BAnd <builtinreducers/BAnd.rst>`__
      - Binary 'And' reduction
    * - `BOr <builtinreducers/BOr.rst>`__
      - Binary 'Or' reduction
    * - `FirstLoc <builtinreducers/FirstLoc.rst>`__
      - Providing the first index satisfying a condition
    * - `LAnd <builtinreducers/LAnd.rst>`__
      - Logical 'And' reduction
    * - `LastLoc <builtinreducers/LastLoc.rst>`__
      - Providing the last index satisfying a condition
    * - `LOr <builtinreducers/LOr.rst>`__
      - Logical 'Or' reduction
    * - `Max <builtinreducers/Max.rst>`__
      - Maximum reduction
    * - `MaxFirstLoc <builtinreducers/MaxFirstLoc.rst>`__
      - Reduction providing maximum and an associated first index
    * - `MaxLoc <builtinreducers/MaxLoc.rst>`__
      - Reduction providing maximum and an associated index
    * - `Min <builtinreducers/Min.rst>`__
      - Minimum reduction
    * - `MinFirstLoc <builtinreducers/MinFirstLoc.rst>`__
      - Reduction providing minimum and an associated first index
    * - `MinLoc <builtinreducers/MinLoc.rst>`__
      - Reduction providing minimum and an associated index
    * - `MinMax <builtinreducers/MinMax.rst>`__
      - Reduction providing both minimum and maximum
    * - `MinMaxFirstLastLoc <builtinreducers/MinMaxFirstLastLoc.rst>`__
      - Reduction providing both minimum and maximum and associated first and last indices
    * - `MinMaxLoc <builtinreducers/MinMaxLoc.rst>`__
      - Reduction providing both minimum and maximum and associated indices
    * - `Prod <builtinreducers/Prod.rst>`__
      - Multiplicative reduction
    * - `Sum <builtinreducers/Sum.rst>`__
      - Sum reduction


:cpp:struct:`reduction_identity` defines the neutral elements (identity values)
for various reduction operations. Specializing it is crucial for enabling
built-in reducers to work with user-defined types.

`Reduction Scalar Types <builtinreducers/ReductionScalarTypes.rst>`__ are template classes for storage for reducers.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./builtinreducers/ReducerConcept
   ./builtinreducers/BAnd
   ./builtinreducers/BOr
   ./builtinreducers/FirstLoc
   ./builtinreducers/LAnd
   ./builtinreducers/LastLoc
   ./builtinreducers/LOr
   ./builtinreducers/Max
   ./builtinreducers/MaxFirstLoc
   ./builtinreducers/MaxLoc
   ./builtinreducers/Min
   ./builtinreducers/MinFirstLoc
   ./builtinreducers/MinLoc
   ./builtinreducers/MinMax
   ./builtinreducers/MinMaxFirstLastLoc
   ./builtinreducers/MinMaxLoc
   ./builtinreducers/Prod
   ./builtinreducers/Sum
   ./builtinreducers/ReductionScalarTypes
   ./builtinreducers/reduction_identity
