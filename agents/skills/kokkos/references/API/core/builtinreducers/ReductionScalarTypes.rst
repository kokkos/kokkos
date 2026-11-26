Reduction Scalar Types
######################

Types designed to hold the result of ``parallel_reduce()``, while using the corresponding builtin reducers.

.. list-table::
   :widths: 20 65 15
   :header-rows: 1

   * - Class template
     - Description
     - Builtin Reducer
   * - :doc:`FirstLocScalar`
     - stores the first location that satisfies a condition
     - :cpp:class:`FirstLoc`
   * - :doc:`LastLocScalar`
     - stores the last location that satisfies a condition
     - :cpp:class:`LastLoc`
   * - :doc:`MinMaxLocScalar`
     - stores a minimum, a maximum, and their respective locations
     - :cpp:class:`MinMaxLoc`
   * - :doc:`MinMaxScalar`
     - stores a minimum value and a maximum value
     - :cpp:class:`MinMax`
   * - :doc:`ValLocScalar`
     - stores a single value and its location
     - :cpp:class:`MinLoc`, :cpp:class:`MaxLoc`

.. toctree::
   :hidden:
   :maxdepth: 1

   FirstLocScalar
   LastLocScalar
   MinMaxLocScalar
   MinMaxScalar
   ValLocScalar
