``MinLoc``
==========

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.rst>`_ storing the minimum value with an index

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinLoc<T,I,S>::value_type result;
   parallel_reduce(N,Functor,MinLoc<T,I,S>(result));

Synopsis
--------

.. code-block:: cpp

   template<class Scalar, class Index, class Space>
   class MinLoc{
     public:
       typedef MinLoc reducer;
       typedef ValLocScalar<typename std::remove_cv<Scalar>::type,
                            typename std::remove_cv<Index>::type > value_type;
       typedef Kokkos::View<value_type, Space> result_view_type;

       KOKKOS_INLINE_FUNCTION
       void join(value_type& dest, const value_type& src) const;

       KOKKOS_INLINE_FUNCTION
       void init(value_type& val) const;

       KOKKOS_INLINE_FUNCTION
       value_type& reference() const;

       KOKKOS_INLINE_FUNCTION
       result_view_type view() const;

       KOKKOS_INLINE_FUNCTION
       MinLoc(value_type& value_);

       KOKKOS_INLINE_FUNCTION
       MinLoc(const result_view_type& value_);
   };

Interface
---------

.. cpp:class:: template<class Scalar, class Index, class Space> MinLoc

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `ValLocScalar <ValLocScalar.rst>`_)

   .. cpp:type:: result_view_type

      A ``Kokkos::View`` referencing the reduction result

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION MinLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION MinLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store minimum with index of ``src`` and ``dest`` into ``dest``:  ``dest = (src.val < dest.val) ? src :dest;``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

      Initialize ``val.val`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.
      Initialize ``val.loc`` using the ``Kokkos::reduction_identity<Index>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION value_type& reference() const;

      Returns a reference to the result provided in class constructor.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION result_view_type view() const;

      Returns a view of the result place provided in class constructor.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``MinLoc<T,I,S>::value_type`` is Specialization of ValLocScalar on non-const ``T`` and non-const ``I``

* ``MinLoc<T,I,S>::result_view_type`` is ``Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator =`` and ``operator <`` defined. ``Kokkos::reduction_identity<Scalar>::min()`` is a valid expression.

* Requires: ``Index`` has ``operator =`` defined. ``Kokkos::reduction_identity<Index>::min()`` is a valid expression.

* In order to use MinLoc with a custom type of either ``Scalar`` or ``Index``, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.rst>`_ for details

Example
-------

.. code-block:: cpp

  #include <Kokkos_Core.hpp>
  struct Idx3D_t {
    int value[3];
    int& operator[](int i) { return value[i]; }
    const int& operator[](int i) const { return value[i]; }
  };
  template <>
  struct Kokkos::reduction_identity<Idx3D_t> {
    static constexpr Idx3D_t min() { return {0, 0, 0}; }
  };
  int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
      Kokkos::View<double***> a("A", 5, 5, 5);
      Kokkos::deep_copy(a, 10);
      a(2, 3, 1)        = 5;
      using MinLoc_t    = Kokkos::MinLoc<double, Idx3D_t>;
      using MinLocVal_t = typename MinLoc_t::value_type;
      MinLocVal_t result;
      Kokkos::parallel_reduce(
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {5, 5, 5}),
          KOKKOS_LAMBDA(int i, int j, int k, MinLocVal_t& val) {
            if (a(i, j, k) < val.val) {
              val.val    = a(i, j, k);
              val.loc[0] = i;
              val.loc[1] = j;
              val.loc[2] = k;
            }
          },
          MinLoc_t(result));
      printf("%lf %i %i %i\n", result.val, result.loc[0], result.loc[1],
             result.loc[2]);
    }
    Kokkos::finalize();
  }
