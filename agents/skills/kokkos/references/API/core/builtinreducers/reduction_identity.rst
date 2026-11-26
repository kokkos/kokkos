``reduction_identity``
======================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. cpp:namespace:: Kokkos

.. cpp:struct:: template <typename ScalarType> reduction_identity
   :no-index-entry:

   The ``reduction_identity`` class template provides static member functions
   that return the identity value for various reduction types.

   .. rubric:: Static members available for integral and floating-point types:

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType sum()

      :returns: Neutral element for built-in reducer :cpp:class:`Sum`

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType prod()

      :returns: Neutral element for built-in reducer :cpp:class:`Prod`

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType min()

      :returns: Neutral element for built-in reducer :cpp:class:`Min`

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType max()

      :returns: Neutral element for built-in reducer :cpp:class:`Max`
   
   .. rubric:: Static members available for integral types:

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType land()

      :returns: Neutral element for built-in reducer :cpp:class:`LAnd` (Logical AND)

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType lor()

      :returns: Neutral element for built-in reducer :cpp:class:`LOr` (Logical OR)

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType band()

      :returns: Neutral element for built-in reducer :cpp:class:`BAnd` (Bitwise AND)

   .. cpp:function:: KOKKOS_FUNCTION static ScalarType bor()

      :returns: Neutral element for built-in reducer :cpp:class:`BOr` (Bitwise OR)

Description
-----------

The ``reduction_identity`` struct provides the identity element (also known as
the neutral element) for various common reduction operations. In the context of
a parallel reduction, the identity element is the starting value for the
reduction variable on each thread. When combined with any other value using the
reduction operation, it does not change the other value. For example, for a sum
reduction, the identity is :math:`0`, because :math:`x+0=x`. For a product
reduction, the identity is :math:`1`, because :math:`x \times 1 = x`.

Kokkos' built-in reducers (e.g., :doc:`Sum`, :doc:`Prod`, :doc:`Min`,
:doc:`Max`) implicitly use specializations of ``reduction_identity`` to
initialize the thread-local reduction accumulators.

Kokkos provides specializations for all arithmetic types (i.e. integral and
floating-point types) as well as for :cpp:class:`complex\<T\> <complex>`.

.. note::

   ``Kokkos::reduction_identity`` is not intended for direct use in your
   application code. Instead, it serves as a customization point within the
   Kokkos framework. You should only specialize reduction_identity when you
   need to enable Kokkos's built-in reducers (like ``Kokkos::Sum``,
   ``Kokkos::Min``, ``Kokkos::Max``, etc.) to work seamlessly with your own
   user-defined data types.  This allows Kokkos to correctly determine the
   initial "identity" value for your custom type during parallel reduction
   operations.

Custom Scalar Types
-------------------

For custom (user-defined) ``ScalarType``\s to be used with Kokkos' built-in
reducers, a template specialization of
``Kokkos::reduction_identity<CustomType>`` must be defined.  This
specialization must provide static member functions corresponding to the
desired reduction operations. These functions should return an instance of
``CustomType`` initialized with the appropriate identity value.

Example: Specializing ``reduction_identity`` for a Custom Array Type
--------------------------------------------------------------------

Consider a custom struct ``array_type`` that holds an an array of integers, for
which we want to perform a sum reduction.

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
 
    namespace sample {
    template <class ScalarType, int N>
    struct array_type {
      ScalarType the_array[N] = {};
 
      KOKKOS_FUNCTION
      array_type& operator+=(const array_type& src) {
        for (int i = 0; i < N; ++i) {
          the_array[i] += src.the_array[i];
        }
        return *this;
      }
    };
 
    using ValueType = array_type<int, 4>;
    } // namespace sample
 
    // Specialization of Kokkos::reduction_identity for sample::ValueType
    template <>
    struct Kokkos::reduction_identity<sample::ValueType> {
      KOKKOS_FUNCTION static sample::ValueType sum() {
        return sample::ValueType(); // Default constructor initializes to zeros
      }
      // If other reduction types were needed (e.g., min, max, prod),
      // their respective identity functions would also be defined here.
    };
 
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        const int E = 100;
        sample::ValueType tr; // Result will be stored here
 
        Kokkos::parallel_reduce(
            "SumArray", E,
            KOKKOS_LAMBDA(const int i, sample::ValueType& lval) {
              lval.the_array[0] += 1;
              lval.the_array[1] += i;
              lval.the_array[2] += i * i;
              lval.the_array[3] += i * i * i;
            },
            Kokkos::Sum<sample::ValueType>(tr));
 
        printf("Computed result for %d is {%d, %d, %d, %d}\n", E,
               tr.the_array[0], tr.the_array[1], tr.the_array[2],
               tr.the_array[3]);
 
        // Expected values:
        // [0]: E = 100
        // [1]: sum(0..99) = 99*100/2 = 4950
        // [2]: sum(0..99) of i*i = 99*(99+1)*(2*99+1)/6 = 328350
        // [3]: sum(0..99) of i*i*i = (99*100/2)^2 = 4950^2 = 24502500
 
        printf("Expected result for %d is {%d, %d, %d, %d}\n", E,
               100, 4950, 328350, 24502500);
      }
      Kokkos::finalize();
      return 0;
    }

