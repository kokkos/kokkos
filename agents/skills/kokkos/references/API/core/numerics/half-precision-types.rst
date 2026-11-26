.. _api-Half-precision-types:

Half precision types
====================

.. warning::
   ``half_t`` and ``bhalf_t`` are still in the ``Kokkos::Experimental`` namespace.

Types
-----
Kokkos offers portable half precision types under the name ``Kokkos::Experimental::half_t`` and ``Kokkos::Experimental::bhalf_t``.
 - ``half_t`` represents the standard half precision float, with a 1 bit Sign Bit, 5 bits Exponent and 10 bits Significand.
 - ``bhalf_t`` corresponds to the type known as 'brain half' and has a 1 bit Sign Bit, 8 bits Exponent and 7 bits Significand.

This types will either map to the current backend own types (for instance ``__half`` on Cuda), or to ``float`` if no such type is available.

The macros ``KOKKOS_HALF_T_IS_FLOAT`` and ``KOKKOS_BHALF_T_IS_FLOAT`` are set to ``true`` when ``half_t`` and ``bhalf_t`` are mapped to ``float``, and ``false`` otherwise.

Functions
---------
The following table list the standard mathematical functions that can be used with the ``half_t`` and ``bhalf_t`` type.

In addition, for the Cuda, SYCL, and HIP backends, the marked functions are performed using specific half precision functions and may thus be more performant.
The default behaviour if the specific function doesn't exist is to cast the half precision float to ``float``, perform the operation with the standard function and cast back the result to half precision.

.. csv-table::
   :header: "Function", "``half_t`` Cuda", "``bhalf_t`` Cuda", "``half_t`` SYCL", "``bhalf_t`` SYCL", "``half_t`` HIP", "``bhalf_t`` HIP"
   :widths: auto

   "abs", "X", "X", "X", , "X", "X"
   "fabs","X", "X", "X", "X", "X", "X"
   "fmod", , , "X", 
   "remainder", , , "X", 
   "fmax","X¹", "X¹", "X", "X", "X", "X"
   "fmin","X¹", "X¹", "X", "X", "X", "X"
   "fdim", "X", "X", "X", , "X", "X"
   "exp", "X", "X", "X", "X", "X", "X"
   "exp2", "X", "X", "X", "X", "X", "X"
   "expm1", , , "X", 
   "log", "X", "X", "X", "X", "X", "X"
   "log10", "X", "X", "X", "X", "X", "X"
   "log2", "X", "X", "X", "X", "X", "X"
   "log1p", , , "X", 
   "pow", , , "X", 
   "sqrt", "X", "X", "X", "X", "X", "X"
   "cbrt", , , "X", 
   "hypot", , , "X", 
   "sin", "X", "X", "X", "X", "X", "X"
   "cos", "X", "X", "X", "X", "X", "X"
   "tan", , , "X", 
   "asin", , , "X", 
   "acos", , , "X", 
   "atan", , , "X", 
   "atan2", , , "X", 
   "sinh", , , "X", 
   "cosh", , , "X", 
   "tanh", , , "X", 
   "asinh", , , "X", 
   "acosh", , , "X", 
   "atanh", , , "X", 
   "erf", , , "X", 
   "erfc", , , "X", 
   "tgamma", , , "X", 
   "lgamma", , , "X", 
   "ceil", "X", "X", "X", "X", "X", "X"
   "floor", "X", "X", "X", "X", "X", "X"
   "trunc", "X", "X", "X", "X", "X", "X"
   "round", , , "X", 
   "nearbyint", "X", "X", , 
   "logb", , , "X", 
   "nextafter", "X²", "X²", "X", "X", "X", "X"
   "copysign", , , "X", 
   "isfinite", , , "X", 
   "isinf", "X³", "X³", "X", , "X", "X"
   "isnan", "X", "X", "X", "X", "X", "X"
   "signbit", , , "X", 

¹Only if GPU_ARCH >= 80

²Not supported with MSVC

³Not for nvcc-12.2 when compiling with --std=c++20 (https://docs.nvidia.com/cuda/archive/12.3.2/cuda-toolkit-release-notes/index.html#cuda-math-release-12-3)

Example
~~~~~~~
.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    #include<iostream>

    int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);
        const int N = 10;

        using half_type = Kokkos::Experimental::bhalf_t;

        Kokkos::View<half_type*> view("half view", N);

        Kokkos::parallel_for("parallel region",
          N,
          KOKKOS_LAMBDA(const int i) {
            // exponential function performed over `bhalf` type if available, over `float` otherwise 
            view (i) = Kokkos::exp(half_type(i));
          });
    }

Numeric Traits
--------------

The following standard numeric traits can be used with ``half_t`` and ``bhalf_t``:
 - infinity
 - finite_min
 - finite_max
 - epsilon
 - round_error
 - norm_min
 - quiet_NaN
 - signaling_NaN
 - digits
 - digits10
 - radix
 - min_exponent
 - max_exponent

Example
~~~~~~~
.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    #include<iostream>

    int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);

        // Print 24 or 11 depending on the value of KOKKOS_HALF_T_IS_FLOAT
        std::cout << Kokkos::Experimental::digits_v<Kokkos::Experimental::half_t> << std::endl;
    }
