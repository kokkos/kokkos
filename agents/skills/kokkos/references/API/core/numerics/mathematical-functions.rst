Common math functions
=====================

.. role:: cpp(code)
    :language: cpp

.. role:: strike
    :class: strike

Motivating example (borrowed from https://llvm.org/docs/CompileCudaWithLLVM.html#standard-library-support)

.. code-block:: cpp

    // clang is OK with everything in this function.
    __device__ void test() {
        std::sin(0.); // nvcc - ok
        std::sin(0);  // nvcc - error, because no std::sin(int) override is available.
        sin(0);       // nvcc - same as above.

        sinf(0.);       // nvcc - ok
        std::sinf(0.);  // nvcc - no such function
    }

Kokkos' goal is to provide a consistent overload set that is available on host
and device and that follows practice from the C++ numerics library.

------------

.. _text: https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalFunctions.hpp

.. |text| replace:: ``<Kokkos_MathematicalFunctions.hpp>``

Defined in header |text|_ which is included from ``<Kokkos_Core.hpp>``

.. _text2: https://en.cppreference.com/w/cpp/numeric/math

.. |text2| replace:: standard C mathematical functions from ``<cmath>``

Provides most of the |text2|_, such as ``fabs``, ``sqrt``, and ``sin``.

Below is the synopsis for ``sqrt`` as an example of unary math function.

.. code-block:: cpp

    namespace Kokkos {
        KOKKOS_FUNCTION float       sqrt ( float x );
        KOKKOS_FUNCTION float       sqrtf( float x );
        KOKKOS_FUNCTION double      sqrt ( double x );
                        long double sqrt ( long double x );
                        long double sqrtl( long double x );
        KOKKOS_FUNCTION double      sqrt ( IntegralType x );
    }

The function is overloaded for any argument of arithmetic type. Additional functions with ``f`` and ``l`` suffixes that work on ``float`` and ``long double`` respectively are also available.  Please note, that ``long double`` overloads are not available on the device.

See below the list of common mathematical functions supported. We refer the reader to cppreference.com for the synopsis of each individual function.

------------

.. [#since_kokkos_4_0] (since Kokkos 4.0)
.. [#since_kokkos_4_1] (since Kokkos 4.1)
.. [#since_kokkos_5_1] (since Kokkos 5.1)
.. [#not_available_with_sycl] (not available with SYCL)

Basic operations
^^^^^^^^^^^^^^^^

.. _abs: https://en.cppreference.com/w/cpp/numeric/math/fabs

.. |abs| replace:: ``abs``

.. _fabs: https://en.cppreference.com/w/cpp/numeric/math/fabs

.. |fabs| replace:: ``fabs``

.. _fmod: https://en.cppreference.com/w/cpp/numeric/math/fmod

.. |fmod| replace:: ``fmod``

.. _remainder: https://en.cppreference.com/w/cpp/numeric/math/remainder

.. |remainder| replace:: ``remainder``

.. _remquo: https://en.cppreference.com/w/cpp/numeric/math/remquo

.. |remquo| replace:: ``remquo``

.. _fma: https://en.cppreference.com/w/cpp/numeric/math/fma

.. |fma| replace:: ``fma``

.. _fmax: https://en.cppreference.com/w/cpp/numeric/math/fmax

.. |fmax| replace:: ``fmax``

.. _fmin: https://en.cppreference.com/w/cpp/numeric/math/fmin

.. |fmin| replace:: ``fmin``

.. _fdim: https://en.cppreference.com/w/cpp/numeric/math/fdim

.. |fdim| replace:: ``fdim``

.. _nan: https://en.cppreference.com/w/cpp/numeric/math/nan

.. |nan| replace:: ``nan``

.. list-table::
   :align: left

   * - |abs|_
       |fabs|_
     - absolute value of a floating point value (:math:`|x|`)
   * - |fmod|_
     - remainder of the floating point division operation
   * - |remainder|_
     - signed remainder of the division operation
   * - |remquo|_ [#since_kokkos_5_1]_
     - signed remainder as well as the three last bits of the division operation
   * - |fma|_ [#since_kokkos_4_0]_
     - fused multiply-add operation
   * - |fmax|_
     - larger of two floating-point values
   * - |fmin|_
     - smaller of two floating point values
   * - |fdim|_
     - positive difference of two floating point values (:math:`\max(0, x-y)`)
   * - |nan|_
     - not-a-number (NaN)

Exponential functions
^^^^^^^^^^^^^^^^^^^^^

.. _exp: https://en.cppreference.com/w/cpp/numeric/math/exp

.. |exp| replace:: ``exp``

.. _exp2: https://en.cppreference.com/w/cpp/numeric/math/exp2

.. |exp2| replace:: ``exp2``

.. _expm1: https://en.cppreference.com/w/cpp/numeric/math/expm1

.. |expm1| replace:: ``expm1``

.. _log: https://en.cppreference.com/w/cpp/numeric/math/log

.. |log| replace:: ``log``

.. _log10: https://en.cppreference.com/w/cpp/numeric/math/log10

.. |log10| replace:: ``log10``

.. _log2: https://en.cppreference.com/w/cpp/numeric/math/log2

.. |log2| replace:: ``log2``

.. _log1p: https://en.cppreference.com/w/cpp/numeric/math/log1p

.. |log1p| replace:: ``log1p``

.. list-table::
   :align: left

   * - |exp|_
     - returns :math:`e` raised to the given power (:math:`e^x`)
   * - |exp2|_
     - returns :math:`2` raised to the given power (:math:`2^x`)
   * - |expm1|_
     - returns :math:`e` raised to the given power, minus :math:`1` (:math:`e^x-1`)
   * - |log|_
     - base :math:`e` logarithm of the given number (:math:`\log(x)`)
   * - |log10|_
     - base :math:`10` logarithm of the given number (:math:`\log_{10}(x)`)
   * - |log2|_
     - base :math:`2` logarithm of the given number (:math:`\log_{2}(x)`)
   * - |log1p|_
     - natural logarithm (to base :math:`e`) of 1 plus the given number (:math:`\ln(1+x)`)

Power functions
^^^^^^^^^^^^^^^

.. _pow: https://en.cppreference.com/w/cpp/numeric/math/pow

.. |pow| replace:: ``pow``

.. _sqrt: https://en.cppreference.com/w/cpp/numeric/math/sqrt

.. |sqrt| replace:: ``sqrt``

.. _cbrt: https://en.cppreference.com/w/cpp/numeric/math/cbrt

.. |cbrt| replace:: ``cbrt``

.. _hypot: https://en.cppreference.com/w/cpp/numeric/math/hypot

.. |hypot| replace:: ``hypot``

.. list-table::
   :align: left

   * - |pow|_
     - raises a number to the given power (:math:`x^y`)
   * - |sqrt|_
     - computes square root (:math:`\sqrt{x}`)
   * - |cbrt|_
     - computes cube root (:math:`\sqrt[3]{x}`)
   * - |hypot|_ [#3_argument_overload_since_kokkos_4_0]_
     - computes hypotenuse (:math:`\sqrt{x^2 + y^2}` and :math:`\sqrt{x^2 + y^2 + z^2}`)

.. [#3_argument_overload_since_kokkos_4_0] 3-argument overload available since Kokkos 4.0

Trigonometric functions
^^^^^^^^^^^^^^^^^^^^^^^

.. _sin: https://en.cppreference.com/w/cpp/numeric/math/sin

.. |sin| replace:: ``sin``

.. _cos: https://en.cppreference.com/w/cpp/numeric/math/cos

.. |cos| replace:: ``cos``

.. _tan: https://en.cppreference.com/w/cpp/numeric/math/tan

.. |tan| replace:: ``tan``

.. _asin: https://en.cppreference.com/w/cpp/numeric/math/asin

.. |asin| replace:: ``asin``

.. _acos: https://en.cppreference.com/w/cpp/numeric/math/acos

.. |acos| replace:: ``acos``

.. _atan: https://en.cppreference.com/w/cpp/numeric/math/atan

.. |atan| replace:: ``atan``

.. _atan2: https://en.cppreference.com/w/cpp/numeric/math/atan2

.. |atan2| replace:: ``atan2``

.. list-table::
   :align: left

   * - |sin|_
     - computes sine (:math:`\sin(x)`)
   * - |cos|_
     - computes cosine (:math:`\cos(x)`)
   * - |tan|_
     - computes tangent (:math:`\tan(x)`)
   * - |asin|_
     - computes arc sine (:math:`\arcsin(x)`)
   * - |acos|_
     - computes arc cosine (:math:`\arccos(x)`)
   * - |atan|_
     - computes arc tangent (:math:`\arctan(x)`)
   * - |atan2|_
     - arc tangent, using signs to determine quadrants

Hyperbolic functions
^^^^^^^^^^^^^^^^^^^^

.. _sinh: https://en.cppreference.com/w/cpp/numeric/math/sinh

.. |sinh| replace:: ``sinh``

.. _cosh: https://en.cppreference.com/w/cpp/numeric/math/cosh

.. |cosh| replace:: ``cosh``

.. _tanh: https://en.cppreference.com/w/cpp/numeric/math/tanh

.. |tanh| replace:: ``tanh``

.. _asinh: https://en.cppreference.com/w/cpp/numeric/math/asinh

.. |asinh| replace:: ``asinh``

.. _acosh: https://en.cppreference.com/w/cpp/numeric/math/acosh

.. |acosh| replace:: ``acosh``

.. _atanh: https://en.cppreference.com/w/cpp/numeric/math/atanh

.. |atanh| replace:: ``atanh``

.. list-table::
   :align: left

   * - |sinh|_
     - computes hyperbolic sine (:math:`\sinh(x)`)
   * - |cosh|_
     - computes hyperbolic cosine (:math:`\cosh(x)`)
   * - |tanh|_
     - computes hyperbolic tangent (:math:`\tanh(x)`)
   * - |asinh|_
     - computes the inverse hyperbolic sine (:math:`\text{arsinh}(x)`)
   * - |acosh|_
     - computes the inverse hyperbolic cosine (:math:`\text{arcosh}(x)`)
   * - |atanh|_
     - computes the inverse hyperbolic tangent (:math:`\text{artanh}(x)`)

Error and gamma functions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _erf: https://en.cppreference.com/w/cpp/numeric/math/erf

.. |erf| replace:: ``erf``

.. _erfc: https://en.cppreference.com/w/cpp/numeric/math/erfc

.. |erfc| replace:: ``erfc``

.. _tgamma: https://en.cppreference.com/w/cpp/numeric/math/tgamma

.. |tgamma| replace:: ``tgamma``

.. _lgamma: https://en.cppreference.com/w/cpp/numeric/math/lgamma

.. |lgamma| replace:: ``lgamma``

.. list-table::
   :align: left

   * - |erf|_
     - error function
   * - |erfc|_
     - complementary error function
   * - |tgamma|_
     - gamma function
   * - |lgamma|_
     - natural logarithm of the gamma function

Nearest integer floating point operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _ceil: https://en.cppreference.com/w/cpp/numeric/math/ceil

.. |ceil| replace:: ``ceil``

.. _floor: https://en.cppreference.com/w/cpp/numeric/math/floor

.. |floor| replace:: ``floor``

.. _trunc: https://en.cppreference.com/w/cpp/numeric/math/trunc

.. |trunc| replace:: ``trunc``

.. _round: https://en.cppreference.com/w/cpp/numeric/math/round

.. |round| replace:: ``round``

.. _lround: https://en.cppreference.com/w/cpp/numeric/math/round

.. |lround| replace:: ``lround``

.. _llround: https://en.cppreference.com/w/cpp/numeric/math/round

.. |llround| replace:: ``llround``

.. _nearbyint: https://en.cppreference.com/w/cpp/numeric/math/nearbyint

.. |nearbyint| replace:: ``nearbyint``

.. _rint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |rint| replace:: ``rint``

.. _lrint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |lrint| replace:: ``lrint``

.. _llrint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |llrint| replace:: ``llrint``

.. list-table::
   :align: left

   * - |ceil|_
     - nearest integer not less than the given value
   * - |floor|_
     - nearest integer not greater than the given value
   * - |trunc|_
     - nearest integer not greater in magnitude than the given value
   * - |round|_
       |lround|_ [#since_kokkos_5_1]_ [#not_available_with_sycl]_
       |llround|_ [#since_kokkos_5_1]_ [#not_available_with_sycl]_
     - nearest integer, rounding away from zero in halfway cases
   * - |nearbyint|_ [#not_available_with_sycl]_
     - nearest integer using current rounding mode
   * - |rint|_ [#since_kokkos_5_1]_
       |lrint|_ [#since_kokkos_5_1]_ [#not_available_with_sycl]_
       |llrint|_ [#since_kokkos_5_1]_ [#not_available_with_sycl]_
     - nearest integer using current rounding mode with exception if the result differs

Floating point manipulation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _frexp: https://en.cppreference.com/w/cpp/numeric/math/frexp

.. |frexp| replace:: ``frexp``

.. _ldexp: https://en.cppreference.com/w/cpp/numeric/math/ldexp

.. |ldexp| replace:: ``ldexp``

.. _modf: https://en.cppreference.com/w/cpp/numeric/math/modf

.. |modf| replace:: ``modf``

.. _scalbn: https://en.cppreference.com/w/cpp/numeric/math/scalbn

.. |scalbn| replace:: ``scalbn``

.. _scalbln: https://en.cppreference.com/w/cpp/numeric/math/scalbln

.. |scalbln| replace:: ``scalbln``

.. _ilogb: https://en.cppreference.com/w/cpp/numeric/math/ilogb

.. |ilogb| replace:: ``ilogb``

.. _logb: https://en.cppreference.com/w/cpp/numeric/math/logb

.. |logb| replace:: ``logb``

.. _nextafter: https://en.cppreference.com/w/cpp/numeric/math/nextafter 

.. |nextafter| replace:: ``nextafter``

.. Next line is not a typo, nexttoward is documented on the same page as nextafter.
.. _nexttoward: https://en.cppreference.com/w/cpp/numeric/math/nextafter

.. |nexttoward| replace:: ``nexttoward``

.. _copysign: https://en.cppreference.com/w/cpp/numeric/math/copysign

.. |copysign| replace:: ``copysign``

.. list-table::
   :align: left

   * - |frexp|_ [#since_kokkos_5_1]_
     - decomposes a number into significand and base-:math:`2` exponent
   * - |ldexp|_ [#since_kokkos_5_1]_
     - multiplies a number by :math:`2` raised to an integral power
   * - |modf|_ [#since_kokkos_5_1]_
     - decomposes a number into integer and fractional parts
   * - |scalbn|_ [#since_kokkos_5_1]_
       |scalbln|_ [#since_kokkos_5_1]_
     - multiplies a number by ``FLT_RADIX`` raised to a power
   * - |ilogb|_ [#since_kokkos_5_1]_
     - extracts exponent of the number
   * - |logb|_
     - extracts exponent of the number
   * - |nextafter|_
       |nexttoward|_ [#since_kokkos_5_1]_
     - next representable floating-point value towards the given value
   * - |copysign|_
     - copies the sign of a floating point value

Classification and comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _fpclassify: https://en.cppreference.com/w/cpp/numeric/math/fpclassify

.. |fpclassify| replace:: ``fpclassify``

.. _isfinite: https://en.cppreference.com/w/cpp/numeric/math/isfinite

.. |isfinite| replace:: ``isfinite``

.. _isinf: https://en.cppreference.com/w/cpp/numeric/math/isinf

.. |isinf| replace:: ``isinf``

.. _isnan: https://en.cppreference.com/w/cpp/numeric/math/isnan

.. |isnan| replace:: ``isnan``

.. _isnormal: https://en.cppreference.com/w/cpp/numeric/math/isnormal

.. |isnormal| replace:: ``isnormal``

.. _signbit: https://en.cppreference.com/w/cpp/numeric/math/signbit

.. |signbit| replace:: ``signbit``

.. _isgreater: https://en.cppreference.com/w/cpp/numeric/math/isgreater

.. |isgreater| replace:: ``isgreater``

.. _isgreaterequal: https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal

.. |isgreaterequal| replace:: ``isgreaterequal``

.. _isless: https://en.cppreference.com/w/cpp/numeric/math/isless

.. |isless| replace:: ``isless``

.. _islessequal: https://en.cppreference.com/w/cpp/numeric/math/islessequal

.. |islessequal| replace:: ``islessequal``

.. _islessgreater: https://en.cppreference.com/w/cpp/numeric/math/islessgreater

.. |islessgreater| replace:: ``islessgreater``

.. _isunordered: https://en.cppreference.com/w/cpp/numeric/math/isunordered

.. |isunordered| replace:: ``isunordered``

.. list-table::
   :align: left

   * - |fpclassify|_ [#since_kokkos_5_1]_
     - categorizes the given floating-point value
   * - |isfinite|_
     - checks if the given number has finite value
   * - |isinf|_
     - checks if the given number is infinite
   * - |isnan|_
     - checks if the given number is NaN
   * - |isnormal|_ [#since_kokkos_5_1]_
     - checks if the given number is normal
   * - |signbit|_
     - checks if the given number is negative
   * - |isgreater|_ [#since_kokkos_5_1]_
     - checks if the first floating-point argument is greater than the second
   * - |isgreaterequal|_ [#since_kokkos_5_1]_
     - checks if the first floating-point argument is greater than or equal to the second
   * - |isless|_ [#since_kokkos_5_1]_
     - checks if the first floating-point argument is less than the second
   * - |islessequal|_ [#since_kokkos_5_1]_
     - checks if the first floating-point argument is less than or equal to the second
   * - |islessgreater|_ [#since_kokkos_5_1]_
     - checks if the first floating-point argument is less or greater than the second
   * - |isunordered|_ [#since_kokkos_5_1]_
     - checks if two floating-point values are unordered

------------

Other (non-standard) functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: These functions are not provided by the C++ standard library.


.. list-table::
   :align: left

   * - ``rsqrt`` [#since_kokkos_4_1]_
     - computes the reciprocal square root (:math:`1/\sqrt{x}`)
   * - ``rcp`` [#since_kokkos_5_1]_
     - computes the reciprocal (:math:`1/x`)

------------

Notes
-----

.. _openIssue: https://github.com/kokkos/kokkos/issues/new

.. |openIssue| replace:: **open an issue**

.. _issue4767: https://github.com/kokkos/kokkos/issues/4767

.. |issue4767| replace:: **Issue #4767**

.. |KnownIssues| replace:: known issues

* **Feel free to** |openIssue|_ **if you need one of the functions that is currently not implemented.** |issue4767|_ **is keeping track of these and has notes about implementability.**
* Beware the using-directive ``using namespace Kokkos;`` will cause
  compilation errors with unqualified calls to math functions.  Use explicit
  qualification (``Kokkos::sqrt``) or using-declaration (``using
  Kokkos::sqrt;``) instead.  (See |KnownIssues|_)
* Math functions were removed from the ``Kokkos::Experimental::`` namespace in version 4.3
* Support for quadruple precision floating-point ``__float128`` can be enabled
  via ``-DKokkos_ENABLE_LIBQUADMATH=ON``.

------------

See also
--------

`Mathematical constant <mathematical-constants.md>`_

`Numeric traits <numeric-traits.md>`_  
