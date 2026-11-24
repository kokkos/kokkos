Mathematical constants
======================

.. role::cpp(code)
    :language: cpp

.. _source_math_constants: https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalConstants.hpp

.. |source_math_constants| replace:: ``<Kokkos_MathematicalConstants.hpp>``

Defined in header |source_math_constants|_
which is included from ``<Kokkos_Core.hpp>``

.. attention::
   **Recommendation:** Since Kokkos 5.X requires C++20, users are encouraged to
   use the standard library constants (``std::numbers::*``) directly. The
   ``Kokkos::numbers`` namespace is maintained for backward compatibility and
   is implemented via `using-declarations
   <https://en.cppreference.com/w/cpp/language/namespace.html#Using-declarations>`__
   of the standard library constants.

Usage
-----

.. code-block:: cpp

   auto const x = Kokkos::numbers::pi_v<float>;
   auto const y = Kokkos::numbers::sqrt2_v<float>;

.. _cpp_reference_numbers: https://en.cppreference.com/w/cpp/numeric/constants

.. |cpp_reference_numbers| replace:: ``<numbers>``

Provides access to the mathematical constants defined in the C++20
|cpp_reference_numbers|_ header in the Standard Library.

All constants are defined in the ``Kokkos::numbers::`` namespace.

Variable Templates
------------------
The following are variable templates defined for standard floating-point types
(``float``, ``double``, ``long double``).

.. list-table::
   :align: left
   :header-rows: 1

   * - Template name
     - Math symbol
     - Description
   * - ``e_v``
     - :math:`e`
     - Base of the natural logarithm
   * - ``log2e_v``
     - :math:`\log_{2}{e}`
     - Base-2 logarithm of e
   * - ``log10e_v``
     - :math:`\log_{10}{e}`
     - Base-10 logarithm of e
   * - ``pi_v``
     - :math:`\pi`
     - Ratio of a circle's circumference to its diameter
   * - ``inv_pi_v``
     - :math:`\frac{1}{\pi}`
     - Inverse of pi
   * - ``inv_sqrtpi_v``
     - :math:`\frac{1}{\sqrt{\pi}}`
     - Inverse of the square root of pi
   * - ``ln2_v``
     - :math:`\ln{2}`
     - Natural logarithm of 2
   * - ``ln10_v``
     - :math:`\ln{10}`
     - Natural logarithm of 10
   * - ``sqrt2_v``
     - :math:`\sqrt{2}`
     - Square root of 2
   * - ``sqrt3_v``
     - :math:`\sqrt{3}`
     - Square root of 3
   * - ``inv_sqrt3_v``
     - :math:`\frac{1}{\sqrt{3}}`
     - Inverse of the square root of 3
   * - ``egamma_v``
     - :math:`\gamma`
     - Euler-Mascheroni constant
   * - ``phi_v``
     - :math:`\varphi`
     - Golden ratio constant :math:`\frac{1+\sqrt{5}}{2}`

Convenience Constants (``double``)
----------------------------------
For each variable template listed above, Kokkos provides an ``inline constexpr
double`` constant without the ``_v`` suffix. These are shorthand for the
``double`` specialization.

* ``Kokkos::numbers::pi`` is equivalent to ``Kokkos::numbers::pi_v<double>``
* ``Kokkos::numbers::e`` is equivalent to ``Kokkos::numbers::e_v<double>``

------------

Notes
-----

.. |KnownIssues| replace:: known issues

.. important::
   **Portability:** Passing mathematical constants by reference or taking their
   address in device code is not supported by some toolchains and hence not
   portable.  (See |KnownIssues|_)

.. note::
   **Quadruple Precision:** Support for quadruple precision floating-point
   ``__float128`` can be enabled via ``-DKokkos_ENABLE_LIBQUADMATH=ON``.

------------

Example
-------

.. code-block:: cpp

    KOKKOS_FUNCTION void example() {
        // Preferred C++20 usage
        constexpr auto pi_f = std::numbers::pi_v<float>;
        
        // Kokkos namespace usage (backward compatibility)
        constexpr auto pi = Kokkos::numbers::pi_v<float>;

        auto const x = Kokkos::sin(pi_f / 6);
    }

------------

See also
--------

.. seealso::
`Common mathematical functions <mathematical-functions.md>`_

   `Numeric traits <numeric-traits.md>`_
