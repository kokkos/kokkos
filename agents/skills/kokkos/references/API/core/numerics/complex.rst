``complex``
===========

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Complex.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Description
-----------

``complex`` is a class template for representing and manipulating complex numbers.

* This is intended as a replacement for ``std::complex<T>``.
* Note: If ``z`` has type ``Kokkos::complex<T>``, casting such as ``reinterpret_cast<T(&)[2]>(z)`` leads to undefined behavior (this differs from ``std::complex``).

Interface
---------

.. cpp:class:: template<class T> complex


  :tparam T: The type of the real and imaginary components.

  * :cpp:any:`T` must be a floating point type (``float``, ``double``, ``long double``) or an extended floating point type.

  * :cpp:any:`T` cannot be ``const`` and/or ``volatile`` qualified.

  * Some types might not work with a specific backend (such as ``long double`` on CUDA or SYCL).

  .. rubric:: Public Types:

  .. cpp:type:: value_type = T

  .. rubric:: Constructors & Assignment Operators:

  .. cpp:function:: complex()

    Default constructor zero initializes the real and imaginary components.

  .. cpp:function:: template<class U> complex(complex<U> z) noexcept

    Conversion constructor initializes the real component to ``static_cast<T>(z.real())`` and the imaginary component to ``static_cast<T>(z.imag())``.

    Constraints: ``U`` is convertible to ``T``.

  .. cpp:function:: complex(std::complex<T> z) noexcept
  .. cpp:function:: complex& operator=(std::complex<T> z) noexcept

    Implicit conversion from ``std::complex`` initializes the real component to ``z.real()`` and the imaginary component to ``z.imag()``.

  .. cpp:function:: constexpr complex(T r) noexcept
  .. cpp:function:: constexpr complex& operator=(T r) noexcept

    Initializes the real component to ``r`` and zero initializes the imaginary component.

  .. cpp:function:: constexpr complex(T r, T i) noexcept

    Initializes the real component to ``r`` and the imaginary component to ``i``.

  .. cpp:function:: template<class U> complex(const volatile complex<U>&) noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator=(const complex&) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: volatile complex& operator=(const volatile complex&) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: complex& operator=(const volatile complex&) noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator=(const volatile T&) noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator=(const T&) volatile noexcept
  
    .. deprecated:: 4.0.0

    .. note::
      
      Some of the deprecated assignment operators have templated implementations so as not to be copy assignment operators.

  .. rubric:: Public Member Functions:

  .. cpp:function:: operator std::complex<T>() const noexcept

    Conversion operator to ``std::complex``.

  .. cpp:function:: constexpr T& real() noexcept
  .. cpp:function:: constexpr T real() const noexcept

    :return: The value of the real component.

  .. cpp:function:: constexpr void real(T r) noexcept

    Assigns ``r`` to the real component.

  .. cpp:function:: constexpr T& imag() noexcept
  .. cpp:function:: constexpr T imag() const noexcept

    :return: The value of the imaginary component.

  .. cpp:function:: constexpr void imag(T i) noexcept

    Assigns ``i`` to the imaginary component.

  .. cpp:function:: constexpr complex& operator+=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator+=(T v) noexcept

    Adds the complex value ``complex(v)`` to the complex value ``*this`` and stores the sum in ``*this``.

  .. cpp:function:: constexpr complex& operator-=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator-=(T v) noexcept

    Subtracts the complex value ``complex(v)`` from the complex value ``*this`` and stores the difference in ``*this``.

  .. cpp:function:: constexpr complex& operator*=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator*=(T v) noexcept

    Multiplies the complex value ``complex(v)`` by the complex value ``*this`` and stores the product in ``*this``.

  .. cpp:function:: constexpr complex& operator/=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator/=(T v) noexcept

    Divides the complex value ``complex(v)`` into the complex value ``*this`` and stores the quotient in ``*this``.

  .. cpp:function:: volatile T& real() volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: T real() const volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: volatile T& imag() volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: T imag() const volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator+=(const volatile complex& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator+=(const volatile T& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator-=(const volatile complex& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator-=(const volatile T& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator*=(const volatile complex& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator*=(const volatile T& v) volatile noexcept
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator/=(const volatile complex& v) volatile noexcept(noexcept(T{}/T{}))
  
    .. deprecated:: 4.0.0

  .. cpp:function:: void operator/=(const volatile T& v) volatile noexcept(noexcept(T{}/T{}))
  
    .. deprecated:: 4.0.0


  .. rubric:: Non-Member Functions:

  .. cpp:function:: template<typename T1, typename T2> bool operator==(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator==(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator==(T1 x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator==(complex<T1> x, std::complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator==(std::complex<T1> x, complex<T2> y) noexcept

    :return: ``true`` if and only if the real component of ``complex(x)`` equals the real component of ``complex(y)`` and the imaginary component of ``complex(x)`` equals the imaginary component of ``complex(y)``.

  .. cpp:function:: template<typename T1, typename T2> bool operator!=(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator!=(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator!=(T1 x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator!=(complex<T1> x, std::complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> bool operator!=(std::complex<T1> x, complex<T2> y) noexcept

    :return: ``!(x == y)``

  .. cpp:function:: template<typename T> complex<T> operator+(complex<T> x) noexcept

    :return: ``x``

  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator+(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator+(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator+(T1 x, complex<T2> y) noexcept

    :return: The complex value ``complex(x)`` added to the complex value ``complex(y)``.

  .. cpp:function:: template<typename T> complex<T> operator-(complex<T> x) noexcept

    :return: ``complex(-x.real(), -x.imag())``

  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator-(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator-(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator-(T1 x, complex<T2> y) noexcept

    :return: The complex value ``complex(y)`` subtracted from the complex value ``complex(x)``.

  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator*(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator*(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator*(T1 x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator*(std::complex<T1> x, complex<T2> y) noexcept

    :return: The complex value ``complex(x)`` multiplied by the complex value ``complex(y)``.

  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator/(complex<T1> x, complex<T2> y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator/(complex<T1> x, T2 y) noexcept
  .. cpp:function:: template<typename T1, typename T2> complex<std::common_type_t<T1, T2>> operator/(T1 x, complex<T2> y) noexcept

    :return: The complex value ``complex(y)`` divided into the complex value ``complex(x)``.

  .. cpp:function:: template<typename T> std::istream& operator>>(std::ostream& i, complex<T>& x)

    Extracts a complex number `x` of the form: ``u``, ``(u)`` or ``(u,v)`` where ``u`` is the real part and ``v`` is the imaginary part and returns ``i``.

  .. cpp:function:: template<typename T> std::ostream& operator<<(std::ostream& o, complex<T> x)

    :return: ``o << std::complex(x)``

  .. cpp:function:: template<typename T> T real(complex<T> x) noexcept

    :return: ``x.real()``.

  .. cpp:function:: template<typename T> T imag(complex<T> x) noexcept

    :return: ``x.imag()``.

  .. cpp:function:: template<typenmame T> complex<T> polar(T rho, T theta = T())

    :return: The ``complex`` value corresponding to a complex number whose magnitude  is ``rho`` and whose phase angle is ``theta``.

  .. cpp:function:: template<typename T> T abs(complex<T> x)

    :return: The magnitude of ``x``.

  .. cpp:function:: template<typename T1, typename T2> complex<U> pow(complex<T1> x, complex<T2> y)
  .. cpp:function:: template<typename T1, typename T2> complex<U> pow(complex<T1> x, T2 y)
  .. cpp:function:: template<typename T1, typename T2> complex<U> pow(T1 x, complex<T2> y)

    :return: The complex power of base ``x`` raised to the ``y``-th power,
             defined as ``exp(y * log(x))``.
             ``U`` is ``float`` if ``T1`` and ``T2`` are ``float``;
             otherwise ``U`` is ``long double`` if ``T1`` or ``T2`` is ``long double``;
             otherwise ``U`` is ``double``.

  .. cpp:function:: template<typename T> complex<T> sqrt(complex<T> x)

    :return: The complex square root of ``x``, in the range of the right half-plane.

  .. cpp:function:: template<typename T> complex<T> conj(complex<T> x) noexcept

    :return: The complex conjugate of ``x``.

  .. cpp:function:: template<typename T> complex<T> exp(complex<T> x)
  .. cpp:function:: template<typename T> complex<T> exp(std::complex<T> x)

    :return: The complex base-e exponential of ``complex(x)``.

  .. cpp:function:: template<typename T> complex<T> log(complex<T> x)

    :return: The complex natural (base-e) logarithm of x.

  .. cpp:function:: template<typename T> complex<T> log10(complex<T> x)

    :return: The complex common (base-10) logarithm of ``x``, defined as ``log(x) / log(10)``.

  .. cpp:function:: template<typename T> complex<T> sin(complex<T> x)

    :return: The complex sine of ``x``.

  .. cpp:function:: template<typename T> complex<T> cos(complex<T> x)

    :return: The complex cosine of ``x``.

  .. cpp:function:: template<typename T> complex<T> tan(complex<T> x)

    :return: The complex tangent of ``x``.

  .. cpp:function:: template<typename T> complex<T> sinh(complex<T> x)

    :return: The complex hyperbolic sine of ``x``.

  .. cpp:function:: template<typename T> complex<T> cosh(complex<T> x)

    :return: The complex hyperbolic cosine of ``x``.

  .. cpp:function:: template<typename T> complex<T> tanh(complex<T> x)

    :return: The complex hyperbolic tangent of ``x``.

  .. cpp:function:: template<typename T> complex<T> asinh(complex<T> x)

    :return: The complex arc hyperbolic sine of ``x``.

  .. cpp:function:: template<typename T> complex<T> acosh(complex<T> x)

    :return: The complex arc hyperbolic cosine of ``x``.

  .. cpp:function:: template<typename T> complex<T> atanh(complex<T> x)

    :return: The complex arc hyperbolic tangent of ``x``.

  .. cpp:function:: template<typename T> complex<T> asin(complex<T> x)

    :return: The complex arc sine of ``x``.

  .. cpp:function:: template<typename T> complex<T> acos(complex<T> x)

    :return: The complex arc cosine of ``x``.

  .. cpp:function:: template<typename T> complex<T> atan(complex<T> x)

    :return: The complex arc tangent of ``x``.

  .. cpp:function:: template<size_t I, typename T> constexpr T& get(complex<T>& z) noexcept
  .. cpp:function:: template<size_t I, typename T> constexpr T&& get(complex<T>&& z) noexcept
  .. cpp:function:: template<size_t I, typename T> constexpr const T& get(const complex<T>& z) noexcept
  .. cpp:function:: template<size_t I, typename T> constexpr const T&& get(complex<T>&& z) noexcept

    Tuple protocol / structured binding support.

    :return: A reference to the real part of ``z`` if ``I == 0`` is ``true``;
             a reference to the imaginary part of ``z`` if ``I == 1`` is ``true``.

