``Array``
=========

.. role:: cpp(code)
    :language: cpp

..
  The (public header) file the user will include in their code

Defined in header ``<Kokkos_Array.hpp>`` which is included from ``<Kokkos_Core.hpp>``

..
  High-level, human-language summary of what the thing does, and if possible, brief statement about why it exists (2 - 3 sentences, max);

Description
-----------

``Kokkos::Array`` is a contiguous aggregate owning container storing a fixed size sequence of objects (models holding exactly N elements).

* This is intended as a replacement for ``std::array<T, N>``.
* This container is an owning container (the data is embedded in the container itself).
* This container is an aggregate type with the same semantics as a struct holding a C-style array ``T[N]`` as its only non-static data member when ``N > 0``; otherwise, it is an empty container.
* Unlike a C-style array, it doesn't decay to ``T*`` automatically.
* As an aggregate type, it can be initialized with aggregate-initialization given at most ``N`` initializers that are convertible to ``T``: ``Kokkos::Array<int, 3> a = { 1, 2, 3 };``.

..
  The API of the entity.

Interface
---------

.. versionchanged:: 4.4.0

.. cpp:struct:: template <class T, size_t N> Array

  ..
    Template parameters (if applicable)
    Omit template parameters that are just used for specialization/are deduced/ and/or should not be exposed to the user.

  .. rubric:: Template Parameters

  :tparam T: The type of the element being stored.
  :tparam N: The number of elements being stored.

  .. rubric:: Public Types

  .. cpp:type:: value_type = T
  .. cpp:type:: pointer = T*
  .. cpp:type:: const_pointer = const T*
  .. cpp:type:: reference = T&
  .. cpp:type:: const_reference = const T&
  .. cpp:type:: size_type = size_t
  .. cpp:type:: difference_type = ptrdiff_t

  .. rubric:: Public Member Functions

  .. cpp:function:: static constexpr bool empty() noexcept

    :return: ``N == 0``
    :since: ``noexcept`` since 5.0

  .. cpp:function:: static constexpr size_type size() noexcept
  .. cpp:function:: constexpr size_type max_size() const noexcept

    :return: ``N``
    :since: ``noexcept`` since 5.0

  .. cpp:function:: constexpr reference operator[](size_t i)
  .. cpp:function:: constexpr const_reference operator[](size_t i) const

    :return: A reference to the ``i``-th element of the array.
    :since: No longer requires the argument to be of an integral type or an unscoped enum type. (since 5.1)

  .. cpp:function:: constexpr pointer data() noexcept
  .. cpp:function:: constexpr const_pointer data() const noexcept

    :return: A pointer to the first element of the array.  If ``N == 0``, the return value is unspecified and not dereferenceable.
    :since: ``noexcept`` since 5.0

  .. cpp:function:: constexpr pointer begin() noexcept
  .. cpp:function:: constexpr const_pointer begin() const noexcept
  .. cpp:function:: constexpr const_pointer cbegin() const  noexcept

    :return: ``data()``
    :since: since 5.0

  .. cpp:function:: constexpr pointer end() noexcept
  .. cpp:function:: constexpr const_pointer end() const noexcept
  .. cpp:function:: constexpr const_pointer cend() const noexcept

    :return: ``data() + size()``. The return value is not dereferenceable. If ``N == 0``, the return value will be equal to ``begin()``.
    :since: since 5.0


Deduction Guides
----------------

.. cpp:function:: template<class T, class... U> Array(T, U...) -> Array<T, 1 + sizeof...(U)>

Non-Member Functions
--------------------

..
  These should only be listed here if they are closely related. E.g. friend operators. However,
  something like view_alloc shouldn't be here for view

.. cpp:function:: template<class T, size_t N> constexpr bool operator==(const Array<T, N>& l, const Array<T, N>& r) noexcept

   :return: ``true`` if and only if ∀ the elements in ``l`` and ``r`` compare equal.

.. cpp:function:: template<class T, size_t N> constexpr bool operator!=(const Array<T, N>& l, const Array<T, N>& r) noexcept

   :return: ``!(l == r)``

.. cpp:function:: template<class T, size_t N> constexpr kokkos_swap(Array<T, N>& l, Array<T, N>& r) noexcept(N == 0 || is_nothrow_swappable_V<T>)

   :return: If ``T`` is swappable or ``N == 0``, each of the elements in `l` and `r` are swapped via ``kokkos_swap``.

.. cpp:function:: template<class T, size_t N> constexpr Array<remove_cv_t<T>, N> to_array(T (&a)[N])
.. cpp:function:: template<class T, size_t N> constexpr Array<remove_cv_t<T>, N> to_array(T (&&a)[N])

   :return: An ``Array`` containing the elements copied/moved from ``a``.

.. cpp:function:: template<size_t I, class T, size_t N> constexpr T& get(Array<T, N>& a) noexcept
.. cpp:function:: template<size_t I, class T, size_t N> constexpr const T& get(const Array<T, N>& a) noexcept

   :return: ``a[I]`` for (tuple protocol / structured binding support)

.. cpp:function:: template<size_t I, class T, size_t N> constexpr T&& get(Array<T, N>&& a) noexcept
.. cpp:function:: template<size_t I, class T, size_t N> constexpr const T&& get(const Array<T, N>&& a) noexcept

   :return: ``std::move(a[I])`` (for tuple protocol / structured binding support)

.. cpp:function:: template<class T, size_t N> constexpr T* begin(Array<T, N>& a) noexcept
.. cpp:function:: template<class T, size_t N> constexpr const T* begin(const Array<T, N>& a) noexcept

   :return: ``a.data()``

.. cpp:function:: template<class T, size_t N> constexpr T* end(Array<T, N>& a) noexcept
.. cpp:function:: template<class T, size_t N> constexpr const T* end(const Array<T, N>& a) noexcept

   :return: ``a.data() + a.size()``

Deprecated Interface
--------------------
.. deprecated:: 4.4.00

.. cpp:struct:: template<class T = void, size_t N = KOKKOS_INVALID_INDEX, class Proxy = void> Array

* The primary template was an contiguous aggregate owning container of exactly ``N`` elements of type ``T``.
* This container did not support move semantics.

.. cpp:struct:: template<class T, class Proxy> Array<T, 0, Proxy>

* This container was an empty container.

.. cpp:struct:: template<class T> Array<T, KOKKOS_INVALID_INDEX, Array<>::contiguous>

* This container was a non-owning container.
* This container had its size determined at construction time.
* This container could be assigned from any ``Array<T, N , Proxy>``.
* Assignment did not change the size of this container.
* This container did not support move semantics.

.. cpp:struct:: template<class T> Array<T, KOKKOS_INVALID_INDEX, Array<>::strided>

* This container was a non-owning container.
* This container had its size and stride determined at construction time.
* This container could be assigned from any ``Array<T, N , Proxy>``.
* Assignment did not change the size or stride of this container.
* This container did not support move semantics.

.. cpp:struct:: template<> Array<void, KOKKOS_INVALID_INDEX, void>

   .. rubric:: Public Types

   .. cpp:type:: contiguous
   .. cpp:type:: stided

* This specialization defined the embedded tag types: ``contiguous`` and ``strided``.

Examples
________

.. code-block:: cpp

 #include "Kokkos_Core.hpp"
 #include <algorithm>
 #include <iostream>
 #include <iterator>
 #include <memory>
 #include <string>
 #include <string_view>
 #include <type_traits>
 #include <utility>

 // creates a constexpr array of string_view's
 constexpr auto w1n = Kokkos::to_array<std::string_view>(
     {"Mary", "Patricia", "Linda", "Barbara", "Elizabeth", "Jennifer"});
 static_assert(
     std::is_same_v<decltype(w1n), const Kokkos::Array<std::string_view, 6>>);
 static_assert(w1n.size() == 6 and w1n[5] == "Jennifer");

 extern int Main(int /* argc */, char const *const /* argv */[]);
 int Main(int /* argc */, char const *const /* argv */[]) {
   Kokkos::ScopeGuard _;

   // Construction uses aggregate initialization
   [[maybe_unused]] Kokkos::Array<int, 3> a1{
       {1, 2, 3}}; // Double-braces required in C++11
                   // and still allowed in C++14 and beyond

   Kokkos::Array<int, 3> a2 = {1, 2, 3}; // Double braces never required after =

   // Output is 3 2 1
   std::reverse_copy(std::data(a2), end(a2),
                     std::ostream_iterator<int>(std::cout, " "));
   std::cout << '\n';

   // Ranged for loop is supported
   // Output is E Ǝ
   Kokkos::Array<std::string, 2> a3{"E", "\u018E"};
   for (const auto &s : a3)
     std::cout << s << ' ';
   std::cout << '\n';

   // Deduction guide for array creation
   [[maybe_unused]] Kokkos::Array a4{3.0, 1.0, 4.0}; // Kokkos::Array<double, 3>

   // Behavior of unspecified elements is the same as with built-in arrays
   [[maybe_unused]] Kokkos::Array<int, 2> a5; // No list init, a5[0] and a5[1]
                                              // are default initialized
   [[maybe_unused]] Kokkos::Array<int, 2>
       a6{}; // List init, both elements are value
             // initialized, a6[0] = a6[1] = 0
   [[maybe_unused]] Kokkos::Array<int, 2> a7{
       1}; // List init, unspecified element is value
           // initialized, a7[0] = 1, a7[1] = 0

   // copies a string literal
   auto t1 = Kokkos::to_array("foo");
   static_assert(t1.size() == 4);

   // deduces both element type and length
   auto t2 = Kokkos::to_array({0, 2, 1, 3});
   static_assert(std::is_same_v<decltype(t2), Kokkos::Array<int, 4>>);

   // deduces length with element type specified
   // implicit conversion happens
   auto t3 = Kokkos::to_array<long>({0, 1, 3});
   static_assert(std::is_same_v<decltype(t3), Kokkos::Array<long, 3>>);

   auto t4 = Kokkos::to_array<std::pair<int, float>>(
       {{3, 0.0f}, {4, 0.1f}, {4, 0.1e23f}});
   static_assert(t4.size() == 3);

   // creates a non-copyable Kokkos::Array
   auto t5 = Kokkos::to_array({std::make_unique<int>(3)});
   static_assert(t5.size() == 1);

   // error: copying multidimensional arrays is not supported
   // char s[2][6] = {"nice", "thing"};
   // auto t6 = Kokkos::to_array(s);

   return 0;
 }
