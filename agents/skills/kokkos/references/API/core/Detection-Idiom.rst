Detection Idiom
===============

.. role:: cpp(code)
    :language: cpp

.. attention::
   Prior to C++20, the Detection Idiom was the best-in-class mechanism for detecting embedded typedefs and the
   validity of C++ expressions.  Concepts, the language feature added in C++20, is superior to and easier to
   use than the Detection Idiom and should be the first approach going forward.

The Detection Idiom is used to recognize, in an SFINAE-friendly way, the validity of any C++ expression.

Header File: ``<Kokkos_DetectionIdiom.hpp>``

The Kokkos Detection Idiom is based upon the detection idiom from Version 2 of the C++ Extensions for
Library Fundamentals, ISO/IEC TS 19568:2017, a draft of which can be found `here <https://cplusplus.github.io/fundamentals-ts/v2.html#meta.detect>`.

The original C++ proposal can be found at `here <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf>`.

API
---

.. code-block:: cpp

    // VOID_T and DETECTOR are exposition-only and not intended to be used directly.

    // Convenient metafunction to leverage SFINAE
    template<class...>
    using VOID_T = void;

    // Primary template for types not supporting the archetypal Op<Args...>
    template<class Default, class /* AlwaysVoid */, template<class...> class /* Op */, class... /* Args */>
    struct DETECTOR {
        using value_t = std::false_type;
        using type    = Default;
    };

    // Specialization for types supporting the archetypal Op<Args...>
    template<class Default, template<class...> class Op, class... Args>
    struct DETECTOR<Default, VOID_T<Op<Args...>>, Op, Args...> {
        using value_t = std::true_type;
        using type    = Op<Args...>;
    };

.. code-block:: cpp

    namespace Kokkos {

    // Simplification of the type returned by detected_t for types not supporting the archetype provided
    struct nonesuch {
        nonesuch(nonesuch&&) = delete;
        ~nonesuch() = delete;
    };

    // is_detected is an alias for std::true_type if Op<Args...> is a valid type
    // otherwise, an alias for std::false_type

    template <template <class...> class Op, class... Args>
    using is_detected =
        typename DETECTOR<nonesuch, void, Op, Args...>::value_t;

    // detected_t is an alias for Op<Args...> if Op<Args...> is a valid type
    //  otherwise, an alias for Kokkos::nonesuch

    template <template <class...> class Op, class... Args>
    using detected_t = typename DETECTOR<nonesuch, void, Op, Args...>::type;

    // detected_or_t is an alias for Op<Args...> if Op<Args...> is a valid type
    //  otherwise, an alias for Default

    template <class Default, template <class...> class Op, class... Args>
    using detected_or_t = typename DETECTOR<Default, void, Op, Args...>::type;

    // is_detected_exact is an alias for std::true_type if Op<Args...> is the same type as Expected
    //  otherwise, an alias for std::false_type

    template <class Expected, template <class...> class Op, class... Args>
    using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

    // is_detected_convertible is an alias for std::true_type if Op<Args...> is convertible to To
    //  otherwise, an alias for std::false_type

    template <class To, template <class...> class Op, class... Args>
    using is_detected_convertible =
        std::is_convertible<detected_t<Op, Args...>, To>;

    // C++17 or later convenience variables

    template <template <class...> class Op, class... Args>
    inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

    template <class Expected, template <class...> class Op, class... Args>
    inline constexpr bool is_detected_exact_v =
        is_detected_exact<Expected, Op, Args...>::value;

    template <class Expected, template <class...> class Op, class... Args>
    inline constexpr bool is_detected_convertible_v =

    } // Kokkos namespace

Examples
--------

Detecting an expression via Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _Concepts: https://eel.is/c++draft/concepts

Suppose we wanted to detect if a given type ``T`` is copy assignable.

First, we write a concept to detect it:

.. code-block:: cpp

   template<class T>
   concept CopyAssignable = requires(T& lhs, const T& rhs) {
      lhs = rhs;
   };

Then, constrain a function template:

.. code-block:: cpp

   template<class U>
       requires(CopyAssignable<U>)
   void DoSomething(U& u) {
    // ...
   }


Alternate terse syntax:

.. code-block:: cpp

   template<CopyAssignable U>
   void DoSomething(U& u) {
    // ...
   }

If we also wanted to check that the return type of the copy assignment is ``T&``, we would use:

.. code-block:: cpp

   #include <concepts>

   template<class T>
   concept CanonicalCopyAssignable = requires(T& lhs, const T& rhs) {
       { lhs = rhs } -> std::same_as<T&>;
   };

.. important::
   Both Kokkos and the C++ standard library have
   already defined many concepts. One should prefer to use those over rolling your own.
   Besides being standardized, they are rigorous about covering corner cases.
   The concepts provided by the standard library can be found at 
   <https://eel.is/c++draft/concepts> (although this list may contain concepts added since C++20).

Constraining a function template with the standard library concept ``std::assignable_from``:

.. code-block:: cpp

   #include <concepts>

   template<class U>
       requires std::assignable_from<U&, const U&>
   void DoSomething(U& u) {
    // ...
   }


Detecting an expression via The Detection Idiom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we needed to write a type trait to detect if a given type ``T`` is copy assignable. First we write an archetype helper alias:

.. code-block:: cpp

    template<class T>
    using copy_assign_t = decltype(std::declval<T&>() = std::declval<T const&>());

Then the trait can be easily expressed as:

.. code-block:: cpp

    template<class T>
    using is_copy_assignable = Kokkos::is_detected<copy_assign_t, T>;

If we also wanted to check that the return type of the copy assignment is ``T&``, we would use:

.. code-block:: cpp

    template<class T>
    using is_canonical_copy_assignable = Kokkos::is_detected_exact<T&, copy_assign_t, T>;

Detecting a nested typedef via Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want to use a nested ``MyType::difference_type`` if it exists, otherwise, we want to use ``std::ptrdiff_t``:

First, we need a concept to detect if ``MyType`` has a nested ``difference_type``:

.. code-block:: cpp

   template<class T>
   concept HasDifferenceType = requires {
       typename T::difference_type;
   };

Next, we write a helper struct to extract the type:

.. code-block:: cpp

   template<class In, class U>
   struct Select {
       using type = U;
   };

   template<class In, class U>
       requires HasDifferenceType<In>
   struct Select<In, U> {
       using type = typename In::difference_type;
   };

   template<class In, class U>
   using Select_t = typename Select<In, U>::type;

Then we can declare our type:

.. code-block:: cpp

   using our_difference_type = Select_t<MyType, std::ptrdiff_t>;


Detecting a nested typedef via The Detection Idiom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want to use a nested ``MyType::difference_type`` if it exists, otherwise, we want to use ``std::ptrdiff_t``:

First we write an archetype helper alias:

.. code-block:: cpp

    template<class T>
    using diff_t = typename T::difference_type;

Then we can declare our type:

.. code-block:: cpp

    using our_difference_type = Kokkos::detected_or_t<std::ptrdiff_t, diff_t, MyType>;
