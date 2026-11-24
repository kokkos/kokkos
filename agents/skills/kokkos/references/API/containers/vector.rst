
.. role:: cpp(code)
    :language: cpp

``vector`` [DEPRECATED]
=======================

Header file: ``<Kokkos_Vector.hpp>`` (deprecated in Kokkos 4.3)

The Kokkos Vector is semantically similar to the std::vector, but it is designed to overcome issues with memory allocations and copies when working with devices that have different memory spaces. The ``Kokkos::Vector`` is a Rank-1 DualView that implements the same interface as the std::vector. This allows programs that rely heavily on std::vector to grant access to program data from within a non-host execution space. Note that many of the std::vector compatible functions are host only, so access may be limited based on kernel complexity. Below is a synopsis of the class and the description for each method specifies whether it is supported on the host, device or both.

Usage
-----

.. code-block:: cpp

    Kokkos::vector<Scalar, Device> v(n,1);  // (deprecated since 4.3)
    v.push_back(2);
    v.resize(n+3);
    v.[n+1] = 3;
    v.[n+2] = 4;

Description
-----------

.. cpp:class:: template<class Scalar, class Arg1Type = void> vector :  public DualView<Scalar*, LayoutLeft, Arg1Type>

   .. rubric:: Public Typedefs

   .. cpp:type:: Scalar value_type;

      Scalar value type

   .. cpp:type:: Scalar* pointer;

      Scalar pointer type

   .. cpp:type:: const Scalar* const_pointer;

      Const Scalar pointer type

   .. cpp:type:: Scalar& reference;

      Scalar reference type

   .. cpp:type:: const Scalar& const_reference;

      Const Scalar reference type

   .. cpp:type:: Scalar* iterator;

      Iterator type

   .. cpp:type:: const Scalar* const_iterator;

      Const iterator type

   .. rubric:: Accessors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION reference operator()(int i) const;

      Accessor

      .. warning:: Host only

   .. cpp:function:: KOKKOS_INLINE_FUNCTION reference operator[](int i) const;

      Accessor

      .. warning:: Host only

   .. rubric:: Constructors

   .. cpp:function:: vector();

      Construct empty vector

   .. cpp:function:: vector(int n, Scalar val = Scalar());

      Construct vector of size n + 10% and initialize values to ``val``

   .. rubric:: Other Public Methods

   .. cpp:function:: void resize(size_t n);

      Resize vector to size n + 10%

   .. cpp:function:: void resize(size_t n, const Scalar& val);

      Resize vector to size n + 10% and set values to ``val``

   .. cpp:function:: void assign(size_t n, const Scalar& val);

      Set n values to ``val`` will auto synchronize between host and device

   .. cpp:function:: void reserve(size_t n);

      Same as resize (for compatibility)

   .. cpp:function:: void push_back(Scalar val);

      Resize vector to size() + 1 and set last value to val

      .. warning:: Host only, auto synchronize device

   .. cpp:function:: void pop_back();

      Reduce size() by 1

   .. cpp:function:: void clear();

      Set size() to 0

   .. cpp:function:: size_type size() const;

      Return number of elements in vector

   .. cpp:function:: size_type max_size() const;

      Return maximum possible number of elements

   .. cpp:function:: size_type span() const;

      Return memory used by vector

   .. cpp:function:: bool empty() const;

      Returns true if vector is empty

   .. cpp:function:: pointer data() const;

      Returns pointer to the underlying array

      .. warning:: Host only

   .. cpp:function:: iterator begin() const;

      Returns iterator starting at the beginning

      .. warning:: Host only

   .. cpp:function:: iterator end() const;

      Returns iterator past the last element

      .. warning:: Host only

   .. cpp:function:: reference front();

      Returns reference to the front of the list

      .. warning:: Host only

   .. cpp:function:: reference back();

      Returns reference to the last element in the list

      .. warning:: Host only

   .. cpp:function:: const_reference front() const;

      Returns const reference to the front of the list

      .. warning:: Host only

   .. cpp:function:: const_reference back() const;

      Returns const reference to the last element in the list

      .. warning:: Host only

   .. cpp:function:: size_t lower_bound(const size_t& start, const size_t& theEnd, const Scalar& comp_val) const;

      Return the index of largest value satisfying val < comp_val within the range start-theEnd

      .. warning:: Host only

   .. cpp:function:: bool is_sorted();

      Return true if the list is sorted

   .. cpp:function:: iterator find(Scalar val) const;

      Return iterator pointing to element matching ``val``

   .. cpp:function:: void device_to_host();

      Copy data from device to host

   .. cpp:function:: void host_to_device() const;

      Copy data from host to device

   .. cpp:function:: void on_host();

      Update/synchronize data in dual view from host perspective

   .. cpp:function:: void on_device();

      Update/synchronize data in dual view from the device perspective

   .. cpp:function:: void set_overallocation(float extra);

      Set the data buffer available at the end of the vector

   .. cpp:function:: constexpr bool is_allocated() const;

      Returns true if the internal views (host and device) are allocated (non-null pointers).
