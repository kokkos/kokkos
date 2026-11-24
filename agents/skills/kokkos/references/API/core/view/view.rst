``View``
========

Header File: ``<Kokkos_Core.hpp>``

.. _CppReferenceSharedPtr: https://en.cppreference.com/w/cpp/memory/shared_ptr

.. |CppReferenceSharedPtr| replace:: ``std::shared_ptr``

.. _ProgrammingGuide: ../../../ProgrammingGuide/View.rst#memory-access-traits

.. |ProgrammingGuide| replace:: Programming Guide

Class Interface
---------------

.. cpp:class:: template <class DataType, class... Properties> View

   Kokkos View is a potentially reference counted multi dimensional array with compile time layouts and memory space.
   Its semantics are similar to that of |CppReferenceSharedPtr|_.
   
   :tparam DataType: Defines the fundamental scalar type of the :cpp:class:`View` and its dimensionality.

      The basic structure is ``ScalarType STARS BRACKETS`` where the number of ``STARS`` denotes
      the number of runtime length dimensions and the number of ``BRACKETS`` defines the compile time dimensions.
      Due to C++ type restrictions runtime dimensions must come first.
      Examples:

      - :cpp:`double**`: 2D View of :cpp:`double` with 2 runtime dimensions
      - :cpp:`const int***[5][3]`: 5D View of :cpp:`int` with 3 runtime and 2 compile dimensions. 
         The data is :cpp:`const`.
      - :cpp:`Foo[6][2]`: 2D View of a class :cpp:`Foo` with 2 compile time dimensions.

   :tparam Properties...: Defines various properties of the :cpp:class:`View`, including layout, memory space, and memory traits.
   
      :cpp:class:`View`'s template parameters after ``DataType`` are variadic and optional, but must be specified in order. That means for example that :cpp:any:`LayoutType` can be omitted but if both :cpp:any:`MemorySpace` and :cpp:`MemoryTraits` are specified, :cpp:any:`MemorySpace` must come before :cpp:any:`MemoryTraits`.

      .. code-block:: cpp
         :caption: The ordering of View template parameters.

         template <class DataType [, class LayoutType] [, class MemorySpace] [, class MemoryTraits]>
         class View;

   :tparam LayoutType: Determines the mapping of indices into the underlying 1D memory storage.
   
      Kokkos comes with some built-in layouts:

      - :cpp:struct:`LayoutRight`: strides increase from the right most to the left most dimension.
         The last dimension has a stride of one.
         This corresponds to how C multi dimensional arrays (e.g :cpp:`foo[][][]`) are laid out in memory.
      - :cpp:struct:`LayoutLeft`: strides increase from the left most to the right most dimension.
         The first dimension has a stride of one. This is the layout Fortran uses for its arrays.
      - :cpp:struct:`LayoutStride`: strides can be arbitrary for each dimension.
   
   :tparam MemorySpace: Controls the storage location of the View.

      If omitted the default memory space of the default execution space is used (i.e. :cpp:expr:`DefaultExecutionSpace::memory_space`)

   :tparam MemoryTraits: Sets access properties via enum parameters for the struct template :cpp:struct:`MemoryTraits`. Possible template parameters are bitwise OR of the following flags: 

      - ``Unmanaged``
      - ``RandomAccess``
      - ``Atomic``
      - ``Restrict``
      - ``Aligned``

      See the sub-section on memory access traits in the |ProgrammingGuide|_ also for further information.

..
   Pushing a "namespace" here; this doesn't create a namespace entity but tells Sphinx that everything between here and the pop is part of the View class.
   All entities are still referenced via the scope (i.e. View::data_type).

.. cpp:namespace-push:: template <class DataType, class... Properties> View

Public Constants
^^^^^^^^^^^^^^^^

.. cpp:member:: static constexpr bool reference_type_is_lvalue_reference

   whether the reference type is a C++ lvalue reference.

Data Types
^^^^^^^^^^

.. cpp:type:: data_type

   The :cpp:any:`DataType` of the :cpp:class:`View`, note :cpp:type:`data_type` contains the array specifiers (e.g. :cpp:`int**[3]`)

.. cpp:type:: const_data_type

   :cpp:`const` version of :cpp:any:`DataType`, same as :cpp:type:`data_type` if that is already  :cpp:`const`.

.. cpp:type:: non_const_data_type

   Non-:cpp:`const` version of :cpp:any:`DataType`, same as :cpp:type:`data_type` if that is already non-:cpp:`const`.

.. cpp:type:: scalar_array_type

   If :cpp:any:`DataType` represents some properly specialised array data type such as Sacado FAD types, :cpp:type:`scalar_array_type` is the underlying fundamental scalar type.

.. cpp:type:: const_scalar_array_type

   :cpp:`const` version of :cpp:type:`scalar_array_type`, same as :cpp:type:`scalar_array_type` if that is already :cpp:`const`

.. cpp:type:: non_const_scalar_array_type

   Non-:cpp:`const` version of :cpp:type:`scalar_array_type`, same as :cpp:type:`scalar_array_type` if that is already non-:cpp:`const`.


Scalar Types
^^^^^^^^^^^^

.. cpp:type:: value_type

   The :cpp:type:`data_type` stripped of its array specifiers, i.e. the scalar type of the data the view is referencing (e.g. if :cpp:type:`data_type` is :cpp:`const int**[3]`, :cpp:type:`value_type` is :cpp:`const int`).

.. cpp:type:: const_value_type

   :cpp:`const` version of :cpp:type:`value_type`.

.. cpp:type:: non_const_value_type

   non-:cpp:`const` version of :cpp:type:`value_type`.


Spaces
^^^^^^

.. cpp:type:: execution_space

   The :ref:`execution space <api-execution-spaces>` associated with the view, will be used for
   performing view initialization, and certain deep_copy operations.

.. cpp:type:: memory_space

   The :ref:`memory space <api-memory-spaces>` where the :cpp:class:`View` data is stored.

.. cpp:type:: device_type

   the compound type defined by :cpp:expr:`Device<execution_space, memory_space>`

.. cpp:type:: memory_traits

   The memory traits of the view.

.. cpp:type:: host_mirror_space

   Host accessible memory space used in :cpp:type:`host_mirror_type`.

View Types
^^^^^^^^^^

.. cpp:type:: non_const_type

   this :cpp:class:`View` type with :cpp:type:`non_const_data_type` passed as the :cpp:any:`DataType` template parameter

.. cpp:type:: const_type

   this :cpp:class:`View` type with :cpp:type:`const_data_type` passed as the :cpp:any:`DataType` template parameter

.. cpp:type:: host_mirror_type

   compatible view type with the same :cpp:type:`data_type` and :cpp:type:`array_layout` stored in host accessible memory space.

   .. versionadded:: 5.0

.. cpp:type:: HostMirror

   compatible view type with the same :cpp:type:`data_type` and :cpp:type:`array_layout` stored in host accessible memory space.

   .. deprecated:: 5.0

Data Handles
^^^^^^^^^^^^

.. cpp:type:: reference_type

   return type of the view access operators.

   .. seealso::
      :cpp:func:`operator()`

      :cpp:func:`access()`


.. cpp:type:: pointer_type

   pointer to :cpp:type:`value_type`.


Other Types
^^^^^^^^^^^

.. cpp:type:: array_layout

   The :cpp:any:`LayoutType` of the :cpp:class:`View`.

.. cpp:type:: size_type

   index type associated with the memory space of this :cpp:class:`View`.

.. cpp:type:: dimension

   An integer array like type, able to represent the extents of the :cpp:class:`View`.

.. cpp:type:: specialize

   A specialization tag used for partial specialization of the mapping construct underlying a :cpp:class:`View`.


Constructors
^^^^^^^^^^^^

.. cpp:function:: View()

   Default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is :cpp:`nullptr`.

.. cpp:function:: template<class DT, class... Prop> View( const View<DT, Prop...>& rhs)

   Copy constructor with a compatible view. Follows :cpp:class:`View` assignment rules.

   .. seealso:: :ref:`api-view-assignment`

.. cpp:function:: View(View&& rhs)

   Move constructor

.. cpp:function:: template<class IntType> View( const std::string& name, const IntType& ... extents)

   Standard allocating constructor. The initialization is executed on the default
   instance of the execution space corresponding to :cpp:type:`memory_space` and fences it.

   :tparam IntType: an integral type

   :param name: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique.

   :param extents: Extents of the :cpp:class:`View`.

   .. rubric:: Requirements:

   - :cpp:expr:`sizeof(IntType...) == rank_dynamic()` or :cpp:expr:`sizeof(IntType...) == rank()`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.
   - :cpp:expr:`array_layout::is_regular == true`.

.. cpp:function:: View( const std::string& name, const array_layout& layout)

   Standard allocating constructor. The initialization is executed on the default
   instance of the execution space corresponding to :cpp:type:`memory_space` and fences it.

   :param name: a user provided label, which is used for profiling and debugging purposes.
      Names are not required to be unique.

   :param layout: an instance of a layout class.
      The number of valid extents must either match the :cpp:func:`rank_dynamic` or :cpp:func:`rank`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.

.. cpp:function:: template<class IntType> View( const ALLOC_PROP &prop, const IntType& ... extents)

   Allocating constructor with allocation properties (created by a call to :cpp:func:`view_alloc`). If an execution space is
   specified in :cpp:any:`prop`, the initialization uses it and does not fence.
   Otherwise, the :cpp:class:`View` is initialized using the default execution space instance corresponding to :cpp:type:`memory_space` and fences it.

   :tparam IntType: an integral type

   :param prop: An allocation properties object that is returned by :cpp:func:`view_alloc`.

   :param extents: Extents of the View.

   .. rubric:: Requirements:

   - :cpp:expr:`sizeof(IntType...) == rank_dynamic()` or :cpp:expr:`sizeof(IntType...) == rank()`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.
   - :cpp:expr:`array_layout::is_regular == true`.

.. cpp:function:: View( const ALLOC_PROP &prop, const array_layout& layout)

   Allocating constructor with allocation properties (created by a call to :cpp:func:`view_alloc`) and a layout object. If an execution space is
   specified in :cpp:any:`prop`, the initialization uses it and does not fence.
   Otherwise, the :cpp:class:`View` is initialized using the default execution space instance corresponding to :cpp:type:`memory_space` and fences it.

   :param prop: An allocation properties object that is returned by :cpp:func:`view_alloc`.

   :param layout: an instance of a layout class.
      The number of valid extents must either match the :cpp:func:`rank_dynamic` or :cpp:func:`rank`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.

.. cpp:function:: template<class IntType> View( pointer_type ptr, const IntType& ... extents)

   Unmanaged data wrapping constructor.

   :tparam IntType: an integral type

   :param ptr: pointer to a user provided memory allocation.
      Must provide storage of size :cpp:expr:`required_allocation_size(extents...)`

   :param extents: Extents of the :cpp:class:`View`.

   .. rubric:: Requirements:

   - :cpp:expr:`sizeof(IntType...) == rank_dynamic()` or :cpp:expr:`sizeof(IntType...) == rank()`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.
   - :cpp:expr:`array_layout::is_regular == true`.

.. cpp:function:: View( pointer_type ptr, const array_layout& layout)

   Unmanaged data wrapper constructor.

   :param ptr: pointer to a user provided memory allocation.
      Must provide storage of size :cpp:expr:`View::required_allocation_size(layout)`

   :param layout: an instance of a layout class.
      The number of valid extents must either match the dynamic rank or the total rank. In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.

.. cpp:function:: template<class IntType> View( const ScratchSpace& space, const IntType& ... extents)

   Constructor which acquires memory from a Scratch Memory handle.

   :tparam IntType: an integral type

   :param space: scratch memory handle.
      Typically returned from :cpp:func:`team_shmem`, :cpp:func:`team_scratch`, or :cpp:func:`thread_scratch` in ``TeamPolicy`` kernels.

   :param extents: Extents of the :cpp:class:`View`.

   .. rubric:: Requirements:

   - :cpp:expr:`sizeof(IntType...) == rank_dynamic()` or :cpp:expr:`sizeof(IntType...) == rank()`.
      In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.
   - :cpp:expr:`array_layout::is_regular == true`.

.. cpp:function:: View( const ScratchSpace& space, const array_layout& layout)

   Constructor which acquires memory from a Scratch Memory handle.

   :param space: scratch memory handle.
      Typically returned from :cpp:func:`team_shmem`, :cpp:func:`team_scratch`, or :cpp:func:`thread_scratch` in ``TeamPolicy`` kernels.

   :param layout: an instance of a layout class.
      The number of valid extents must either match the dynamic rank or the total rank. In the latter case, the extents corresponding to compile-time dimensions must match the :cpp:class:`View` type's compile-time extents.

.. cpp:function:: template<class DT, class... Prop> View( const View<DT, Prop...>& rhs, Args ... args)

   :param rhs: the :cpp:class:`View` to take a subview of
   :param args...: the subview slices as specified in :cpp:func:`subview`

   Subview constructor.

   .. seealso:: :cpp:func:`subview`

.. cpp:function:: explicit(traits::is_managed) View( const NATURAL_MDSPAN_TYPE& mds )

   :param mds: the mdspan to convert from.

   .. warning::

      :cpp:`explicit(bool)` is only available on C++20 and later. When building Kokkos with C++17, this constructor will be fully implicit.
      Be aware that later upgrading to C++20 will in some cases cause compilation issues in cases where :cpp:`traits::is_managed` is :cpp:`false`.

   :cpp:`NATURAL_MDSPAN_TYPE` is the :ref:`natural mdspan <api-view-natural-mdspans>` of the View. The *natural mdspan* is only available if :cpp:type:`array_layout` is one of :cpp:struct:`LayoutLeft`, :cpp:struct:`LayoutRight`,
   or :cpp:class:`LayoutStride`. This constructor is only available if *natural mdspan* is available.

   Constructs a :cpp:class:`View` by converting from :cpp:any:`mds`. The :cpp:class:`View` will be unmanaged and constructed as if by :cpp:`View(mds.data(), array_layout_from_mapping(mds.mapping()))`

   .. seealso:: :ref:`Natural mdspans <api-view-natural-mdspans>`

   .. versionadded:: 4.4.0

.. cpp:function:: template <class ElementType, class ExtentsType, class LayoutType, class AccessorType> explicit(SEE_BELOW) View(const mdspan<ElementType, ExtentsType, LayoutType, AccessorType>& mds)

   :tparam ElementType: the mdspan element type
   :tparam ExtentsType: the mdspan extents
   :tparam LayoutType: the mdspan layout
   :tparam AccessorType: the mdspan extents

   :param mds: the mdspan to convert from

   .. warning::

      :cpp:`explicit(bool)` is only available on C++20 and later. When building Kokkos with C++17, this constructor will be fully implicit.
      Be aware that later upgrading to C++20 will in some cases cause compilation issues in cases where the condition is false.

   Constructs a :cpp:class:`View` by converting from :cpp:any:`mds`.
   The :cpp:class:`View`'s :ref:`natural mdspan <api-view-natural-mdspans>` must be constructible from :cpp:any:`mds`. The :cpp:class:`View` will be constructed as if by :cpp:`View(NATURAL_MDSPAN_TYPE(mds))`

   In C++20:
      This constructor is implicit if :cpp:any:`mds` is implicitly convertible to the *natural mdspan* of the :cpp:class:`View`.

   .. versionadded:: 4.4.0


Data Access Functions
^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: template<class IntType> reference_type operator() (const IntType& ... indices) const

   :tparam IntType: an integral type

   :param indices: the indices of the element to get a reference to
   :return: a reference to the element at the given indices

   Returns a value of :cpp:type:`reference_type` which may or not be referenceable itself.
   The number of index arguments must match the :cpp:func:`rank` of the view.

   .. rubric:: Requirements:
   
   - :cpp:expr:`sizeof(IntType...) == rank_dynamic()`

.. cpp:function:: template<class IntType> reference_type access(const IntType& i0=0, const IntType& i1=0, \
         const IntType& i2=0, const IntType& i3=0, const IntType& i4=0, \
         const IntType& i5=0, const IntType& i6=0, const IntType& i7=0) const

   :tparam IntType: an integral type
   
   :param i0, i1, i2, i3, i4, i5, i6, i7: the indices of the element to get a reference to
   :return: a reference to the element at the given indices

   Returns a value of :cpp:type:`reference_type` which may or not be referenceable itself.
   The number of index arguments must be equal or larger than the :cpp:func:`rank` of the view.
   Index arguments beyond :cpp:func:`rank` must be :cpp:`0`, which will be enforced if :cpp:any:`KOKKOS_DEBUG` is defined.


Data Layout, Dimensions, Strides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: static constexpr size_t rank()

   :return: the rank of the view.

   .. versionadded:: 4.1

.. cpp:function:: static constexpr size_t rank_dynamic()

   :return: the number of runtime determined dimensions.

   .. versionadded:: 4.1

.. note::

   In practice, :cpp:func:`rank()` and :cpp:func:`rank_dynamic()` are not actually implemented as static member functions but ``rank`` and ``rank_dynamic`` underlying types have a nullary member function (i.e. callable with no argument).

.. versionchanged:: 4.1

   :cpp:func:`rank` and :cpp:func:`rank_dynamic` are static member constants that are convertible to :cpp:`size_t`.
   Their underlying types are unspecified, but equivalent to :cpp:`std::integral_constant` with a nullary member function callable from host and device side.
   Users are encouraged to use :cpp:`rank()` and :cpp:`rank_dynamic()` (akin to a static member function call) instead of relying on implicit conversion to an integral type.

   The actual type of :cpp:func:`rank` and :cpp:func:`rank_dynamic` as they were defined until Kokkos 4.1 was left up to the implementation (that is, up to the compiler not to Kokkos) but in practice it was often :cpp:`int` which means this change may yield warnings about comparing signed and unsigned integral types.
   It may also break code that was using the type of :cpp:func:`rank`.
   Furthermore, it appears that MSVC has issues with the implicit conversion to :cpp:`size_t` in certain constexpr contexts. Calling :cpp:func:`rank()` or :cpp:func:`rank_dynamic()` will work in those cases.

.. cpp:function:: constexpr array_layout layout() const

   :return: the layout object that can be used to to construct other views with the same dimensions.

.. cpp:function:: template<class iType> constexpr size_t extent( const iType& dim) const

   :tparam iType: an integral type
   :param dim: the dimension to get the extent of
   :return: the extent of dimension :cpp:any:`dim`

   .. rubric:: Preconditions:

   - :cpp:any:`dim` must be smaller than :cpp:func:`rank`.

.. cpp:function:: template<class iType> constexpr int extent_int( const iType& dim) const

   :tparam iType: an integral type
   :param dim: the dimension to get the extent of
   :return: the extent of dimension :cpp:any:`dim` as an :cpp:`int`

   Compared to :cpp:func:`extent` this function can be
   useful on architectures where :cpp:`int` operations are more efficient than :cpp:`size_t`.
   It also may eliminate the need for type casts in applications which
   otherwise perform all index operations with :cpp:`int`.

   .. rubric:: Preconditions:

   - :cpp:any:`dim` must be smaller than :cpp:func:`rank`.

.. cpp:function:: template<class iType> constexpr size_t stride(const iType& dim) const

   :tparam iType: an integral type
   :param dim: the dimension to get the stride of
   :return: the stride of dimension :cpp:any:`dim`

   Example: :cpp:expr:`a.stride(3) == (&a(i0,i1,i2,i3+1,i4)-&a(i0,i1,i2,i3,i4))`

   .. rubric:: Preconditions:

   - :cpp:any:`dim` must be smaller than :cpp:func:`rank`.

.. cpp:function:: constexpr size_t stride_0() const

   :return: the stride of dimension 0.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_1() const

   :return: the stride of dimension 1.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_2() const

   :return: the stride of dimension 2.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_3() const

   :return: the stride of dimension 3.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_4() const

   :return: the stride of dimension 4.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_5() const

   :return: the stride of dimension 5.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_6() const

   :return: the stride of dimension 6.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: constexpr size_t stride_7() const

   :return: the stride of dimension 7.

   .. deprecated:: 5.0
      Use :cpp:func:`View::stride` instead.

.. cpp:function:: template<class iType> void stride(iType* strides) const

   :tparam iType: an integral type
   :param strides: the output array of length :cpp:expr:`rank() + 1`

   Sets :cpp:expr:`strides[r]` to :cpp:expr:`stride(r)` for all :math:`r` with :math:`0 \le r \lt \texttt{rank()}`.
   Sets :cpp:expr:`strides[rank()]` to :cpp:func:`span()`.

   .. rubric:: Preconditions:

   - :cpp:any:`strides` must be an array of length :cpp:expr:`rank() + 1`

.. cpp:function:: constexpr size_t span() const

   :return: the size of the span of memory between the element with the lowest and highest address

   Obtains the memory span in elements between the element with the
   lowest and the highest address. This can be larger than the product
   of extents due to padding, and or non-contiguous data layout as for example :cpp:struct:`LayoutStride` allows.

.. cpp:function:: constexpr size_t size() const

   :return: the product of extents, i.e. the logical number of elements in the :cpp:class:`View`.

.. cpp:function:: constexpr pointer_type data() const

   :return: the pointer to the underlying data allocation.

   .. warning::
   
      Calling any function that manipulates the behavior of the memory (e.g. ``memAdvise``) on memory managed by Kokkos results in undefined behavior.

.. cpp:function:: bool span_is_contiguous() const

   :return: whether the span is contiguous (i.e. whether every memory location between in span belongs to the index space covered by the :cpp:class:`View`).

.. cpp:function:: static constexpr size_t required_allocation_size(size_t N0=0, size_t N1=0, \
         size_t N2=0, size_t N3=0, \
         size_t N4=0, size_t N5=0, \
         size_t N6=0, size_t N7=0);
   
   :param N0, N1, N2, N3, N4, N5, N6, N7: the dimensions to query
   :return: the number of bytes necessary for an unmanaged :cpp:class:`View` of the provided dimensions.

   .. rubric:: Requirements:
   
   - :cpp:expr:`array_layout::is_regular == true`.

.. cpp:function:: static constexpr size_t required_allocation_size(const array_layout& layout);

   :param layout: the layout to query
   :return: the number of bytes necessary for an unmanaged :cpp:class:`View` of the provided layout.

Other Utility Methods
^^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: template <class T> constexpr bool is_view_v

   Type trait, ``true`` if ``T`` is a ``View`` type.

.. cpp:function:: int use_count() const;

   :return: the current reference count of the underlying allocation.

.. cpp:function:: const std::string label() const;

   :return: the label of the View.

.. cpp:function:: void assign_data(pointer_type arg_data);

   :param arg_data: the pointer to set the underlying :cpp:class:`View` data pointer to

   Decrement reference count of previously assigned data and set the underlying pointer to arg_data.
   Note that the effective result of this operation is that the view is now an unmanaged view; thus, the deallocation of memory associated with arg_data is not linked in anyway to the deallocation of the view.

.. cpp:function:: constexpr bool is_allocated() const;

   :return: true if the view points to a valid memory location.

   This function works for both managed and unmanaged views.
   With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

Conversion to mdspan
^^^^^^^^^^^^^^^^^^^^

.. cpp:function:: template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class OtherAccessor> constexpr operator mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>()

   :tparam OtherElementType: the target mdspan element type
   :tparam OtherExtents: the target mdspan extents
   :tparam OtherLayoutPolicy: the target mdspan layout
   :tparam OtherAccessor: the target mdspan accessor

   :constraints: :cpp:class:`View`\ 's :ref:`natural mdspan <api-view-natural-mdspans>` must be assignable to :cpp:`mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>`

   :returns: an mdspan with extents and a layout converted from the :cpp:class:`View`'s *natural mdspan*.

.. cpp:function:: template <class OtherAccessorType = default_accessor<typename traits::value_type>> constexpr auto to_mdspan(const OtherAccessorType& other_accessor = OtherAccessorType{})

   :tparam OtherAccessor: the target mdspan accessor

   :constraints: :cpp:`typename OtherAccessorType::data_handle_type` must be assignable to :cpp:`value_type*`

   :returns: :cpp:class:`View`\ 's :ref:`natural mdspan <api-view-natural-mdspans>`, but with an accessor policy constructed from :cpp:any:`other_accessor`

.. cpp:namespace-pop::


Non-Member Functions
--------------------

.. cpp:function:: template <class... ViewTDst, class... ViewTSrc> bool is_assignable(const View<ViewTDst...>& dst, const View<ViewTSrc...>& src)

   :return: true if src can be assigned to dst.

   .. seealso:: :ref:`api-view-assignment`

.. cpp:function:: template <class LT, class... LP, class RT, class... RP> bool operator==(const View<LT, LP...>& lhs, const View<RT, RP...>& rhs)

   :return: :cpp:`true` if :cpp:type:`~View::value_type`, :cpp:type:`~View::array_layout`, :cpp:type:`~View::memory_space`, :cpp:func:`~View::rank()`, :cpp:func:`~View::data()` and :cpp:expr:`extent(r)`, for :math:`0 \le r \lt \texttt{rank()}`, match.

.. cpp:function:: template <class LT, class... LP, class RT, class... RP> bool operator!=(const View<LT, LP...>& lhs, const View<RT, RP...>& rhs)

   :return: :cpp:expr:`!(lhs == rhs)`

.. _api-view-assignment:

Assignment Rules
----------------

Assignment rules cover the assignment operator as well as copy constructors.
We aim at making all logically legal assignments possible, while intercepting illegal assignments if possible at compile time, otherwise at runtime.
In the following we use ``DstType`` and ``SrcType`` as the type of the destination view and source view respectively. 
``dst_view`` and ``src_view`` refer to the runtime instances of the destination and source views, i.e.:

.. code-block:: cpp

    SrcType src_view(...);
    DstType dst_view(src_view);
    dst_view = src_view;

The following conditions must be met at and are evaluated at compile time:

* :cpp:`DstType::rank() == SrcType::rank()`
* :cpp:`DstType::non_const_value_type` is the same as :cpp:`SrcType::non_const_value_type`
* If :cpp:`std::is_const_v<SrcType::value_type> == true` then :cpp:`std::is_const_v<DstType::value_type>` must also be :cpp:`true`.
* :cpp:`MemorySpaceAccess<DstType::memory_space,SrcType::memory_space>::assignable == true`
* If :cpp:`DstType::rank_dynamic() != DstType::rank()` and :cpp:`SrcType::rank_dynamic() != SrcType::rank()` then for each dimension :cpp:`k` that is compile time for both it must be true that :cpp:`dst_view.extent(k) == src_view.extent(k)`

Additionally the following conditions must be met at runtime:

* If :cpp:`DstType::rank_dynamic() != DstType::rank()` then for each compile time dimension :cpp:`k` it must be true that :cpp:`dst_view.extent(k) == src_view.extent(k)`.

Furthermore there are rules which must be met if :cpp:`DstType::array_layout` is not the same as :cpp:`SrcType::array_layout`.
These rules only cover cases where both layouts are one of :cpp:class:`LayoutLeft`, :cpp:class:`LayoutRight` or :cpp:class:`LayoutStride`

* If neither :cpp:`DstType::array_layout` nor :cpp:`SrcType::array_layout` is :cpp:class:`LayoutStride`:

  - If :cpp:`DstType::rank > 1` then :cpp:`DstType::array_layout` must be the same as :cpp:`SrcType::array_layout`.

* If either :cpp:`DstType::array_layout` or :cpp:`SrcType::array_layout` is :cpp:class:`LayoutStride`

  - For each dimension :cpp:`k` it must hold that :cpp:`dst_view.extent(k) == src_view.extent(k)`

.. code-block:: cpp
   :caption: Assignment Examples

    View<int*>       a1 = View<int*>("A1",N);     // OK
    View<int**>      a2 = View<int*[10]>("A2",N); // OK
    View<int*[10]>   a3 = View<int**>("A3",N,M);  // OK if M == 10 otherwise runtime failure
    View<const int*> a4 = a1;                     // OK
    View<int*>       a5 = a4;                     // Error: const to non-const assignment
    View<int**>      a6 = a1;                     // Error: Ranks do not match
    View<int*[8]>    a7 = a3;                     // Error: compile time dimensions do not match
    View<int[4][10]> a8 = a3;                     // OK if N == 4 otherwise runtime failure
    View<int*, LayoutLeft>    a9  = a1;           // OK since a1 is either LayoutLeft or LayoutRight
    View<int**, LayoutStride> a10 = a8;           // OK
    View<int**>               a11 = a10;          // OK
    View<int*, HostSpace> a12 = View<int*, CudaSpace>("A12",N); // Error: non-assignable memory spaces
    View<int*, HostSpace> a13 = View<int*, CudaHostPinnedSpace>("A13",N); // OK

.. _api-view-natural-mdspans:

Natural mdspans
---------------

.. versionadded:: 4.4.0

C++23 introduces `mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`_, a non-owning multidimensional array view.
:cpp:class:`View` is compatible with :cpp:`std::mdspan` and can be implicitly converted from and to valid mdspans.
These conversion rules are dictated by the *natural mdspan* of a view.
For an mdspan :cpp:`m` of type :cpp:`M` that is the *natural mdspan* of a :cpp:class:`View` :cpp:`v` of type :cpp:`V`, the following properties hold:

#. :cpp:`M::value_type` is :cpp:`V::value_type`
#. :cpp:`M::index_type` is :cpp:`std::size_t`.
#. :cpp:`M::extents_type` is :cpp:`std::extents<M::index_type, Extents...>` where

   * :cpp:`sizeof(Extents...)` is :cpp:`V::rank()`
   * and each element at index :cpp:`r` of :cpp:`Extents...` is :cpp:`V::static_extents(r)` if :cpp:`V::static_extents(r) != 0`, otherwise :cpp:`std::dynamic_extent`

#. :cpp:`M::layout_type` is

   * :cpp:`std::layout_left_padded<std::dynamic_extent>` if :cpp:`V::array_layout` is :cpp:`LayoutLeft`
   * :cpp:`std::layout_right_padded<std::dynamic_extent>` if :cpp:`V::array_layout` is :cpp:`LayoutRight`
   * :cpp:`std::layout_stride` if :cpp:`V::array_layout` is :cpp:any:`LayoutStride`

#. :cpp:`M::accessor_type` is :cpp:`std::default_accessor<V::value_type>`

Additionally, the *natural mdspan* is constructed so that :cpp:`m.data() == v.data()` and for each extent :cpp:`r`, :cpp:`m.extents().extent(r) == v.extent(r)`.

Examples
--------

.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    #include<cstdio>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc,argv);

        int N0 = atoi(argv[1]);
        int N1 = atoi(argv[2]);

        Kokkos::View<double*> a("A",N0);
        Kokkos::View<double*> b("B",N1);

        Kokkos::parallel_for("InitA", N0, KOKKOS_LAMBDA (const int& i) {
            a(i) = i;
        });

        Kokkos::parallel_for("InitB", N1, KOKKOS_LAMBDA (const int& i) {
            b(i) = i;
        });

        Kokkos::View<double**,Kokkos::LayoutLeft> c("C",N0,N1);
        {
            Kokkos::View<const double*> const_a(a);
            Kokkos::View<const double*> const_b(b);
            Kokkos::parallel_for("SetC", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{N0,N1}),
                KOKKOS_LAMBDA (const int& i0, const int& i1) {
                c(i0,i1) = a(i0) * b(i1);
            });
        }

        Kokkos::finalize();
    }
