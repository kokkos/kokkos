
``DualView``
============

Header file: ``<Kokkos_DualView.hpp>``

|

Container to manage mirroring a ``Kokkos::View`` that references device memory with
a ``Kokkos::View`` that is host-accessible. The class provides capabilities to manage
data which exists in two different memory spaces at the same time. It supports Views with
the same layout on two memory spaces as well as modified flags for both allocations.
Users are responsible for updating the modified flags manually if they change the data in either memory space, by calling the ``modify()`` function, which is templated on the device with the modified data.
Users may also synchronize data by calling the ``sync()`` method, which is templated on the device that requires synchronization (i.e., the target of the one-way copy operation).

The DualView class also provides convenience methods such as realloc, resize and capacity
which call the appropriate methods of the underlying `Kokkos::View <../core/view/view.rst>`_ objects.

The four template arguments are the same as those of ``Kokkos::View``.

* DataType, The type of the entries stored in the container.

* Layout, The array's layout in memory.

* Device, The Kokkos Device type. If its memory space is not host-accessible,
  then DualView will contain two separate Views: one in device memory,
  and one in host memory. Otherwise, DualView will only store one View.

* MemoryTraits (optional) The user's intended memory access behavior. Please see the documentation
  of `Kokkos::View <../core/view/view.rst>`_ for examples. The default suffices for most users.

Usage
-----

.. code-block:: cpp

    using view_type = Kokkos::DualView<Scalar**,
                                       Kokkos::LayoutLeft,
                                       Device>
    view_type a("A", n, m);

    Kokkos::deep_copy(a.view_device(), 1); // set device-side entries to 1
    a.template modify<typename view_type::execution_space>(); // mark device-side as modified
    a.template sync<typename view_type::host_mirror_space>(); // sync modified data to host

    Kokkos::deep_copy(a.view_host(), 2); // set host-side entries to 2
    a.template modify<typename ViewType::host_mirror_space>(); // mark host-side as modified
    a.template sync<typename ViewType::execution_space>(); // sync modified data to device

Description
-----------



.. cpp:class:: template <class DataType, class Arg1Type = void, class Arg2Type = void, class Arg3Type = void> DualView

    |

    .. rubric:: *Public* typedefs

    .. cpp:type:: ViewTraits<DataType, Arg1Type, Arg2Type, Arg3Type> traits

       Typedefs for device types and various ``Kokkos::View`` specializations.

    .. cpp:type:: traits::host_mirror_space host_mirror_space

       The Kokkos Host Device type

    .. cpp:type:: View<typename traits::data_type, Arg1Type, Arg2Type, Arg3Type> t_dev

       The type of a ``Kokkos::View`` on the device.

    .. cpp:type:: typename t_dev::host_mirror_type t_host

       The type of a ``Kokkos::View`` host mirror of ``t_dev``.

    .. cpp:type:: View<typename traits::const_data_type, Arg1Type, Arg2Type, Arg3Type> t_dev_const

       The type of a const View on the device.

    .. cpp:type:: typename t_dev_const::host_mirror_type t_host_const

       The type of a const View host mirror of ``t_dev_const``.

    .. cpp:type:: View<typename traits::const_data_type, typename traits::array_layout, typename traits::device_type, Kokkos::MemoryTraits<Kokkos::RandomAccess> > t_dev_const_randomread

       The type of a const, random-access View on the device.

    .. cpp:type:: typename t_dev_const_randomread::host_mirror_type t_host_const_randomread

       The type of a const, random-access View host mirror of ``t_dev_const_randomread``.

    .. cpp:type:: View<typename traits::data_type, typename traits::array_layout, typename traits::device_type, MemoryUnmanaged> t_dev_um

       The type of an unmanaged View on the device.

    .. cpp:type:: View<typename t_host::data_type, typename t_host::array_layout, typename t_host::device_type, MemoryUnmanaged> t_host_um

       The type of an unmanaged View host mirror of \\c t_dev_um.

    .. cpp:type:: View<typename traits::const_data_type, typename traits::array_layout, typename traits::device_type, MemoryUnmanaged> t_dev_const_um

       The type of a const unmanaged View on the device.

    .. cpp:type:: View<typename t_host::const_data_type, typename t_host::array_layout, typename t_host::device_type, MemoryUnmanaged> t_host_const_um

       The type of a const unmanaged View host mirror of \\c t_dev_const_um.

    .. cpp:type:: View<typename t_host::const_data_type, typename t_host::array_layout, typename t_host::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> > t_dev_const_randomread_um

       The type of a const, random-access View on the device.

    .. cpp:type:: typename t_dev_const_randomread::host_mirror_type t_host_const_randomread_um

       The type of a const, random-access View host mirror of ``t_dev_const_randomread``.

    .. cpp:type:: View<unsigned int[2], LayoutLeft, typename t_host::execution_space> t_modified_flags;

    .. cpp:type:: View<unsigned int, LayoutLeft, typename t_host::execution_space> t_modified_flag;

    .. rubric:: Data Members

    .. cpp:member:: t_dev d_view

       The view instance on the *device*, public access deprecated from Kokkos 4.6 on.

    .. cpp:member:: t_host h_view

       The view instance on the *host*, public access deprecated from Kokkos 4.6 on.

    .. cpp:member:: t_modified_flags modified_flags

    .. cpp:member:: t_modified_flag modified_host;

    .. cpp:member:: t_modified_flag modified_device;

    |

    .. rubric:: *Public* constructors

    .. cpp:function:: DualView();

       Empty constructor. Both device and host View objects are constructed using their default constructors.
       The "modified" flags are both initialized to "unmodified."

    .. cpp:function:: DualView(const std::string& label, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

       Constructor that allocates View objects on both host and device.
       The first argument is a string label, which is entirely for your benefit. (Different DualView objects may have the same label if you like.)
       The arguments that follow are the dimensions of the View objects. For example, if the View has three dimensions,
       the first three integer arguments will be nonzero, and you may omit the integer arguments that follow.

    .. cpp:function:: DualView(ALLOC_PROP const& arg_prop, const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

       Constructor that allocates View objects on both host and device allowing to pass an object created by ``Kokkos::view_alloc`` as first argument,
       e.g., to provide a label, avoid initialization, or specifying an execution space instance.
       The arguments that follow are the dimensions of the View objects.
       For example, if the View has three dimensions, the first three integer arguments will be nonzero, and you may omit the integer arguments that follow.

    .. cpp:function:: DualView(const DualView<SS, LS, DS, MS>& src);

       Copy constructor (shallow copy)

    .. cpp:function:: DualView(const DualView<SD, S1, S2, S3>& src, const Arg0& arg0, Args... args);

       Subview constructor

    .. cpp:function:: DualView(const t_dev& d_view_, const t_host& h_view_);

       Create DualView from existing device and host View objects.
       This constructor assumes that the device and host View objects are synchronized. You, the caller, are responsible for making sure this
       is the case before calling this constructor. After this constructor returns, you may use DualView's ``sync()`` and ``modify()``
       methods to ensure synchronization of the View objects. In case the DualView only stores one View, i.e., DualView's memory space is host-accessible,
       both arguments must reference the same allocation.

       - ``d_view_`` Device View

       - ``h_view_`` Host View (must have type ``t_host = t_dev::host_mirror_type``)

    |

    .. rubric:: *Public* Methods for synchronizing, marking as modified, and getting Views.

    .. cpp:function:: template <class Device> KOKKOS_INLINE_FUNCTION const auto& view();

    .. cpp:function:: template <class Device> static int get_device_side();

       * Return a View on a specific device ``Device``. ``Device`` can be a ``Kokkos::Device`` type, a memory space or a execution space corresponding to either the device View or the host-accessible View.
       * For example, suppose you create a DualView on Cuda, like this:

         .. code-block:: cpp

           using dual_view_type = Kokkos::DualView<float, Kokkos::Cuda>;
           dual_view_type DV ("my dual view", 100);

         If you want to get the CUDA device View, do this:

         .. code-block:: cpp

           dual_view_type::t_dev cudaView = DV.view<dual_view_type::t_dev::memory_space>();

         and if you want to get the host mirror of that View, do this:

         .. code-block:: cpp

           dual_view_type::t_host hostView = DV.view<dual_view_type::t_host::memory_space>();

    .. cpp:function:: const t_host& view_host() const;

       *  Return the host-accessible View. Returns the View by value with `Kokkos_ENABLE_DEPRECATED_CODE_4=ON`

    .. cpp:function:: const t_dev& view_device() const;

       * Return the View on the device. Returns the View by value with `Kokkos_ENABLE_DEPRECATED_CODE_4=ON`.

    .. cpp:function:: template <class Device> void sync(const typename Impl::enable_if<(std::is_same<typename traits::data_type, typename traits::non_const_data_type>::value) || (std::is_same<Device, int>::value), int>::type& = 0);

    .. cpp:function:: template <class Device> void sync(const typename Impl::enable_if<(!std::is_same<typename traits::data_type, typename traits::non_const_data_type>::value) || (std::is_same<Device, int>::value), int>::type& = 0);

       * Update data on device or host only if data in the other space has been marked as modified.
       * If ``Device`` is the same as this DualView's device type, then copy data from host to device. Otherwise, copy data from device to host. In either case, only copy if the source of the copy has been modified.
       * This is a one-way synchronization only. If the target of the copy has been modified, this operation will discard those modifications. It will also reset both device and host modified flags.
       * This method doesn't know on its own whether you modified the data in either View. You must manually mark modified data as modified, by calling the ``modify()`` method with the appropriate template parameter.

    .. cpp:function:: template <class Device> bool need_sync() const;

    .. cpp:function:: template <class Device> void modify();

    .. cpp:function:: inline void clear_sync_state();

       Mark data as modified on the given device \\c Device. If ``Device`` is the same as this
       DualView's device type, then mark the device's data as modified. Otherwise, mark the host's data as modified.

    |

    .. rubric:: *Public* Methods for reallocating or resizing the View objects

    .. cpp:function:: constexpr bool is_allocated() const;

       Return allocation state of underlying views. Returns true if both the host and device
       views points to a valid memory location. This function works for both managed and unmanaged views.
       With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

    .. cpp:function:: void realloc(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

       Reallocate both View objects. This discards any existing contents of the objects,
       and resets their modified flags. It does *not* copy the old contents of either View into the new View objects.

    .. cpp:function:: void resize(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG, const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

       Resize both views, copying old contents into new if necessary. This method only copies the old
       contents into the new View objects for the device which was last marked as modified. Thus, users are required to call ``sync()`` before using the resized object.

    |

    .. rubric:: *Public* Methods for querying capacity, stride, or dimension(s).

    .. cpp:function:: KOKKOS_INLINE_FUNCTION constexpr size_t span() const;

       Return the allocation size (same as ``Kokkos::View::span``).

    .. cpp:function:: KOKKOS_INLINE_FUNCTION bool span_is_contiguous();

       Return true if the span is contiguous

    .. cpp:function:: template <typename iType> void stride(iType* stride_) const;

       Get stride(s) for each dimension. Sets ``stride_`` [rank] to span().

    .. cpp:function:: template <typename iType> KOKKOS_INLINE_FUNCTION constexpr typename std::enable_if<std::is_integral<iType>::value, size_t>::type extent(const iType& r) const;

       Return the extent for the requested rank

    .. cpp:function:: template <typename iType> KOKKOS_INLINE_FUNCTION constexpr typename std::enable_if<std::is_integral<iType>::value, int>::type extent_int(const iType& r) const;

       Return integral extent for the requested rank
