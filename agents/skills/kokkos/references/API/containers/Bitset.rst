
.. role:: cpp(code)
   :language: cpp

``Bitset``
==========

Header file: ``<Kokkos_Bitset.hpp>``

Class Interface
---------------

.. cpp:class:: template <typename Device> Bitset

  :cpp:`Kokkos::Bitset` represents a thread safe view to a fixed-size (at run-time) sequence of N bits.

  :tparam Device: Device that physically contains the bits.

  .. rubric:: Static Constants

  .. cpp:member:: static constexpr unsigned BIT_SCAN_REVERSE = 1u

    :cpp:`BIT_SCAN_REVERSE` : Bit mask for scanning direction

  .. cpp:member:: static constexpr unsigned MOVE_HINT_BACKWARD = 2u

    :cpp:`MOVE_HINT_BACKWARD` : Bit mask for hint direction

  .. cpp:member:: static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_FORWARD = 0u

    :cpp:`BIT_SCAN_FORWARD_MOVE_HINT_FORWARD` : When passed as :cpp:`scan_direction` to :cpp:`find_any_set_near(...)` or :cpp:`find_any_reset_near(...)`, scans for the bit in the forward (increasing index) direction. If the bit was not found, selects a new hint past the current hint.

  .. cpp:member:: static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_FORWARD = BIT_SCAN_REVERSE

    :cpp:`BIT_SCAN_REVERSE_MOVE_HINT_FORWARD`: When passed as :cpp:`scan_direction` to :cpp:`find_any_set_near(...)` or :cpp:`find_any_reset_near(...)`, scans for the bit in the reverse (decreasing index) direction. If the bit was not found, selects a new hint past the current hint.

  .. cpp:member:: static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD = MOVE_HINT_BACKWARD

    :cpp:`BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD`: When passed as :cpp:`scan_direction` to :cpp:`find_any_set_near(...)` or :cpp:`find_any_reset_near(...)`, scans for the bit in the forward (increasing index) direction. If the bit was not found, selects a new hint before the current hint.

  .. cpp:member:: static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD = BIT_SCAN_REVERSE | MOVE_HINT_BACKWARD

    :cpp:`BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD`: When passed as :cpp:`scan_direction` to :cpp:`find_any_set_near(...)` or :cpp:`find_any_reset_near(...)`, scans for the bit in the reverse (decreasing index) direction. If the bit was not found, selects a new hint before the current hint.

  .. rubric:: Constructors

  .. cpp:function:: Bitset(unsigned arg_size = 0u)

    Host/Device: Construct a bitset with :cpp:`arg_size` bits.

  .. rubric:: Data Access Functions

  .. cpp:function:: unsigned size() const

    Host/Device: return the number of bits.

  .. cpp:function:: unsigned count() const

    Host: return the number of bits which are set to ``1``.

  .. cpp:function:: void set()

    Host: set all the bits to ``1``.

  .. cpp:function:: void reset();
  .. cpp:function:: void clear();

    Host/Device: set all the bits to ``0``.

  .. cpp:function:: void set(unsigned i)

    Device: set the ``i``\ 'th bit to ``1``.

  .. cpp:function:: void reset(unsigned i)

    Device: set the ``i``\ 'th bit to ``0``.

  .. cpp:function:: bool test(unsigned i) const

    Device: return :cpp:`true` if and only if the ``i``\ 'th bit is set to ``1``.

  .. cpp:function:: unsigned max_hint() const

    Host/Device: used with :cpp:`find_any_set_near(...)` & :cpp:`find_any_reset_near(...)` functions.

    Returns the max number of times those functions should be call when searching for an available bit.

  .. cpp:function:: Kokkos::pair<bool, unsigned> find_any_set_near(unsigned hint, unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const

    Host/Device: starting at the :cpp:`hint` position, find the first bit set to ``1``.

    Returns a :cpp:`pair<bool, unsigned>`.

    When :cpp:`result.first` is :cpp:`true` then :cpp:`result.second` is the bit position found.

    When :cpp:`result.first` is :cpp:`false` then :cpp:`result.second` is a new hint position.

    If :cpp:`scan_direction & BIT_SCAN_REVERSE`\ , then the scanning for the bit happens in decreasing index order;
    otherwise, it happens in increasing index order.

    If :cpp:`scan_direction & MOVE_HINT_BACKWARDS`\ , then the new hint position occurs at a smaller index than :cpp:`hint`\ ;
    otherwise, it occurs at a larger index than :cpp:`hint`.

  .. cpp:function:: Kokkos::pair<bool, unsigned> find_any_unset_near(unsigned hint, unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const;

    Host/Device: starting at the :cpp:`hint` position, find the first bit set to ``0``.

    Returns a :cpp:`pair<bool, unsigned>`.

    When :cpp:`result.first` is :cpp:`true` then :cpp:`result.second` is the bit position found.

    When :cpp:`result.first` is :cpp:`false` then :cpp:`result.second` is a new hint position.

    If :cpp:`scan_direction & BIT_SCAN_REVERSE`\ , then the scanning for the bit happens in decreasing index order; otherwise, it happens in increasing index order.

    If :cpp:`scan_direction & MOVE_HINT_BACKWARDS`\ , then the new hint position occurs at a smaller index than :cpp:`hint`\ ; otherwise, it occurs at a larger index than :cpp:`hint`.

  .. cpp:function:: constexpr bool is_allocated() const

    Host/Device: the bits are allocated on the device.

``ConstBitset``
===============

Class Interface
---------------

.. cpp:class:: template <typename Device> ConstBitset

  :tparam Device: Device that physically contains the bits.

  .. rubric:: Constructors / assignment

  .. cpp:function:: ConstBitset()

    Host/Device: Construct a bitset with no bits.

  .. cpp:function:: ConstBitset(ConstBitset const& rhs) = default
  .. cpp:function:: ConstBitset& operator=(ConstBitset const& rhs) = default

    Copy constructor/assignment operator.

  .. cpp:function:: ConstBitset(Bitset<Device> const& rhs)
  .. cpp:function:: ConstBitset& operator=(Bitset<Device> const& rhs)

    Host/Device: Copy/assign a :cpp:`Bitset` to a :cpp:`ConstBitset`.

  .. cpp:function:: unsigned size() const

    Host/Device: return the number of bits.

  .. cpp:function:: unsigned count() const

     Host/Device: return the number of bits which are set to ``1``.

  .. cpp:function:: bool test(unsigned i) const

    Host/Device: Return ``true`` if and only if the ``i``\ 'th bit set to ``1``.

Non-Member Functions
--------------------

  .. cpp:function:: template <typename DstDevice, typename SrcDevice> void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src)

    Copy a ``Bitset`` from ``src`` on ``SrcDevice`` to ``dst`` on ``DstDevice``.

  .. cpp:function:: template <typename DstDevice, typename SrcDevice> void deep_copy(Bitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src)

    Copy a ``ConstBitset`` from ``src`` on ``SrcDevice`` to a ``Bitset`` ``dst`` on ``DstDevice``.
