/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_VIEWWITHPROPERTY_HPP
#define KOKKOS_VIEWWITHPROPERTY_HPP

namespace Kokkos {
namespace Impl {

template <typename DataType, typename ArrayLayout, typename DeviceType, typename MemoryTraits>
struct OrderedViewProperties
{
  using data_type = DataType;
  using device_type = DeviceType;
  using array_layout = ArrayLayout;
  using memory_traits = MemoryTraits;
};

template <typename OrderedProps>
using view_with_ordered_props_t = Kokkos::View<
  typename OrderedProps::DataType,
  typename OrderedProps::ArrayLayout,
  typename OrderedProps::DeviceType,
  typename OrderedProps::MemoryTraits>;

template <typename ViewType>
using ordered_props_of_view_t = OrderedViewProperties<
  typename ViewType::data_type,
  typename ViewType::array_layout,
  typename ViewType::device_type,
  typename ViewType::memory_traits>;

template <typename Property, typename OrderedPropsIn>
struct SetProperty;

template <typename DataType, typename OrderedPropsIn>
struct SetDataType
{
  using type = OrderedViewProperties<
    DataType,
    typename OrderedPropsIn::array_layout,
    typename OrderedPropsIn::device_type,
    typename OrderedPropsIn::memory_traits>;
};

template <typename ArrayLayout, typename OrderedPropsIn>
struct SetArrayLayout
{
  using type = OrderedViewProperties<
    typename OrderedPropsIn::data_type,
    ArrayLayout,
    typename OrderedPropsIn::device_type,
    typename OrderedPropsIn::memory_traits>;
};

template <typename DeviceType, typename OrderedPropsIn>
struct SetDeviceType
{
  using type = OrderedViewProperties<
    typename OrderedPropsIn::data_type,
    typename OrderedPropsIn::array_layout,
    DeviceType,
    typename OrderedPropsIn::memory_traits>;
};

template <typename MemoryTraits, typename OrderedPropsIn>
struct SetMemoryTraits;

template <unsigned flags1, unsigned flags2, typename DataType, typename ArrayLayout, typename DeviceType>
struct SetMemoryTraits<
  Kokkos::MemoryTraits<flags1>,
  OrderedViewProperties<DataType, ArrayLayout, DeviceType, Kokkos::MemoryTraits<flags2> > >
{
  using type = OrderedViewProperties<
    DataType,
    ArrayLayout,
    DeviceType,
    Kokkos::MemoryTraits<flags1 | flags2> >;
};

template <typename Property, typename OrderedPropsIn>
struct SetProperty {
  using type =
    typename std::conditional< Kokkos::Impl::is_space<Property>::value, SetDeviceType<Property, OrderedPropsIn>
  , typename std::conditional< Kokkos::Impl::is_array_layout<Property>::value, SetArrayLayout<Property, OrderedPropsIn>
  , typename std::conditional< Kokkos::Impl::is_memory_traits<Property>::value, SetMemoryTraits<Property, OrderedPropsIn>
  , SetDataType<Property, OrderedPropsIn>
  >::type >::type >::type::type;
};

template <typename Property, typename OrderedPropsIn>
using set_prop_t = typename SetProperty<Property, OrderedPropsIn>::type;

template <typename Property, typename ViewType>
using with_property_t = view_with_ordered_props_t<set_prop_t<Property, ordered_props_of_view_t<ViewType>>>;

}
}

namespace Kokkos {
template <typename Property, typename ViewType>
Kokkos::Impl::with_property_t<Property, ViewType> with_property(ViewType const& in) {
  return in;
}
}

#endif
