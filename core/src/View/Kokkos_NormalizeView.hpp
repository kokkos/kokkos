/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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

#ifndef KOKKOS_KOKKOS_NORMALIZEVIEW_HPP
#define KOKKOS_KOKKOS_NORMALIZEVIEW_HPP

#include <Kokkos_Core_fwd.hpp>

#include <View/Kokkos_ExtractExtents.hpp>
#include <Kokkos_Concepts.hpp>        // is_execution_space, is_memory_space
#include <Kokkos_Layout.hpp>          // LayoutRight
#include <impl/Kokkos_Utilities.hpp>  // type_list
#include <Kokkos_MemoryTraits.hpp>    // DefaultMemoryTraits

namespace Kokkos {

//==============================================================================
// <editor-fold desc="View parameter normalization"> {{{1

namespace Impl {

/**
 * Valid ways in which template arguments may be specified:
 *   - View< DataType >
 *   - View< DataType , Layout >
 *   - View< DataType , Layout , Space >
 *   - View< DataType , Layout , Space , MemoryTraits >
 *   - View< DataType , Space >
 *   - View< DataType , Space , MemoryTraits >
 *   - View< DataType , MemoryTraits >
 */

//------------------------------------------------------------------------------
// <editor-fold desc="NormalizeDevice"> {{{2

template <class Space, class Enable = void>
struct NormalizeDevice;

template <class ExecSpace>
struct NormalizeDevice<ExecSpace,
                       std::enable_if_t<is_execution_space<ExecSpace>::value>>
    : NormalizeDevice<Device<ExecSpace, typename ExecSpace::memory_space>> {};

template <class MemSpace>
struct NormalizeDevice<MemSpace,
                       std::enable_if_t<is_memory_space<MemSpace>::value>>
    : NormalizeDevice<Device<typename MemSpace::execution_space, MemSpace>> {};

template <class ExecSpace, class MemSpace>
struct NormalizeDevice<Device<ExecSpace, MemSpace>> {
  using type = Device<ExecSpace, MemSpace>;
};

// </editor-fold> end NormalizeDevice }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="NormalizeLayout"> {{{2

template <class Layout, int Rank>
struct NormalizeLayout : identity<Layout> {};

// LayoutRight for rank 1 becomes LayoutLeft
template <>
struct NormalizeLayout<LayoutRight, 1> : identity<LayoutLeft> {};

// Any layout for rank 0 becomes LayoutLeft
template <class Layout>
struct NormalizeLayout<Layout, 0> : identity<LayoutLeft> {};

// </editor-fold> end NormalizeLayout }}}2
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// <editor-fold desc="NormalizeViewProperties"> {{{2

template <class DataType, class PropertiesList, class Enable = void>
struct NormalizeViewProperties;

///  View< DataType >
template <class DataType>
struct NormalizeViewProperties<DataType, type_list<>>
    : NormalizeViewProperties<DataType,
                              type_list<Kokkos::DefaultExecutionSpace>> {
  static_assert(is_space<Kokkos::DefaultExecutionSpace>::value,
                "Kokkos include order has led to incomplete default execution "
                "space instance. Contact a developer.");
};

///  View< DataType, Space >
template <class DataType, class Space>
struct NormalizeViewProperties<
    DataType, type_list<Space>,
    std::enable_if_t<is_space<Space>::value || is_device<Space>::value>>
    : NormalizeViewProperties<
          DataType,
          type_list<typename Space::execution_space::array_layout, Space>> {};

///  View< DataType, MemoryTraits >
template <class DataType, class MemoryTraits>
struct NormalizeViewProperties<
    DataType, type_list<MemoryTraits>,
    std::enable_if_t<is_memory_traits<MemoryTraits>::value>>
    : NormalizeViewProperties<DataType,
                              type_list<DefaultExecutionSpace, MemoryTraits>> {
};

///  View< DataType, Space, MemoryTraits >
template <class DataType, class Space, class MemTraits>
struct NormalizeViewProperties<
    DataType, type_list<Space, MemTraits>,
    std::enable_if_t<is_space<Space>::value || is_device<Space>::value>>
    : NormalizeViewProperties<
          DataType, type_list<typename Space::execution_space::array_layout,
                              Space, MemTraits>> {};

/// View< DataType , Layout >
template <class DataType, class Layout>
struct NormalizeViewProperties<DataType, type_list<Layout>,
                               std::enable_if_t<is_array_layout<Layout>::value>>
    : NormalizeViewProperties<
          DataType, type_list<Layout, Kokkos::DefaultExecutionSpace>> {};

/// View< DataType , Layout , Space >
template <class DataType, class Layout, class Space>
struct NormalizeViewProperties<DataType, type_list<Layout, Space>,
                               std::enable_if_t<is_array_layout<Layout>::value>>
    : NormalizeViewProperties<
          DataType, type_list<Layout, Space, Kokkos::DefaultMemoryTraits>> {
  // This is unambigous, so use static_assert instead of SFINAE
  static_assert(is_space<Space>::value || is_device<Space>::value,
                "Third template parameter to Kokkos View must be a space or a "
                "device when "
                " the second parameter is a Layout");
};

template <class DataType>
struct NormalizeViewProperties<DataType, type_list<void, void, void>,void>
  : NormalizeViewProperties<DataType, type_list<Kokkos::DefaultExecutionSpace>> {};

template <class DataType, class T1>
struct NormalizeViewProperties<DataType, type_list<T1, void, void>,void>
  : NormalizeViewProperties<DataType, type_list<T1>> {};

template <class DataType, class T1, class T2>
struct NormalizeViewProperties<DataType, type_list<T1, T2, void>,void>
  : NormalizeViewProperties<DataType, type_list<T1,T2>> {};

/// View< DataType , Layout , Space, MemoryTraits >
template <class DataType, class Layout, class Space, class MemTraits>
struct NormalizeViewProperties<DataType, type_list<Layout, Space, MemTraits>> {
  /*
  // This is unambigous, so use static_assert instead of SFINAE
  static_assert(is_array_layout<Layout>::value,
                "The second template parameter to Kokkos::View must be an "
                "array layout when four paramters are given");
  static_assert(is_space<Space>::value || is_device<Space>::value,
                "Third template parameter to Kokkos::View must be a space or a "
                "device when four paramters are given");
  static_assert(is_memory_traits<MemTraits>::value,
                "The fourth template parameter to Kokkos::View must be memory "
                "traits when four paramters are given");
  */

 private:
  static constexpr auto rank = ExtractExtents<DataType>::extents_type::rank();

  using normalized_layout = typename NormalizeLayout<Layout, rank>::type;
  using normalized_device = typename NormalizeDevice<Space>::type;

 public:
  using type = Kokkos::BasicView<DataType, normalized_layout, normalized_device,
                                 MemTraits>;
};

// </editor-fold> end NormalizeViewProperties }}}2
//------------------------------------------------------------------------------

}  // end namespace Impl

// </editor-fold> end View parameter normalization }}}1
//==============================================================================

}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_NORMALIZEVIEW_HPP
