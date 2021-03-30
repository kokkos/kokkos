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

#ifndef KOKKOS_KOKKOS_VIEWALLOCATION_HPP
#define KOKKOS_KOKKOS_VIEWALLOCATION_HPP

#include <impl/Kokkos_SharedAlloc.hpp>
#include <View/Kokkos_ViewValueFunctor.hpp>

namespace Kokkos {
namespace Impl {

// This code only depends on execution space, memory space, and value type,
// so we can save a lot of compilation time by only depending on those three
// template parameters.
// Naming note: I intentionally avoided the term "allocator" for now
template <class ValueType>
struct ViewAllocationMechanism {
  template <class ExecutionSpace, class MemorySpace>
  static auto* allocate_shared(ExecutionSpace const& ex, MemorySpace const& mem,
                               std::string const& arg_label,
                               std::size_t arg_size, bool initialize) {
    using functor_type = ViewValueFunctor<ExecutionSpace, ValueType>;
    using record_type  = SharedAllocationRecord<MemorySpace, functor_type>;

    // TODO @mdspan pad for alignment
    auto alloc_size = arg_size;

    auto* record = record_type::allocate(mem, arg_label, alloc_size);
    if (arg_size > 0 && initialize) {
      record->m_destroy = functor_type(static_cast<ValueType*>(record->data()),
                                       arg_size, arg_label);
      // The construction code is, counterintuitively, on the deleter...
      record->m_destroy.construct_shared_allocation();
    }

    return record;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_KOKKOS_VIEWALLOCATION_HPP
