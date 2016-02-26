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
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_BASIC_ALLOCATORS_HPP
#define KOKKOS_BASIC_ALLOCATORS_HPP

#if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW )

namespace Kokkos { namespace Impl {

/// class UnmanagedAllocator
/// does nothing when deallocate(ptr,size) is called
class UnmanagedAllocator
{
public:
  static const char * name() { return "Unmanaged Allocator"; }

  static void deallocate(void * /*ptr*/, size_t /*size*/) {}
};


/// class MallocAllocator
class MallocAllocator
{
public:
  static const char * name()
  {
    return "Malloc Allocator";
  }

  static void* allocate(size_t size);

  static void deallocate(void * ptr, size_t size);

  static void * reallocate(void * old_ptr, size_t old_size, size_t new_size);
};


/// class AlignedAllocator
/// memory aligned to Kokkos::Impl::MEMORY_ALIGNMENT
class AlignedAllocator
{
public:
  static const char * name()
  {
    return "Aligned Allocator";
  }

  static void* allocate(size_t size);

  static void deallocate(void * ptr, size_t size);

  static void * reallocate(void * old_ptr, size_t old_size, size_t new_size);
};


/// class PageAlignedAllocator
/// memory aligned to PAGE_SIZE
class PageAlignedAllocator
{
public:
  static const char * name()
  {
    return "Page Aligned Allocator";
  }

  static void* allocate(size_t size);

  static void deallocate(void * ptr, size_t size);

  static void * reallocate(void * old_ptr, size_t old_size, size_t new_size);
};


}} // namespace Kokkos::Impl

#endif /* #if ! defined( KOKKOS_USING_EXPERIMENTAL_VIEW ) */

#endif //KOKKOS_BASIC_ALLOCATORS_HPP


