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

#ifndef KOKKOS_SYCL_KERNELLAUNCH_HPP_
#define KOKKOS_SYCL_KERNELLAUNCH_HPP_

#include <SYCL/Kokkos_SYCL_Instance.hpp>

/*--------------------------------------------------------------------------*/
// Temp place for Trivially copyable check - velesko
template <class Obj>
constexpr void isTriviallyCopyable() {
  static_assert(std::is_trivially_copyable<Obj>::value, "");

  static_assert(std::is_copy_constructible<Obj>::value ||
                    std::is_move_constructible<Obj>::value ||
                    std::is_copy_assignable<Obj>::value ||
                    std::is_move_assignable<Obj>::value,
                "Obj copy/move constructors/assignments deleted");

  static_assert(std::is_trivially_copy_constructible<Obj>::value ||
                    !std::is_copy_constructible<Obj>::value,
                "Obj not trivially copy constructible");

  static_assert(std::is_trivially_move_constructible<Obj>::value ||
                    !std::is_move_constructible<Obj>::value,
                "Obj not trivially move constructible");

  static_assert(std::is_trivially_copy_assignable<Obj>::value ||
                    !std::is_copy_assignable<Obj>::value,
                "Obj not trivially copy assignable");

  static_assert(std::is_trivially_move_assignable<Obj>::value ||
                    !std::is_move_assignable<Obj>::value,
                "Obj not trivially move assignable");

  static_assert(std::is_trivially_destructible<Obj>::value,
                "Obj not trivially destructible");
}

template <class T>
class kokkos_sycl_functor;

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <class Driver>
void sycl_launch_bind(Driver tmp, cl::sycl::handler& cgh) {
  cgh.parallel_for(
      cl::sycl::range<1>(tmp.m_policy.end() - tmp.m_policy.begin()), tmp);
}

template <class Driver>
void sycl_launch(const Driver driver) {
  isTriviallyCopyable<Driver>();
  isTriviallyCopyable<decltype(driver.m_functor)>();
  driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
  driver.m_policy.space().impl_internal_space_instance()->m_queue->submit(
      [&](cl::sycl::handler& cgh) {
        cgh.parallel_for(
            cl::sycl::range<1>(driver.m_policy.end() - driver.m_policy.begin()),
            [=](cl::sycl::item<1> item) {
              int id = item.get_linear_id();
              driver.m_functor(id);
            });
      });
  driver.m_policy.space().impl_internal_space_instance()->m_queue->wait();
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_SYCL_KERNELLAUNCH_HPP_
