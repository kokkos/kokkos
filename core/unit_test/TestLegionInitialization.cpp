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

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace {
// The purpose of this test is to mimic Legion's use case of initializing and
// finalizing individual backends
TEST(initialization, legion_initialization) {
  Kokkos::InitializationSettings kokkos_init_settings;
  Kokkos::Impl::pre_initialize(kokkos_init_settings);
#ifdef KOKKOS_ENABLE_HPX
  EXPECT_FALSE(Kokkos::Experimental::HPX::impl_is_initialized());
  Kokkos::Experimental::HPX::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Experimental::HPX::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_THREADS
  EXPECT_FALSE(Kokkos::Threads::impl_is_initialized());
  Kokkos::Threads::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Threads::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  EXPECT_FALSE(Kokkos::OpenMP::impl_is_initialized());
  Kokkos::OpenMP::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::OpenMP::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  EXPECT_FALSE(Kokkos::Serial::impl_is_initialized());
  Kokkos::Serial::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Serial::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_SYCL
  EXPECT_FALSE(Kokkos::Experimental::SYCL::impl_is_initialized());
  Kokkos::Experimental::SYCL::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Experimental::SYCL::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  EXPECT_FALSE(Kokkos::Experimental::OpenMPTarget::impl_is_initialized());
  Kokkos::Experimental::OpenMPTarget::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Experimental::OpenMPTarget::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_HIP
  EXPECT_FALSE(Kokkos::HIP::impl_is_initialized());
  Kokkos::HIP::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::HIP::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_CUDA
  EXPECT_FALSE(Kokkos::Cuda::impl_is_initialized());
  Kokkos::Cuda::impl_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::Cuda::impl_is_initialized());
#endif
  EXPECT_FALSE(Kokkos::is_initialized());
  Kokkos::Impl::post_initialize(kokkos_init_settings);
  EXPECT_TRUE(Kokkos::is_initialized());

#ifdef KOKKOS_ENABLE_HPX
  Kokkos::Experimental::HPX::impl_finalize();
  EXPECT_FALSE(Kokkos::Experimental::HPX::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_THREADS
  Kokkos::Threads::impl_finalize();
  EXPECT_FALSE(Kokkos::Threads::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  Kokkos::OpenMP::impl_finalize();
  EXPECT_FALSE(Kokkos::OpenMP::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_SERIAL
  Kokkos::Serial::impl_finalize();
  EXPECT_FALSE(Kokkos::Serial::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_SYCL
  Kokkos::Experimental::SYCL::impl_finalize();
  EXPECT_FALSE(Kokkos::Experimental::SYCL::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
  Kokkos::Experimental::OpenMPTarget::impl_finalize();
  EXPECT_FALSE(Kokkos::Experimental::OpenMPTarget::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_HIP
  Kokkos::HIP::impl_finalize();
  EXPECT_FALSE(Kokkos::HIP::impl_is_initialized());
#endif
#ifdef KOKKOS_ENABLE_CUDA
  Kokkos::Cuda::impl_finalize();
  EXPECT_FALSE(Kokkos::Cuda::impl_is_initialized());
#endif
  Kokkos::finalize();
  EXPECT_FALSE(Kokkos::is_initialized());
}
}  // namespace

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
