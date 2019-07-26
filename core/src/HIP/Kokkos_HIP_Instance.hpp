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

/*--------------------------------------------------------------------------*/

#ifndef KOKKOS_HIP_INSTANCE_HPP_
#define KOKKOS_HIP_INSTANCE_HPP_
#include <Kokkos_HIP.hpp>

namespace Kokkos {
namespace Experimental {
namespace Impl {

//----------------------------------------------------------------------------

class HIPInternal {
 private:
  HIPInternal(const HIPInternal &);
  HIPInternal &operator=(const HIPInternal &);

 public:
  using size_type = ::Kokkos::Experimental::HIP::size_type;

  int m_hipDev;
  int m_hipArch;
  unsigned m_multiProcCount;
  unsigned m_maxWorkgroup;
  unsigned m_maxSharedWords;
  size_type m_scratchSpaceCount;
  size_type m_scratchFlagsCount;
  size_type *m_scratchSpace;
  size_type *m_scratchFlags;

  hipStream_t m_stream;

  static int was_finalized;

  static HIPInternal &singleton();

  int verify_is_initialized(const char *const label) const;

  int is_initialized() const {
    return m_hipDev >= 0;
  }  // 0 != m_scratchSpace && 0 != m_scratchFlags ; }

  void initialize(int hip_device_id);
  void finalize();

  void print_configuration(std::ostream &) const;

  ~HIPInternal();

  HIPInternal()
      : m_hipDev(-1),
        m_hipArch(-1),
        m_multiProcCount(0),
        m_maxWorkgroup(0),
        m_maxSharedWords(0),
        m_scratchSpaceCount(0),
        m_scratchFlagsCount(0),
        m_scratchSpace(0),
        m_scratchFlags(0) {}

  size_type *scratch_space(const size_type size);
  size_type *scratch_flags(const size_type size);
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
