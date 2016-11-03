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

#ifndef KOKKOS_EXPERIMENTAL_ERROR_REPORTER_HPP
#define KOKKOS_EXPERIMENTAL_ERROR_REPORTER_HPP

#include <vector>
#include <Kokkos_Core.hpp>
#include <Kokkos_View.hpp>
#include <Kokkos_DualView.hpp>

namespace kokkos {
namespace Experimental {

template <typename ReportType, typename DeviceType>
class ErrorReporter
{
public:

  typedef ReportType                                      report_type;
  typedef DeviceType                                      device_type;
  typedef typename device_type::execution_space           execution_space;

  ErrorReporter(int max_results)
    : m_numReportsAttempted(""),
      m_reports("", max_results),
      m_reporters("", max_results)
  {
    clear();
  }

  int getCapacity() const { return m_reports.h_view.dimension_0(); }

  int getNumReports();

  int getNumReportAttempts();

  void getReports(std::vector<int> &reporters_out, std::vector<report_type> &reports_out);

  void clear();

  void resize(const size_t new_size);

  bool full() {return (getNumReportAttempts() >= getCapacity()); }

  KOKKOS_INLINE_FUNCTION
  bool add_report(int reporter_id, report_type report) const
  {
    int idx = Kokkos::atomic_fetch_add(&m_numReportsAttempted.d_view(), 1);

    if (idx >= 0 && (idx < static_cast<int>(m_reports.d_view.dimension_0()))) {
      m_reporters.d_view(idx) = reporter_id;
      m_reports.d_view(idx)   = report;
      return true;
    }
    else {
      return false;
    }
  }

private:

  typedef Kokkos::View<report_type *, execution_space>        reports_view_t;
  typedef Kokkos::DualView<report_type *, execution_space>    reports_dualview_t;

  typedef typename reports_dualview_t::host_mirror_space  host_mirror_space;
  Kokkos::DualView<int, execution_space>   m_numReportsAttempted;
  reports_dualview_t                   m_reports;
  Kokkos::DualView<int *, execution_space> m_reporters;

};


template <typename ReportType, typename DeviceType>
inline int ErrorReporter<ReportType, DeviceType>::getNumReports() 
{
  m_numReportsAttempted.template sync<host_mirror_space>();

  int num_reports = m_numReportsAttempted.h_view();
  if (num_reports > static_cast<int>(m_reports.h_view.dimension_0())) {
    num_reports = m_reports.h_view.dimension_0();
  }
  return num_reports;
}

template <typename ReportType, typename DeviceType>
inline int ErrorReporter<ReportType, DeviceType>::getNumReportAttempts()
{
  m_numReportsAttempted.template sync<host_mirror_space>();
  return m_numReportsAttempted.h_view();
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::getReports(std::vector<int> &reporters_out, std::vector<report_type> &reports_out)
{
  int num_reports = getNumReports();
  reporters_out.clear();
  reporters_out.reserve(num_reports);
  reports_out.clear();
  reports_out.reserve(num_reports);

  if (num_reports > 0) {
    m_reports.template sync<host_mirror_space>();
    m_reporters.template sync<host_mirror_space>();

    for (int i = 0; i < num_reports; ++i) {
      reporters_out.push_back(m_reporters.h_view(i));
      reports_out.push_back(m_reports.h_view(i));
    }
  }
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::clear()
{
  m_numReportsAttempted.template modify<host_mirror_space>();
  m_numReportsAttempted.h_view() = 0;
  m_numReportsAttempted.template sync<execution_space>();

  m_numReportsAttempted.template modify<execution_space>();
  m_reports.template modify<execution_space>();
  m_reporters.template modify<execution_space>();
}

template <typename ReportType, typename DeviceType>
void ErrorReporter<ReportType, DeviceType>::resize(const size_t new_size)
{
  m_reports.resize(new_size);
  m_reporters.resize(new_size);
  Kokkos::fence();
}


} // namespace Experimental
} // namespace kokkos

#endif
