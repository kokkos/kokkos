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

#ifndef SYSTEM_H_
#define SYSTEM_H_

#include <types.h>

struct Box {
  double xprd, yprd, zprd;
  double xlo, xhi;
  double ylo, yhi;
  double zlo, zhi;
};

struct System {
  Box box;

  int natoms;
  int nlocal;
  int nghost;

  t_x_array d_x;
  t_x_array_host h_x;

  t_f_array f;

  t_neighbors neighbors;
  t_int_1d numneigh;

  double delta;

  double neigh_cut, neigh_cutsq;

  int mbins;
  int nbinx, nbiny, nbinz;
  int mbinx, mbiny, mbinz;
  int mbinxlo, mbinylo, mbinzlo;
  double binsizex, binsizey, binsizez;
  double bininvx, bininvy, bininvz;

  t_int_1d bincount;
  t_int_2d bins;
  t_int_scalar d_resize;
  t_int_scalar_host h_resize;
  t_int_1d d_stencil;
  t_int_1d_host h_stencil;
  int nstencil;

  double force_cut, force_cutsq;
};
#endif
