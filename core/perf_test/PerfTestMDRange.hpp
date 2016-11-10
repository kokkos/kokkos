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

namespace Test {
template< class DeviceType 
        , typename ScalarType = double  
        , typename TestLayout = Kokkos::LayoutRight  
        >
struct MultiDimRangePerf3D
{
  typedef DeviceType execution_space;
  typedef typename execution_space::size_type  size_type;

  using iterate_type = Kokkos::Experimental::Iterate;

  typedef Kokkos::View<ScalarType***, TestLayout, DeviceType> view_type;
  typedef typename view_type::HostMirror host_view_type;

  view_type A;
  view_type B;
  const int irange;
  const int jrange;
  const int krange;

  MultiDimRangePerf3D(const view_type & A_, const view_type & B_, const int &irange_,  const int &jrange_, const int &krange_)
  : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j, const int k) const
  {
    A(i,j,k) = 0.143*(double)( B(i+2,j,k) + B(i+1,j,k)
                             + B(i,j+2,k) + B(i,j+1,k)
                             + B(i,j,k+2) + B(i,j,k+1)
                             + B(i,j,k) );
  }


  struct InitZeroTag {};
//  struct InitViewTag {};

  struct Init
  {

    Init(const view_type & input_, const int &irange_,  const int &jrange_, const int &krange_)
    : input(input_), irange(irange_), jrange(jrange_), krange(krange_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const
    {
      input(i,j,k) = 1.0;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const InitZeroTag&, const int i, const int j, const int k) const
    {
      input(i,j,k) = 0;
    }

    view_type input;
    const int irange;
    const int jrange;
    const int krange;
  };


  static double test_multi_index(const unsigned icount, const unsigned jcount, const unsigned kcount, const unsigned int Ti = 1, const unsigned int Tj = 1, const unsigned int Tk = 1, const int iter = 1)
  {
    //This test performs multidim range over all dims
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount+2, jcount+2, kcount+2);
    typedef MultiDimRangePerf3D<execution_space,ScalarType,TestLayout> FunctorType;

    double dt_min = 0;

    if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value ) {
      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3, iterate_type::Right, iterate_type::Right>, execution_space > policy_init({{0,0,0}},{{icount,jcount,kcount}},{{Ti,Tj,Tk}}); 
      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3, iterate_type::Right, iterate_type::Right>, execution_space > policy_initB({{0,0,0}},{{icount+2,jcount+2,kcount+2}},{{Ti,Tj,Tk}}); 

      typedef typename Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3, iterate_type::Right, iterate_type::Right>, execution_space > MDRangeType;
      using tile_type = typename MDRangeType::tile_type;
      using point_type = typename MDRangeType::point_type;

      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3, iterate_type::Right, iterate_type::Right>, execution_space > policy(point_type{{0,0,0}},point_type{{icount,jcount,kcount}},tile_type{{Ti,Tj,Tk}} );

      Kokkos::Experimental::md_parallel_for( policy_init, Init(Atest, icount, jcount, kcount) );
      execution_space::fence();
      Kokkos::Experimental::md_parallel_for( policy_initB, Init(Btest, icount+2, jcount+2, kcount+2) );
      execution_space::fence();

    for (int i = 0; i < iter; ++i)
    {
      Kokkos::Timer timer;
      Kokkos::Experimental::md_parallel_for( policy, FunctorType(Atest, Btest, icount, jcount, kcount) );
      execution_space::fence();
      const double dt = timer.seconds();
      if ( 0 == i ) dt_min = dt ;
      else dt_min = dt < dt_min ? dt : dt_min ;

      //Correctness check - only the first run
      if ( 0 == i )
      {
        int numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount+2, jcount+2, kcount+2);
        Kokkos::deep_copy(Bhost, Btest);

        for ( int l = 0; l < icount; ++l ) {
        for ( int j = 0; j < jcount; ++j ) {
        for ( int k = 0; k < kcount; ++k ) {
          //double check = (l*l + j - k*l + 2.0*k*j - j*j*j);
          double check  = 0.143*(double)( Bhost(i+2,j,k) + Bhost(i+1,j,k)
                                        + Bhost(i,j+2,k) + Bhost(i,j+1,k)
                                        + Bhost(i,j,k+2) + Bhost(i,j,k+1)
                                        + Bhost(i,j,k) );
          if ( Ahost(l,j,k) - check != 0 ) {
            ++numErrors;
            std::cout << "  Correctness error at " << l << " "<<j<<" "<<k<<"\n"
                      << "  multi Ahost = " << Ahost(l,j,k) << "  expected = " << check  
                      << "  multi Bhost(ijk) = " << Bhost(l,j,k) 
                      << "  multi Bhost(i+1jk) = " << Bhost(l+1,j,k) 
                      << "  multi Bhost(i+2jk) = " << Bhost(l+2,j,k) 
                      << std::endl;
            //exit(-1);
          }
        } } }
        if ( numErrors != 0 ) { std::cout << "LR multi: errors " << numErrors <<  std::endl; }
        //else { std::cout << " multi: No errors!" <<  std::endl; }
      }
    } //end for

    } 
    else {
      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3,iterate_type::Left,iterate_type::Left>, execution_space > policy_init({{0,0,0}},{{icount,jcount,kcount}},{{Ti,Tj,Tk}}); 
      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3,iterate_type::Right,iterate_type::Right>, execution_space > policy_initB({{0,0,0}},{{icount+2,jcount+2,kcount+2}},{{Ti,Tj,Tk}}); 

      Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<3, iterate_type::Left, iterate_type::Left>, execution_space > policy({{0,0,0}},{{icount,jcount,kcount}},{{Ti,Tj,Tk}} ); 

      Kokkos::Experimental::md_parallel_for( policy_init, Init(Atest, icount, jcount, kcount) );
      execution_space::fence();
      Kokkos::Experimental::md_parallel_for( policy_initB, Init(Btest, icount+2, jcount+2, kcount+2) );
      execution_space::fence();

    for (int i = 0; i < iter; ++i)
    {
      Kokkos::Timer timer;
      Kokkos::Experimental::md_parallel_for( policy, FunctorType(Atest, Btest, icount, jcount, kcount) );
      execution_space::fence();
      const double dt = timer.seconds();
      if ( 0 == i ) dt_min = dt ;
      else dt_min = dt < dt_min ? dt : dt_min ;

      //Correctness check - only the first run
      if ( 0 == i )
      {
        int numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount+2, jcount+2, kcount+2);
        Kokkos::deep_copy(Bhost, Btest);

        for ( int l = 0; l < icount; ++l ) {
        for ( int j = 0; j < jcount; ++j ) {
        for ( int k = 0; k < kcount; ++k ) {
          //double check = (l*l + j - k*l + 2.0*k*j - j*j*j);
          double check  = 0.143*(double)( Bhost(i+2,j,k) + Bhost(i+1,j,k)
                                        + Bhost(i,j+2,k) + Bhost(i,j+1,k)
                                        + Bhost(i,j,k+2) + Bhost(i,j,k+1)
                                        + Bhost(i,j,k) );
          if ( Ahost(l,j,k) - check != 0 ) {
            ++numErrors;
            std::cout << "  Correctness error at " << l << " "<<j<<" "<<k<<"\n"
                      << "  multi Ahost = " << Ahost(l,j,k) << "  expected = " << check  
                      << "  multi Bhost(ijk) = " << Bhost(l,j,k) 
                      << "  multi Bhost(i+1jk) = " << Bhost(l+1,j,k) 
                      << "  multi Bhost(i+2jk) = " << Bhost(l+2,j,k) 
                      << std::endl;
            //exit(-1);
          }
        } } }
        if ( numErrors != 0 ) { std::cout << " LL multi run: errors " << numErrors <<  std::endl; }
        //else { std::cout << " multi: No errors!" <<  std::endl; }

      }
    } //end for
    }

    return dt_min;
  } 

};


template< class DeviceType 
        , typename ScalarType = double  
        , typename TestLayout = Kokkos::LayoutRight  
        >
struct MultiDimRangePerf3D_Collapse
{
  // 3D Range, but will collapse only 2 dims => Rank<2> for multi-dim; unroll 2 dims in one-dim

  typedef DeviceType execution_space;
  typedef typename execution_space::size_type  size_type;

  using iterate_type = Kokkos::Experimental::Iterate;

  typedef Kokkos::View<ScalarType***, TestLayout, DeviceType> view_type;
  typedef typename view_type::HostMirror host_view_type;

  view_type A;
  view_type B;
  const int irange;
  const int jrange;
  const int krange;

  MultiDimRangePerf3D_Collapse(const view_type & A_, const view_type & B_, const int &irange_,  const int &jrange_, const int &krange_)
  : A(A_), B(B_) , irange(irange_), jrange(jrange_), krange(krange_)
  {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
  {
    if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value )
    {
//id(i,j,k) = k + j*Nk + i*Nk*Nj = k + Nk*(j + i*Nj) = k + Nk*r
//      const int i = r / (jrange*krange);
//      const int j = (r - i*jrange*krange)/jrange;
//r = j + i*Nj
      const int i = r / (jrange); 
      const int j = ( r - i*jrange);
      for (int k = 0; k < krange; ++k) {
    //    A(i,j,k) = (double)(i*i + j - k*i + 2.0*k*j - j*j*j);
        A(i,j,k) = 0.143*(double)( B(i+2,j,k) + B(i+1,j,k)
                                 + B(i,j+2,k) + B(i,j+1,k)
                                 + B(i,j,k+2) + B(i,j,k+1)
                                 + B(i,j,k) );
      }
    }
    else if ( std::is_same<TestLayout, Kokkos::LayoutLeft>::value )
    {
//id(i,j,k) = i + j*Ni + k*Ni*Nj = i + Ni*(j + k*Nj) = i + Ni*r
//      const int k = r / (irange*jrange); 
//      const int j = ( r - k*irange*jrange)/irange;
//r = j + k*Nj
      const int k = r / (jrange); 
      const int j = ( r - k*jrange);
      for (int i = 0; i < irange; ++i) {
    //    A(i,j,k) = (double)(i*i + j - k*i + 2.0*k*j - j*j*j);
        A(i,j,k) = 0.143*(double)( B(i+2,j,k) + B(i+1,j,k)
                                 + B(i,j+2,k) + B(i,j+1,k)
                                 + B(i,j,k+2) + B(i,j,k+1)
                                 + B(i,j,k) );
      }
    }

  }


  struct Init
  {
    view_type input;
    const int irange;
    const int jrange;
    const int krange;

    Init(const view_type & input_, const int &irange_,  const int &jrange_, const int &krange_)
    : input(input_), irange(irange_), jrange(jrange_), krange(krange_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int r) const
    {
      if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value )
      {
        const int i = r / (jrange*krange); 
        const int j = ( r - i*jrange*krange)/krange;
        const int k = r - i*jrange*krange - j*krange;
        input(i,j,k) = 1;
      }
      else if ( std::is_same<TestLayout, Kokkos::LayoutLeft>::value )
      {
        const int k = r / (irange*jrange); 
        const int j = ( r - k*irange*jrange)/irange;
        const int i = r - k*irange*jrange - j*irange;
        input(i,j,k) = 1;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const
    {
      input(i,j,k) = 1;
    }

  };


  static double test_index_collapse(const unsigned icount, const unsigned jcount, const unsigned kcount, const int iter = 1)
  {
    // This test refers to collapsing two dims while using the RangePolicy
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount+2, jcount+2, kcount+2);
    typedef MultiDimRangePerf3D_Collapse<execution_space,ScalarType,TestLayout> FunctorType;

    int collapse_index_range = 0;
    int collapse_index_rangeB = 0;
    if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value ) {
      collapse_index_range = icount*jcount;
      collapse_index_rangeB = (icount+2)*(jcount+2);
//      std::cout << "   LayoutRight " << std::endl;
    } else if ( std::is_same<TestLayout, Kokkos::LayoutLeft>::value ) {
      collapse_index_range = kcount*jcount;
      collapse_index_rangeB = (kcount+2)*(jcount+2);
//      std::cout << "   LayoutLeft " << std::endl;
    } else {
      std::cout << "  LayoutRight or LayoutLeft required - will pass 0 as range instead " << std::endl;
      exit(-1);
    }

    Kokkos::RangePolicy<execution_space> policy(0, (collapse_index_range) );
    Kokkos::RangePolicy<execution_space> policy_initB(0, (collapse_index_rangeB) );
//    std::cout << "   Full flattened range (i.e. product of ranges) " << icount*jcount*kcount << std::endl;
//    std::cout << "   Value outside of the if-guard " << collapse_index_range << std::endl;

    double dt_min = 0;

    Kokkos::parallel_for( policy, Init(Atest,icount,jcount,kcount) );
    execution_space::fence();
    Kokkos::parallel_for( policy_initB, Init(Btest,icount+2,jcount+2,kcount+2) );
    execution_space::fence();

    for (int i = 0; i < iter; ++i)
    {
      Kokkos::Timer timer;
      Kokkos::parallel_for(policy, FunctorType(Atest, Btest, icount, jcount, kcount));
      execution_space::fence();
      const double dt = timer.seconds();
      if ( 0 == i ) dt_min = dt ;
      else dt_min = dt < dt_min ? dt : dt_min ;

      //Correctness check
      if ( 0 == i )
      {
        int numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount+2, jcount+2, kcount+2);
        Kokkos::deep_copy(Bhost, Btest);

        for ( int l = 0; l < icount; ++l ) {
        for ( int j = 0; j < jcount; ++j ) {
        for ( int k = 0; k < kcount; ++k ) {
          //double check = (l*l + j - k*l + 2.0*k*j - j*j*j);
          double check  = 0.143*(double)( Bhost(l+2,j,k) + Bhost(l+1,j,k)
                                        + Bhost(l,j+2,k) + Bhost(l,j+1,k)
                                        + Bhost(l,j,k+2) + Bhost(l,j,k+1)
                                        + Bhost(l,j,k) );
          if ( Ahost(l,j,k) - check != 0 ) {
            ++numErrors;
//           std::cout << "  Correctness error at " << l << " "<<j<<" "<<k<<"\n"
//                      << "  flat Ahost = " << Ahost(l,j,k) << "  expected = " << check  << std::endl;
            //exit(-1);
          }
        } } }
        if ( numErrors != 0 ) { std::cout << " RP collapse2: errors " << numErrors <<  std::endl; }
        //else { std::cout << " RP collapse2: Pass! " << std::endl; }
//        if ( numErrors == 0 ) { std::cout << " flattened: 0 errors, good deal " << std::endl; }
      }
    }

    return dt_min;
  } 

};

template< class DeviceType 
        , typename ScalarType = double  
        , typename TestLayout = Kokkos::LayoutRight  
        >
struct MultiDimRangePerf3D_CollapseAll
{
  // 3D Range, but will collapse only 2 dims => Rank<2> for multi-dim; unroll 2 dims in one-dim

  typedef DeviceType execution_space;
  typedef typename execution_space::size_type  size_type;

  using iterate_type = Kokkos::Experimental::Iterate;

  iterate_type inner_direction;
  iterate_type outer_direction;

//  typedef MultiDimRangePerf3D<DeviceType,ScalarType,TestLayout> self_type;

  typedef Kokkos::View<ScalarType***, TestLayout, DeviceType> view_type;
  typedef typename view_type::HostMirror host_view_type;

  view_type A;
  view_type B;
  const int irange;
  const int jrange;
  const int krange;

  MultiDimRangePerf3D_CollapseAll(const view_type & A_, const view_type & B_, const int &irange_,  const int &jrange_, const int &krange_)
  : A(A_), B(B_), irange(irange_), jrange(jrange_), krange(krange_)
  {
    if ( std::is_same<TestLayout , Kokkos::LayoutRight>::value ) {
      inner_direction = iterate_type::Right;
      outer_direction = iterate_type::Right;
    }
    else {
      inner_direction = iterate_type::Left;
      outer_direction = iterate_type::Left;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int r) const
  {
    if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value )
    {
      const int i = r / (jrange*krange); 
      const int j = ( r - i*jrange*krange)/krange;
      const int k = r - i*jrange*krange - j*krange;
        //A(i,j,k) = (double)(i*i + j - k*i + 2.0*k*j - j*j*j);
        A(i,j,k) = 0.143*(double)( B(i+2,j,k) + B(i+1,j,k)
            + B(i,j+2,k) + B(i,j+1,k)
            + B(i,j,k+2) + B(i,j,k+1)
            + B(i,j,k) );
    }
    else if ( std::is_same<TestLayout, Kokkos::LayoutLeft>::value )
    {
      const int k = r / (irange*jrange); 
      const int j = ( r - k*irange*jrange)/irange;
      const int i = r - k*irange*jrange - j*irange;
        //A(i,j,k) = (double)(i*i + j - k*i + 2.0*k*j - j*j*j);
        A(i,j,k) = 0.143*(double)( B(i+2,j,k) + B(i+1,j,k)
            + B(i,j+2,k) + B(i,j+1,k)
            + B(i,j,k+2) + B(i,j,k+1)
            + B(i,j,k) );
    }

  }


  struct Init
  {
    view_type input;
    const int irange;
    const int jrange;
    const int krange;

    Init(const view_type & input_, const int &irange_,  const int &jrange_, const int &krange_)
    : input(input_), irange(irange_), jrange(jrange_), krange(krange_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int r) const
    {
    if ( std::is_same<TestLayout, Kokkos::LayoutRight>::value )
    {
      const int i = r / (jrange*krange); 
      const int j = ( r - i*jrange*krange)/krange;
      const int k = r - i*jrange*krange - j*krange;
      input(i,j,k) = 1;
    }
    else if ( std::is_same<TestLayout, Kokkos::LayoutLeft>::value )
    {
      const int k = r / (irange*jrange); 
      const int j = ( r - k*irange*jrange)/irange;
      const int i = r - k*irange*jrange - j*irange;
      input(i,j,k) = 1;
    }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const
    {
      input(i,j,k) = 1;
    }

  };


  static double test_collapse_all(const unsigned icount, const unsigned jcount, const unsigned kcount, const int iter = 1)
  {
    //This test refers to collapsing all dims using the RangePolicy
    view_type Atest("Atest", icount, jcount, kcount);
    view_type Btest("Btest", icount+2, jcount+2, kcount+2);
    typedef MultiDimRangePerf3D_CollapseAll<execution_space,ScalarType,TestLayout> FunctorType;

    int flat_index_range = 0;
    flat_index_range = icount*jcount*kcount;
    Kokkos::RangePolicy<execution_space> policy(0, flat_index_range );
    Kokkos::RangePolicy<execution_space> policy_initB(0, (icount+2)*(jcount+2)*(kcount+2) );
//    std::cout << "   Full flattened range (i.e. product of ranges) " << icount*jcount*kcount << std::endl;
//    std::cout << "   Value outside of the if-guard " << flat_index_range << std::endl;

    double dt_min = 0;

    Kokkos::parallel_for( policy, Init(Atest,icount,jcount,kcount) );
    Kokkos::parallel_for( policy_initB, Init(Btest,icount+2,jcount+2,kcount+2) );
    execution_space::fence();

    for (int i = 0; i < iter; ++i)
    {
      Kokkos::Timer timer;
      Kokkos::parallel_for(policy, FunctorType(Atest, Btest, icount, jcount, kcount));
      execution_space::fence();
      const double dt = timer.seconds();
      if ( 0 == i ) dt_min = dt ;
      else dt_min = dt < dt_min ? dt : dt_min ;

      //Correctness check
      if ( 0 == i )
      {
        int numErrors = 0;
        host_view_type Ahost("Ahost", icount, jcount, kcount);
        Kokkos::deep_copy(Ahost, Atest);
        host_view_type Bhost("Bhost", icount+2, jcount+2, kcount+2);
        Kokkos::deep_copy(Bhost, Btest);

        for ( int l = 0; l < icount; ++l ) {
        for ( int j = 0; j < jcount; ++j ) {
        for ( int k = 0; k < kcount; ++k ) {
          //double check = (l*l + j - k*l + 2.0*k*j - j*j*j);
          double check  = 0.143*(double)( Bhost(l+2,j,k) + Bhost(l+1,j,k)
                                        + Bhost(l,j+2,k) + Bhost(l,j+1,k)
                                        + Bhost(l,j,k+2) + Bhost(l,j,k+1)
                                        + Bhost(l,j,k) );
          if ( Ahost(l,j,k) - check != 0 ) {
            ++numErrors;
//           std::cout << "  Correctness error at " << l << " "<<j<<" "<<k<<"\n"
//                      << "  flat Ahost = " << Ahost(l,j,k) << "  expected = " << check  << std::endl;
            //exit(-1);
          }
        } } }
        if ( numErrors != 0 ) { std::cout << " RP collapse all: errors " << numErrors <<  std::endl; }
        //else { std::cout << " RP collapse all: Pass! " << std::endl; }
//        if ( numErrors == 0 ) { std::cout << " flattened: 0 errors, good deal " << std::endl; }
      }
    }

    return dt_min;
  } 

};

} //end namespace Test
