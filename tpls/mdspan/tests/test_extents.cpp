//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <mdspan/mdspan.hpp>
#include "offload_utils.hpp"
#include <gtest/gtest.h>


// Making actual test implementations part of the class,
// can't create CUDA lambdas in the primary test functions, since they are private
// making a non-member function would require replicating all the type info

template <class> struct TestExtents;
template <size_t... Extents, size_t... DynamicSizes>
struct TestExtents<
  std::tuple<
    Kokkos::extents<size_t, Extents...>,
    std::integer_sequence<size_t, DynamicSizes...>
  >
> : public ::testing::Test {
  using extents_type = Kokkos::extents<size_t,Extents...>;
  // Double Braces here to make it work with GCC 5
  // Otherwise: "error: array must be initialized with a brace-enclosed initializer"
  const std::array<size_t, sizeof...(Extents)> static_sizes {{ Extents... }};
  const std::array<size_t, sizeof...(DynamicSizes)> dyn_sizes {{ DynamicSizes... }};
  extents_type exts { DynamicSizes... };
  using Fixture = TestExtents< std::tuple<
                    Kokkos::extents<size_t,Extents...>,
                    std::integer_sequence<size_t, DynamicSizes...>
                  >>;

  void test_rank() {
    size_t* result = allocate_array<size_t>(2);

    dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
      extents_type exts(DynamicSizes...);
      // Silencing an unused warning in nvc++ the condition will never be true
      size_t dyn_val = exts.rank()>0?static_cast<size_t>(exts.extent(0)):1;
      result[0] = dyn_val > 1e9 ? dyn_val : exts.rank();
      result[1] = exts.rank_dynamic();
      // Some compilers warn about unused exts since the functions are all static constexpr
      (void) exts;
    });
    EXPECT_EQ(result[0], static_sizes.size());
    EXPECT_EQ(result[1], dyn_sizes.size());

    free_array(result);
  }

  void test_static_extent() {
    size_t* result = allocate_array<size_t>(extents_type::rank());

    dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
      extents_type exts(DynamicSizes...);
      for(size_t r=0; r<exts.rank(); r++) {
        // Silencing an unused warning in nvc++ the condition will never be true
        size_t dyn_val = static_cast<size_t>(exts.extent(r));
        result[r] = dyn_val > 1e9 ? dyn_val : exts.static_extent(r);
      }
      // Some compilers warn about unused exts since the functions are all static constexpr
      (void) exts;
    });
    for(size_t r=0; r<extents_type::rank(); r++) {
      EXPECT_EQ(result[r], static_sizes[r]);
    }

    free_array(result);
  }

  void test_extent() {
    size_t* result = allocate_array<size_t>(extents_type::rank());

    dispatch([=] MDSPAN_IMPL_HOST_DEVICE () {
      extents_type exts(DynamicSizes...);
      for(size_t r=0; r<exts.rank(); r++ )
        result[r] = exts.extent(r);
      // Some compilers warn about unused _exts since the functions are all static constexpr
      (void) exts;
    });
    int dyn_count = 0;
    for(size_t r=0; r<extents_type::rank(); r++) {
      bool is_dynamic = static_sizes[r] == Kokkos::dynamic_extent;
      auto expected = is_dynamic ? dyn_sizes[dyn_count++] : static_sizes[r];
      EXPECT_EQ(result[r], expected);
    }

    free_array(result);
  }
};

template <size_t... Ds>
using sizes = std::integer_sequence<size_t, Ds...>;
template <size_t... Ds>
using exts = Kokkos::extents<size_t,Ds...>;

using extents_test_types =
  ::testing::Types<
    std::tuple<exts<10>, sizes<>>,
    std::tuple<exts<Kokkos::dynamic_extent>, sizes<10>>,
    std::tuple<exts<10, 3>, sizes<>>,
    std::tuple<exts<Kokkos::dynamic_extent, 3>, sizes<10>>,
    std::tuple<exts<10, Kokkos::dynamic_extent>, sizes<3>>,
    std::tuple<exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent>, sizes<10, 3>>
  >;

TYPED_TEST_SUITE(TestExtents, extents_test_types);

TYPED_TEST(TestExtents, rank) {
  MDSPAN_IMPL_TESTS_RUN_TEST(this->test_rank())
}

TYPED_TEST(TestExtents, static_extent) {
  MDSPAN_IMPL_TESTS_RUN_TEST(this->test_static_extent())
}

TYPED_TEST(TestExtents, extent) {
  MDSPAN_IMPL_TESTS_RUN_TEST(this->test_extent())
}

TYPED_TEST(TestExtents, default_ctor) {
  auto e = typename TestFixture::extents_type();
  auto e2 = typename TestFixture::extents_type{};
  EXPECT_EQ(e, e2);
  for (size_t r = 0; r < e.rank(); ++r) {
    bool is_dynamic = (e.static_extent(r) == Kokkos::dynamic_extent);
    EXPECT_EQ(e.extent(r), is_dynamic ? 0 : e.static_extent(r));
  }
}

TYPED_TEST(TestExtents, array_ctor) {
  auto e = typename TestFixture::extents_type(this->dyn_sizes);
  EXPECT_EQ(e, this->exts);
}

TYPED_TEST(TestExtents, copy_ctor) {
  typename TestFixture::extents_type e { this->exts };
  EXPECT_EQ(e, this->exts);
}

TYPED_TEST(TestExtents, copy_assign) {
  typename TestFixture::extents_type e;
  e = this->exts;
  EXPECT_EQ(e, this->exts);
}

// Only types can be part of the gtest type iteration
// so i need this wrapper to wrap around true/false values
template<bool Val1, bool Val2>
struct BoolPairDeducer {
  static constexpr bool val1 = Val1;
  static constexpr bool val2 = Val2;
};


template <class> struct TestExtentsCompatCtors;
template <size_t... Extents, size_t... DynamicSizes, size_t... Extents2, size_t... DynamicSizes2, bool ImplicitExts1ToExts2, bool ImplicitExts2ToExts1>
struct TestExtentsCompatCtors<std::tuple<
  Kokkos::extents<size_t,Extents...>,
  std::integer_sequence<size_t, DynamicSizes...>,
  Kokkos::extents<size_t,Extents2...>,
  std::integer_sequence<size_t, DynamicSizes2...>,
  BoolPairDeducer<ImplicitExts1ToExts2,ImplicitExts2ToExts1>
>> : public ::testing::Test {
  using extents_type1 = Kokkos::extents<size_t,Extents...>;
  using extents_type2 = Kokkos::extents<size_t,Extents2...>;
  extents_type1 exts1 { DynamicSizes... };
  extents_type2 exts2 { DynamicSizes2... };
  static constexpr bool implicit_exts1_to_exts2 = ImplicitExts1ToExts2;
  static constexpr bool implicit_exts2_to_exts1 = ImplicitExts2ToExts1;

  // Idx starts at rank()
  template<class Exts, size_t Idx, bool construct_all, bool Static>
  struct make_extents;

  // Construct all extents - doesn't matter if idx is static or not
  template<class Exts, size_t Idx, bool Static>
  struct make_extents<Exts,Idx,true,Static>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,true,Static>::construct(Idx*5, sizes...);
    }
  };

  // Construct dynamic extents - opnly add a parameter if the current index is not static
  template<class Exts, size_t Idx>
  struct make_extents<Exts,Idx,false,false>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,false,(Idx>1?Exts::static_extent(Idx-2)!=Kokkos::dynamic_extent:true)>::construct(Idx*5, sizes...);
    }
  };
  template<class Exts, size_t Idx>
  struct make_extents<Exts,Idx,false,true>
  {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
      return make_extents<Exts, Idx-1,false,(Idx>1?Exts::static_extent(Idx-2)!=Kokkos::dynamic_extent:true)>::construct(sizes...);
    }
  };

  // End case
  template<class Exts>
  struct make_extents<Exts,0,false,true> {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
       return Exts{sizes...};
    }
  };
  template<class Exts>
  struct make_extents<Exts,0,true,true> {
    template<class ... SizeTypes>
    static Exts construct(SizeTypes...sizes) {
       return Exts{sizes...};
    }
  };
};

using compatible_extents_test_types =
  ::testing::Types<
    std::tuple<exts<Kokkos::dynamic_extent>, sizes<5>, exts<5>, sizes<>, BoolPairDeducer<false,true>>,
    std::tuple<exts<5>, sizes<>, exts<Kokkos::dynamic_extent>, sizes<5>, BoolPairDeducer<true,false>>,
    //--------------------
    std::tuple<exts<Kokkos::dynamic_extent, 10>, sizes<5>, exts<5, Kokkos::dynamic_extent>, sizes<10>,  BoolPairDeducer<false, false>>,
    std::tuple<exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent>, sizes<5, 10>, exts<5, Kokkos::dynamic_extent>, sizes<10>,  BoolPairDeducer<false, true>>,
    std::tuple<exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent>, sizes<5, 10>, exts<Kokkos::dynamic_extent, 10>, sizes<5>,  BoolPairDeducer<false, true>>,
    std::tuple<exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent>, sizes<5, 10>, exts<5, 10>, sizes<>,  BoolPairDeducer<false, true>>,
    std::tuple<exts<5, 10>, sizes<>, exts<5, Kokkos::dynamic_extent>, sizes<10>,  BoolPairDeducer<true, false>>,
    std::tuple<exts<5, 10>, sizes<>, exts<Kokkos::dynamic_extent, 10>, sizes<5>,  BoolPairDeducer<true, false>>,
    //--------------------
    std::tuple<exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent, 15>, sizes<5, 10>, exts<5, Kokkos::dynamic_extent, 15>, sizes<10>,  BoolPairDeducer<false, true>>,
    std::tuple<exts<5, 10, 15>, sizes<>, exts<5, Kokkos::dynamic_extent, 15>, sizes<10>,  BoolPairDeducer<true, false>>,
    std::tuple<exts<5, 10, 15>, sizes<>, exts<Kokkos::dynamic_extent, Kokkos::dynamic_extent, Kokkos::dynamic_extent>, sizes<5, 10, 15>,  BoolPairDeducer<true, false>>
  >;

TYPED_TEST_SUITE(TestExtentsCompatCtors, compatible_extents_test_types);

TYPED_TEST(TestExtentsCompatCtors, compatible_construct_1) {
  auto e1 = typename TestFixture::extents_type1(this->exts2);
  EXPECT_EQ(e1, this->exts2);
}

TYPED_TEST(TestExtentsCompatCtors, compatible_construct_2) {
  auto e2 = typename TestFixture::extents_type2(this->exts1);
  EXPECT_EQ(e2, this->exts1);
}

TYPED_TEST(TestExtentsCompatCtors, compatible_assign_1) {
#if MDSPAN_HAS_CXX_17
  if constexpr(std::is_convertible_v<typename TestFixture::extents_type2,
                           typename TestFixture::extents_type1>) {
    this->exts1 = this->exts2;
  } else {
    this->exts1 = typename TestFixture::extents_type1(this->exts2);
  }
#else
  this->exts1 = typename TestFixture::extents_type1(this->exts2);
#endif
  EXPECT_EQ(this->exts1, this->exts2);

}

TYPED_TEST(TestExtentsCompatCtors, compatible_assign_2) {
#if MDSPAN_HAS_CXX_17
  if constexpr(std::is_convertible_v<typename TestFixture::extents_type1,
                           typename TestFixture::extents_type2>) {
    this->exts2 = this->exts1;
  } else {
    this->exts2 = typename TestFixture::extents_type2(this->exts1);
  }
#else
  this->exts2 = typename TestFixture::extents_type2(this->exts1);
#endif
  EXPECT_EQ(this->exts1, this->exts2);
}


template<class Extents, bool>
struct ImplicitConversionToExts;
template<class Extents>
struct ImplicitConversionToExts<Extents,true> {
  static  bool convert(Extents) {
    return true;
  }
};
template<class Extents>
struct ImplicitConversionToExts<Extents,false> {
  template<class E>
  static  bool convert(E) {
    return false;
  }
};


#if MDSPAN_HAS_CXX_20
TYPED_TEST(TestExtentsCompatCtors, implicit_construct_1) {
  bool exts1_convertible_exts2 =
    std::is_convertible<typename TestFixture::extents_type1,
                        typename TestFixture::extents_type2>::value;
  bool exts2_convertible_exts1 =
    std::is_convertible<typename TestFixture::extents_type2,
                        typename TestFixture::extents_type1>::value;

  bool exts1_implicit_exts2 = ImplicitConversionToExts<
                               typename TestFixture::extents_type2,
                                        TestFixture::implicit_exts1_to_exts2>::convert(this->exts1);

  bool exts2_implicit_exts1 = ImplicitConversionToExts<
                               typename TestFixture::extents_type1,
                                        TestFixture::implicit_exts2_to_exts1>::convert(this->exts2);

  EXPECT_EQ(exts1_convertible_exts2,exts1_implicit_exts2);
  EXPECT_EQ(exts2_convertible_exts1,exts2_implicit_exts1);
  EXPECT_EQ(exts1_convertible_exts2,TestFixture::implicit_exts1_to_exts2);
  EXPECT_EQ(exts2_convertible_exts1,TestFixture::implicit_exts2_to_exts1);
}
#endif

TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_array_convertible_to_size_t) {
  std::array<int, 2> i{2, 2};
  Kokkos::dextents<size_t,2> e{i};
  ASSERT_EQ(e.rank(), 2);
  ASSERT_EQ(e.rank_dynamic(), 2);
  ASSERT_EQ(e.extent(0), 2);
  ASSERT_EQ(e.extent(1), 2);
}

#ifdef __cpp_lib_span
TEST(TestExtentsCtorStdArrayConvertibleToSizeT, test_extents_ctor_std_span_convertible_to_size_t) {
  std::array<int, 2> i{2, 2};
  std::span<int ,2> s(i.data(),2);
  Kokkos::dextents<size_t,2> e{s};
  ASSERT_EQ(e.rank(), 2);
  ASSERT_EQ(e.rank_dynamic(), 2);
  ASSERT_EQ(e.extent(0), 2);
  ASSERT_EQ(e.extent(1), 2);
}
#endif

TYPED_TEST(TestExtentsCompatCtors, construct_from_dynamic_sizes) {
  using e1_t = typename TestFixture::extents_type1;
  constexpr bool e1_last_ext_static = e1_t::rank()>0?e1_t::static_extent(e1_t::rank()-1)!=Kokkos::dynamic_extent:true;
  e1_t e1 = TestFixture::template make_extents<e1_t,e1_t::rank(),false,e1_last_ext_static>::construct();
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  constexpr bool e2_last_ext_static = e2_t::rank()>0?e2_t::static_extent(e2_t::rank()-1)!=Kokkos::dynamic_extent:true;
  e2_t e2 = TestFixture::template make_extents<e2_t,e2_t::rank(),false,e2_last_ext_static>::construct();
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_all_sizes) {
  using e1_t = typename TestFixture::extents_type1;
  e1_t e1 = TestFixture::template make_extents<e1_t,e1_t::rank(),true,true>::construct();
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  e2_t e2 = TestFixture::template make_extents<e2_t,e1_t::rank(),true,true>::construct();
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_dynamic_array) {
  using e1_t = typename TestFixture::extents_type1;
  std::array<int, e1_t::rank_dynamic()> ext1_array_dynamic;

  int dyn_idx = 0;
  for(size_t r=0; r<e1_t::rank(); r++) {
    if(e1_t::static_extent(r)==Kokkos::dynamic_extent) {
      ext1_array_dynamic[dyn_idx] = static_cast<int>((r+1)*5);
      dyn_idx++;
    }
  }
  e1_t e1(ext1_array_dynamic);
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  std::array<int, e2_t::rank_dynamic()> ext2_array_dynamic;

  dyn_idx = 0;
  for(size_t r=0; r<e2_t::rank(); r++) {
    if(e2_t::static_extent(r)==Kokkos::dynamic_extent) {
      ext2_array_dynamic[dyn_idx] = static_cast<int>((r+1)*5);
      dyn_idx++;
    }
  }
  e2_t e2(ext2_array_dynamic);
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

TYPED_TEST(TestExtentsCompatCtors, construct_from_all_array) {
  using e1_t = typename TestFixture::extents_type1;
  std::array<int, e1_t::rank()> ext1_array_all;

  for(size_t r=0; r<e1_t::rank(); r++) ext1_array_all[r] = static_cast<int>((r+1)*5);
  e1_t e1(ext1_array_all);
  for(size_t r=0; r<e1.rank(); r++)
    ASSERT_EQ(e1.extent(r), (r+1)*5);

  using e2_t = typename TestFixture::extents_type2;
  std::array<int, e2_t::rank()> ext2_array_all;

  for(size_t r=0; r<e2_t::rank(); r++) ext2_array_all[r] = static_cast<int>((r+1)*5);
  e2_t e2(ext2_array_all);
  for(size_t r=0; r<e2.rank(); r++)
    ASSERT_EQ(e2.extent(r), (r+1)*5);
}

#if defined(MDSPAN_IMPL_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
TEST(TestExtentsCTADPack, test_extents_ctad_pack) {
  Kokkos::extents m0;
  ASSERT_EQ(m0.rank(), 0);
  ASSERT_EQ(m0.rank_dynamic(), 0);

  Kokkos::extents m1(64);
  ASSERT_EQ(m1.rank(), 1);
  ASSERT_EQ(m1.rank_dynamic(), 1);
  ASSERT_EQ(m1.extent(0), 64);

  Kokkos::extents m2(64, 128);
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 64);
  ASSERT_EQ(m2.extent(1), 128);

  Kokkos::extents m3(64, 128, 256);
  ASSERT_EQ(m3.rank(), 3);
  ASSERT_EQ(m3.rank_dynamic(), 3);
  ASSERT_EQ(m3.extent(0), 64);
  ASSERT_EQ(m3.extent(1), 128);
  ASSERT_EQ(m3.extent(2), 256);
}

// TODO: It appears to currently be impossible to write a deduction guide that
// makes this work.
/*
TEST(TestExtentsCTADStdArray, test_extents_ctad_std_array) {
  Kokkos::extents m0{std::array<size_t, 0>{}};
  ASSERT_EQ(m0.rank(), 0);
  ASSERT_EQ(m0.rank_dynamic(), 0);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  Kokkos::extents m1{std::array{64UL}};
  ASSERT_EQ(m1.rank(), 1);
  ASSERT_EQ(m1.rank_dynamic(), 1);
  ASSERT_EQ(m1.extent(0), 64);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  Kokkos::extents m2{std::array{64UL, 128UL}};
  ASSERT_EQ(m2.rank(), 2);
  ASSERT_EQ(m2.rank_dynamic(), 2);
  ASSERT_EQ(m2.extent(0), 64);
  ASSERT_EQ(m2.extent(1), 128);

  // TODO: `extents` should accept an array of any type convertible to `size_t`.
  Kokkos::extents m3{std::array{64UL, 128UL, 256UL}};
  ASSERT_EQ(m3.rank(), 3);
  ASSERT_EQ(m3.rank_dynamic(), 3);
  ASSERT_EQ(m3.extent(0), 64);
  ASSERT_EQ(m3.extent(1), 128);
  ASSERT_EQ(m3.extent(2), 256);
}
*/
#endif
