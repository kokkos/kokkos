// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_TEST_SIMD_BITWISE_OPS_HPP
#define KOKKOS_TEST_SIMD_BITWISE_OPS_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.simd;
#else
#include <Kokkos_SIMD.hpp>
#endif
#include <SIMDTesting_Utilities.hpp>

template <class Abi, class Loader, bool is_compound_op, class BinaryOp, class T>
void host_check_bitwise_op_one_loader(BinaryOp binary_op, std::size_t n,
                                      T const* first_args,
                                      T const* second_args) {
  Loader loader;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();
  for (std::size_t i = 0; i < n; i += width) {
    std::size_t const nremaining = n - i;
    std::size_t const nlanes     = Kokkos::min(nremaining, width);
    if ((std::is_same_v<BinaryOp, divides> ||
         std::is_same_v<BinaryOp, divides_eq>) &&
        nremaining < width)
      continue;
    simd_type first_arg;
    bool const loaded_first_arg =
        loader.host_load(first_args + i, nlanes, first_arg);
    simd_type second_arg;
    bool const loaded_second_arg =
        loader.host_load(second_args + i, nlanes, second_arg);
    if (!(loaded_first_arg && loaded_second_arg)) continue;

    T expected_val[width];
    for (std::size_t lane = 0; lane < width; ++lane) {
      T tmp              = first_arg[lane];
      expected_val[lane] = binary_op.on_host(tmp, T(second_arg[lane]));
    }

    simd_type expected_result =
        Kokkos::Experimental::simd_unchecked_load<simd_type>(
            expected_val, Kokkos::Experimental::simd_flag_default);
    simd_type const computed_result = binary_op.on_host(first_arg, second_arg);
    host_check_equality(expected_result, computed_result, nlanes);
    if constexpr (is_compound_op) {
      host_check_equality(first_arg, expected_result, nlanes);
    }
  }
}

template <class Abi, class Loader, bool, class UnaryOp, class T>
void host_check_bitwise_op_one_loader(UnaryOp unary_op, std::size_t n,
                                      T const* args) {
  Loader loader;
  using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;

  constexpr std::size_t width = simd_type::size();
  for (std::size_t i = 0; i < n; i += width) {
    std::size_t const nremaining = n - i;
    std::size_t const nlanes     = Kokkos::min(nremaining, width);
    simd_type arg;
    bool const loaded_arg = loader.host_load(args + i, nlanes, arg);
    if (!loaded_arg) continue;

    auto unary_op_result   = unary_op.on_host(arg);
    using result_simd_type = decltype(unary_op_result);

    typename result_simd_type::value_type expected_val[width];
    for (std::size_t lane = 0; lane < width; ++lane) {
      expected_val[lane] = unary_op.on_host(T(arg[lane]));
    }

    result_simd_type expected_result =
        Kokkos::Experimental::simd_unchecked_load<result_simd_type>(
            expected_val, Kokkos::Experimental::simd_flag_default);
    auto computed_result = unary_op.on_host(arg);
    host_check_equality(expected_result, computed_result, nlanes);
  }
}

template <class Abi, bool is_compound_op, class Op, class... T>
inline void host_check_bitwise_op_all_loaders(Op op, std::size_t n,
                                              T const*... args) {
  host_check_bitwise_op_one_loader<Abi, load_element_aligned, is_compound_op>(
      op, n, args...);
  host_check_bitwise_op_one_loader<Abi, load_masked, is_compound_op>(op, n,
                                                                     args...);
  host_check_bitwise_op_one_loader<Abi, load_as_scalars, is_compound_op>(
      op, n, args...);
  host_check_bitwise_op_one_loader<Abi, load_vector_aligned, is_compound_op>(
      op, n, args...);
}

template <typename Abi, typename DataType, size_t n>
inline void host_check_all_bitwise_ops(const DataType (&first_args)[n],
                                       const DataType (&second_args)[n]) {
  host_check_bitwise_op_all_loaders<Abi, false>(bitwise_not(), n, first_args);
  host_check_bitwise_op_all_loaders<Abi, false>(bitwise_and(), n, first_args,
                                                second_args);
  host_check_bitwise_op_all_loaders<Abi, false>(bitwise_or(), n, first_args,
                                                second_args);
  host_check_bitwise_op_all_loaders<Abi, false>(bitwise_xor(), n, first_args,
                                                second_args);
  host_check_bitwise_op_all_loaders<Abi, true>(bitwise_and_eq(), n, first_args,
                                               second_args);
  host_check_bitwise_op_all_loaders<Abi, true>(bitwise_or_eq(), n, first_args,
                                               second_args);
  host_check_bitwise_op_all_loaders<Abi, true>(bitwise_xor_eq(), n, first_args,
                                               second_args);
}

template <typename Abi, typename DataType>
inline void host_check_bitwise_ops() {
  if constexpr (is_simd_avail_v<DataType, Abi> &&
                std::is_integral_v<DataType>) {
    constexpr size_t alignment =
        Kokkos::Experimental::basic_simd<DataType, Abi>::size() *
        sizeof(DataType);

    constexpr int half_shift   = (CHAR_BIT * sizeof(DataType)) / 2;
    constexpr DataType zero    = static_cast<DataType>(0);
    constexpr DataType all_set = ~zero;
    constexpr DataType hi_set  = all_set << half_shift;
    constexpr DataType lo_set  = ~hi_set;

    alignas(alignment) DataType const first_args[] = {
        0,         0,          1,        all_set,  all_set,   all_set,
        all_set,   lo_set,     hi_set,   0,        704475968, 1845076239,
        432747131, 1285171335, 17011965, 139561533};
    alignas(alignment) DataType const second_args[] = {
        0,         1,          1,          0,         all_set,    hi_set,
        lo_set,    lo_set,     hi_set,     hi_set,    1841853988, 747428271,
        357605498, 1412297337, 1663131103, 2062867687};
    host_check_all_bitwise_ops<Abi>(first_args, second_args);
  }
}

template <typename Abi, typename... DataTypes>
inline void host_check_bitwise_ops_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (host_check_bitwise_ops<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void host_check_bitwise_ops_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (host_check_bitwise_ops_all_types<Abis>(DataTypes()), ...);
}

template <class Abi, class Loader, bool is_compound_op, class BinaryOp, class T>
void device_check_bitwise_op_one_loader(BinaryOp binary_op, std::size_t n,
                                        T const* first_args,
                                        T const* second_args) {
  Loader loader;
  using simd_type             = Kokkos::Experimental::basic_simd<T, Abi>;
  constexpr std::size_t width = simd_type::size();
  for (std::size_t i = 0; i < n; i += width) {
    std::size_t const nremaining = n - i;
    std::size_t const nlanes     = Kokkos::min(nremaining, width);
    if ((std::is_same_v<BinaryOp, divides> ||
         std::is_same_v<BinaryOp, divides_eq>) &&
        nremaining < width)
      continue;
    simd_type first_arg;
    bool const loaded_first_arg =
        loader.device_load(first_args + i, nlanes, first_arg);
    simd_type second_arg;
    bool const loaded_second_arg =
        loader.device_load(second_args + i, nlanes, second_arg);
    if (!(loaded_first_arg && loaded_second_arg)) continue;

    T expected_val[width];
    for (std::size_t lane = 0; lane < width; ++lane) {
      T tmp              = first_arg[lane];
      expected_val[lane] = binary_op.on_device(tmp, T(second_arg[lane]));
    }

    simd_type expected_result =
        Kokkos::Experimental::simd_unchecked_load<simd_type>(
            expected_val, Kokkos::Experimental::simd_flag_default);
    simd_type const computed_result =
        binary_op.on_device(first_arg, second_arg);
    device_check_equality(expected_result, computed_result, nlanes);
    if constexpr (is_compound_op) {
      device_check_equality(first_arg, expected_result, nlanes);
    }
  }
}

template <class Abi, class Loader, bool, class UnaryOp, class T>
void device_check_bitwise_op_one_loader(UnaryOp unary_op, std::size_t n,
                                        T const* args) {
  Loader loader;
  using simd_type = Kokkos::Experimental::basic_simd<T, Abi>;

  constexpr std::size_t width = simd_type::size();
  for (std::size_t i = 0; i < n; i += width) {
    std::size_t const nremaining = n - i;
    std::size_t const nlanes     = Kokkos::min(nremaining, width);
    simd_type arg;
    bool const loaded_arg = loader.device_load(args + i, nlanes, arg);
    if (!loaded_arg) continue;

    auto unary_op_result   = unary_op.on_device(arg);
    using result_simd_type = decltype(unary_op_result);

    typename result_simd_type::value_type expected_val[width];
    for (std::size_t lane = 0; lane < width; ++lane) {
      expected_val[lane] = unary_op.on_device(T(arg[lane]));
    }

    result_simd_type expected_result =
        Kokkos::Experimental::simd_unchecked_load<result_simd_type>(
            expected_val, Kokkos::Experimental::simd_flag_default);
    auto computed_result = unary_op.on_device(arg);
    device_check_equality(expected_result, computed_result, nlanes);
  }
}

template <class Abi, bool is_compound_op, class Op, class... T>
inline void device_check_bitwise_op_all_loaders(Op op, std::size_t n,
                                                T const*... args) {
  device_check_bitwise_op_one_loader<Abi, load_element_aligned, is_compound_op>(
      op, n, args...);
  device_check_bitwise_op_one_loader<Abi, load_masked, is_compound_op>(op, n,
                                                                       args...);
  device_check_bitwise_op_one_loader<Abi, load_as_scalars, is_compound_op>(
      op, n, args...);
  device_check_bitwise_op_one_loader<Abi, load_vector_aligned, is_compound_op>(
      op, n, args...);
}

template <typename Abi, typename DataType, size_t n>
inline void device_check_all_bitwise_ops(const DataType (&first_args)[n],
                                         const DataType (&second_args)[n]) {
  device_check_bitwise_op_all_loaders<Abi, false>(bitwise_not(), n, first_args);
  device_check_bitwise_op_all_loaders<Abi, false>(bitwise_and(), n, first_args,
                                                  second_args);
  device_check_bitwise_op_all_loaders<Abi, false>(bitwise_or(), n, first_args,
                                                  second_args);
  device_check_bitwise_op_all_loaders<Abi, false>(bitwise_xor(), n, first_args,
                                                  second_args);
  device_check_bitwise_op_all_loaders<Abi, true>(bitwise_and_eq(), n,
                                                 first_args, second_args);
  device_check_bitwise_op_all_loaders<Abi, true>(bitwise_or_eq(), n, first_args,
                                                 second_args);
  device_check_bitwise_op_all_loaders<Abi, true>(bitwise_xor_eq(), n,
                                                 first_args, second_args);
}

template <typename Abi, typename DataType>
inline void device_check_bitwise_ops() {
  if constexpr (is_simd_avail_v<DataType, Abi> &&
                std::is_integral_v<DataType>) {
    constexpr size_t alignment =
        Kokkos::Experimental::basic_simd<DataType, Abi>::size() *
        sizeof(DataType);

    constexpr int half_shift   = (CHAR_BIT * sizeof(DataType)) / 2;
    constexpr DataType zero    = static_cast<DataType>(0);
    constexpr DataType all_set = ~zero;
    constexpr DataType hi_set  = all_set << half_shift;
    constexpr DataType lo_set  = ~hi_set;

    alignas(alignment) DataType const first_args[] = {
        0,         0,          1,        all_set,  all_set,   all_set,
        all_set,   lo_set,     hi_set,   0,        704475968, 1845076239,
        432747131, 1285171335, 17011965, 139561533};
    alignas(alignment) DataType const second_args[] = {
        0,         1,          1,          0,         all_set,    hi_set,
        lo_set,    lo_set,     hi_set,     hi_set,    1841853988, 747428271,
        357605498, 1412297337, 1663131103, 2062867687};
    device_check_all_bitwise_ops<Abi>(first_args, second_args);
  }
}

template <typename Abi, typename... DataTypes>
inline void device_check_bitwise_ops_all_types(
    Kokkos::Experimental::Impl::data_types<DataTypes...>) {
  (device_check_bitwise_ops<Abi, DataTypes>(), ...);
}

template <typename... Abis>
inline void device_check_bitwise_ops_all_abis(
    Kokkos::Experimental::Impl::abi_set<Abis...>) {
  using DataTypes = Kokkos::Experimental::Impl::data_type_set;
  (device_check_bitwise_ops_all_types<Abis>(DataTypes()), ...);
}

class simd_device_bitwise_ops_functor {
 public:
  KOKKOS_INLINE_FUNCTION void operator()(int) const {
    device_check_bitwise_ops_all_abis(
        Kokkos::Experimental::Impl::device_abi_set());
  }
};

TEST(simd, host_bitwise_ops) {
  host_check_bitwise_ops_all_abis(Kokkos::Experimental::Impl::host_abi_set());
}

TEST(simd, device_bitwise_ops) {
  Kokkos::parallel_for(1, simd_device_bitwise_ops_functor());
}

#endif
