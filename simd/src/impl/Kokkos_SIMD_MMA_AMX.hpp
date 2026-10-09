// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_SIMD_MMA_IMPL_AMX_HPP
#define KOKKOS_SIMD_MMA_IMPL_AMX_HPP

#ifndef KOKKOS_ENABLE_EXPERIMENTAL_SIMD_AMX
#error "Kokkos_SIMD_MMA_AMX.hpp requires CPU backend that is AMX-capable"
#endif

#include <asm/prctl.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sys/syscall.h>
#include <type_traits>
#include <unistd.h>

#include <immintrin.h>

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <PrecisionType P>
struct FragmentDTypeImpl<Kokkos::DefaultHostExecutionSpace, P> {
  using type = std::conditional_t<
      P == PrecisionType::BF16, Kokkos::Experimental::bhalf_t,
      std::conditional_t<
          P == PrecisionType::Float, float,
          std::conditional_t<
              P == PrecisionType::Int8, std::int8_t,
              std::conditional_t<P == PrecisionType::Int32, std::int32_t,
                                 NotImplementedError<void>>>>>;
};

template <class ValueT>
inline constexpr int amx_input_bytes() {
  if constexpr (std::is_same_v<ValueT, Kokkos::Experimental::bhalf_t> ||
                std::is_same_v<ValueT, float>) {
    return 2;
  } else if constexpr (std::is_same_v<ValueT, std::int8_t> ||
                       std::is_same_v<ValueT, std::int32_t>) {
    return 1;
  } else {
    return 0;
  }
}

template <int InputBytes, int MMA_M, int MMA_N, int MMA_K>
struct AMXSupportedShape : std::false_type {};

template <>
struct AMXSupportedShape<2, 16, 16, 32> : std::true_type {};

template <>
struct AMXSupportedShape<2, 32, 32, 32> : std::true_type {};

template <>
struct AMXSupportedShape<1, 16, 16, 16> : std::true_type {};

template <class ValueT, int MMA_M, int MMA_N, int MMA_K>
inline constexpr bool amx_supported_shape_v =
    AMXSupportedShape<amx_input_bytes<ValueT>(), MMA_M, MMA_N, MMA_K>::value;

template <FragmentUse U, class DType>
inline constexpr bool amx_supported_dtype_v =
    (U == FragmentUse::Accumulator &&
     (std::is_same_v<DType, float> || std::is_same_v<DType, std::int32_t>)) ||
    ((U == FragmentUse::MatrixA || U == FragmentUse::MatrixB) &&
     (std::is_same_v<DType, Kokkos::Experimental::bhalf_t> ||
      std::is_same_v<DType, std::int8_t>));

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K, class DType>
inline constexpr bool amx_supported_fragment_v =
    amx_supported_dtype_v<U, DType> &&
    amx_supported_shape_v<DType, MMA_M, MMA_N, MMA_K>;

struct alignas(64) AMXTileConfig {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

template <class ValueT, int MMA_M, int MMA_N, int MMA_K>
struct AMXShapeLayout {
  static_assert(amx_supported_shape_v<ValueT, MMA_M, MMA_N, MMA_K>,
                "Kokkos SIMD MMA AMX supports BF16 "
                "mma_shape<16,16,32>/mma_shape<32,32,32> and INT8 "
                "mma_shape<16,16,16>");

  static constexpr int input_bytes = amx_input_bytes<ValueT>();
  static constexpr int vnni_pack   = 4 / input_bytes;

  static constexpr int physical_m = 16;
  static constexpr int physical_n = 16;
  static constexpr int physical_k = input_bytes == 1 ? 16 : 32;

  static constexpr int accumulator_tile_rows = MMA_M / physical_m;
  static constexpr int accumulator_tile_cols = MMA_N / physical_n;
  static constexpr int accumulator_tile_count =
      accumulator_tile_rows * accumulator_tile_cols;
  static constexpr int matrix_a_tile_count = accumulator_tile_rows;
  static constexpr int matrix_b_tile_count = accumulator_tile_cols;
  static constexpr int used_tile_count =
      accumulator_tile_count + matrix_a_tile_count + matrix_b_tile_count;

  static_assert(used_tile_count <= 8,
                "AMX MMA shape requires more than 8 tile registers");

  static constexpr int accumulator_storage_stride_bytes = MMA_N * 4;
  static constexpr int matrix_a_storage_stride_bytes    = MMA_K * input_bytes;
  static constexpr int matrix_b_storage_stride_bytes =
      MMA_N * vnni_pack * input_bytes;

  static constexpr int accumulator_tile_colsb = physical_n * 4;
  static constexpr int matrix_a_tile_colsb    = physical_k * input_bytes;
  static constexpr int matrix_b_tile_colsb =
      physical_n * vnni_pack * input_bytes;
};

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K, class DType,
          class Layout>
struct AMXFragment {
  static_assert(is_supported_operand_layout_v<Layout>,
                "AMX fragments support only Kokkos::layout_left and "
                "Kokkos::layout_right operand layouts");

  using value_type = DType;

  static constexpr FragmentUse use = U;
  using layout                     = Layout;

  static constexpr int mma_m = MMA_M;
  static constexpr int mma_n = MMA_N;
  static constexpr int mma_k = MMA_K;

  static constexpr int input_bytes = amx_input_bytes<DType>();
  static constexpr int vnni_pack   = 4 / input_bytes;

  static constexpr int storage_bytes =
      U == FragmentUse::Accumulator ? MMA_M * MMA_N * 4
      : U == FragmentUse::MatrixA
          ? MMA_M * MMA_K * input_bytes
          : (MMA_K / vnni_pack) * MMA_N * vnni_pack * input_bytes;

  alignas(64) uint8_t storage[storage_bytes];
};

inline uint16_t amx_float_to_bf16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t lsb           = (bits >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

template <class ValueT>
inline uint16_t amx_to_bf16_bits(ValueT value) {
  return amx_float_to_bf16_bits(static_cast<float>(value));
}

template <int TileId>
inline void amx_tile_loadd(const void* base, const int stride) {
  static_assert(TileId >= 0 && TileId <= 7, "AMX tile id must be in [0, 7]");

  if constexpr (TileId == 0) {
    _tile_loadd(0, base, stride);
  } else if constexpr (TileId == 1) {
    _tile_loadd(1, base, stride);
  } else if constexpr (TileId == 2) {
    _tile_loadd(2, base, stride);
  } else if constexpr (TileId == 3) {
    _tile_loadd(3, base, stride);
  } else if constexpr (TileId == 4) {
    _tile_loadd(4, base, stride);
  } else if constexpr (TileId == 5) {
    _tile_loadd(5, base, stride);
  } else if constexpr (TileId == 6) {
    _tile_loadd(6, base, stride);
  } else if constexpr (TileId == 7) {
    _tile_loadd(7, base, stride);
  }
}

template <int TileId>
inline void amx_tile_stored(void* base, const int stride) {
  static_assert(TileId >= 0 && TileId <= 7, "AMX tile id must be in [0, 7]");

  if constexpr (TileId == 0) {
    _tile_stored(0, base, stride);
  } else if constexpr (TileId == 1) {
    _tile_stored(1, base, stride);
  } else if constexpr (TileId == 2) {
    _tile_stored(2, base, stride);
  } else if constexpr (TileId == 3) {
    _tile_stored(3, base, stride);
  } else if constexpr (TileId == 4) {
    _tile_stored(4, base, stride);
  } else if constexpr (TileId == 5) {
    _tile_stored(5, base, stride);
  } else if constexpr (TileId == 6) {
    _tile_stored(6, base, stride);
  } else if constexpr (TileId == 7) {
    _tile_stored(7, base, stride);
  }
}

inline void amx_request_tiledata_permission() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, ARCH_XCOMP_TILEDATA) !=
        0) {
      std::fprintf(stderr, "Failed to request AMX XTILEDATA permission: %s\n",
                   std::strerror(errno));
      std::abort();
    }
  });
}

inline void amx_ensure_tile_config_shape(int mma_m, int mma_n, int mma_k,
                                         int input_bytes) {
  thread_local int configured_m     = -1;
  thread_local int configured_n     = -1;
  thread_local int configured_k     = -1;
  thread_local int configured_bytes = -1;

  if (configured_m == mma_m && configured_n == mma_n && configured_k == mma_k &&
      configured_bytes == input_bytes) {
    return;
  }

  const bool bf16_shape =
      input_bytes == 2 && ((mma_m == 16 && mma_n == 16 && mma_k == 32) ||
                           (mma_m == 32 && mma_n == 32 && mma_k == 32));
  const bool int8_shape =
      input_bytes == 1 && mma_m == 16 && mma_n == 16 && mma_k == 16;

  if (!bf16_shape && !int8_shape) {
    std::fprintf(stderr,
                 "Kokkos SIMD MMA AMX supports BF16 "
                 "mma_shape<16,16,32>/mma_shape<32,32,32> and INT8 "
                 "mma_shape<16,16,16>\n");
    std::abort();
  }

  const int vnni_pack = 4 / input_bytes;
  const int row_tiles = mma_m / 16;
  const int col_tiles = mma_n / 16;
  const int acc_count = row_tiles * col_tiles;
  const int a_base    = acc_count;
  const int b_base    = a_base + row_tiles;

  AMXTileConfig config = {};
  config.palette_id    = 1;

  for (int rb = 0; rb < row_tiles; ++rb) {
    for (int cb = 0; cb < col_tiles; ++cb) {
      const int tile     = rb * col_tiles + cb;
      config.rows[tile]  = 16;
      config.colsb[tile] = 16 * 4;
    }
  }

  for (int rb = 0; rb < row_tiles; ++rb) {
    const int tile     = a_base + rb;
    config.rows[tile]  = 16;
    config.colsb[tile] = mma_k * input_bytes;
  }

  for (int cb = 0; cb < col_tiles; ++cb) {
    const int tile     = b_base + cb;
    config.rows[tile]  = mma_k / vnni_pack;
    config.colsb[tile] = 16 * vnni_pack * input_bytes;
  }

  _tile_loadconfig(&config);

  configured_m     = mma_m;
  configured_n     = mma_n;
  configured_k     = mma_k;
  configured_bytes = input_bytes;
}

template <class FragmentT>
inline void amx_ensure_tile_config() {
  using Layout =
      AMXShapeLayout<typename FragmentT::value_type, FragmentT::mma_m,
                     FragmentT::mma_n, FragmentT::mma_k>;

  static_assert(FragmentT::mma_k % Layout::vnni_pack == 0,
                "AMX fragments require K divisible by the VNNI pack factor");
  static_assert(Layout::physical_m <= 16, "AMX supports at most 16 tile rows");
  static_assert(Layout::accumulator_tile_colsb <= 64,
                "AMX accumulator tile rows must fit in 64 bytes");
  static_assert(Layout::matrix_a_tile_colsb <= 64,
                "AMX A tile rows must fit in 64 bytes");
  static_assert(Layout::matrix_b_tile_colsb <= 64,
                "AMX packed B tile rows must fit in 64 bytes");

  amx_request_tiledata_permission();
  amx_ensure_tile_config_shape(FragmentT::mma_m, FragmentT::mma_n,
                               FragmentT::mma_k, Layout::input_bytes);
}

template <class FragmentT, class ValueT>
inline void load_matrix_sync(
    ExecutionSpaceTag<Kokkos::DefaultHostExecutionSpace>,
    FragmentT& destination, const ValueT* source, const int stride) {
  auto read_source = [&](const int i, const int j) {
    if constexpr (std::is_same_v<typename FragmentT::layout,
                                 Kokkos::layout_right>) {
      return source[i * stride + j];
    } else {
      return source[i + j * stride];
    }
  };

  constexpr bool is_int8 =
      std::is_same_v<typename FragmentT::value_type, std::int8_t>;
  constexpr int pack = FragmentT::vnni_pack;

  if constexpr (FragmentT::use == FragmentUse::MatrixA) {
    if constexpr (is_int8) {
      auto* packed = reinterpret_cast<std::int8_t*>(destination.storage);
      for (int i = 0; i < FragmentT::mma_m; ++i) {
        for (int k = 0; k < FragmentT::mma_k; ++k) {
          packed[i * FragmentT::mma_k + k] =
              static_cast<std::int8_t>(read_source(i, k));
        }
      }
    } else {
      auto* packed = reinterpret_cast<uint16_t*>(destination.storage);
      for (int i = 0; i < FragmentT::mma_m; ++i) {
        for (int k = 0; k < FragmentT::mma_k; ++k) {
          packed[i * FragmentT::mma_k + k] =
              amx_to_bf16_bits(read_source(i, k));
        }
      }
    }
  } else if constexpr (FragmentT::use == FragmentUse::MatrixB) {
    if constexpr (is_int8) {
      auto* packed = reinterpret_cast<std::int8_t*>(destination.storage);
      for (int k_group = 0; k_group < FragmentT::mma_k / pack; ++k_group) {
        for (int n = 0; n < FragmentT::mma_n; ++n) {
          for (int r = 0; r < pack; ++r) {
            packed[k_group * (pack * FragmentT::mma_n) + pack * n + r] =
                static_cast<std::int8_t>(read_source(pack * k_group + r, n));
          }
        }
      }
    } else {
      auto* packed = reinterpret_cast<uint16_t*>(destination.storage);
      for (int k_group = 0; k_group < FragmentT::mma_k / pack; ++k_group) {
        for (int n = 0; n < FragmentT::mma_n; ++n) {
          for (int r = 0; r < pack; ++r) {
            packed[k_group * (pack * FragmentT::mma_n) + pack * n + r] =
                amx_to_bf16_bits(read_source(pack * k_group + r, n));
          }
        }
      }
    }
  } else {
    NotImplementedError<FragmentT>();
  }
}

template <class FragmentT, class ValueT>
inline void fill_fragment(ExecutionSpaceTag<Kokkos::DefaultHostExecutionSpace>,
                          FragmentT& fragment, ValueT value) {
  if constexpr (FragmentT::use == FragmentUse::Accumulator) {
    using acc_t  = typename FragmentT::value_type;
    auto* packed = reinterpret_cast<acc_t*>(fragment.storage);
    for (int i = 0; i < FragmentT::mma_m * FragmentT::mma_n; ++i) {
      packed[i] = static_cast<acc_t>(value);
    }
  } else {
    NotImplementedError<FragmentT>();
  }
}

template <int TileId, int RowBlock, int ColBlock, class FragmentT>
inline void amx_load_accumulator_tile(const FragmentT& fragment) {
  using Layout =
      AMXShapeLayout<typename FragmentT::value_type, FragmentT::mma_m,
                     FragmentT::mma_n, FragmentT::mma_k>;

  const auto* packed = reinterpret_cast<const uint8_t*>(fragment.storage);
  const void* tile_base =
      packed +
      RowBlock * Layout::physical_m * Layout::accumulator_storage_stride_bytes +
      ColBlock * Layout::physical_n * 4;
  amx_tile_loadd<TileId>(tile_base, Layout::accumulator_storage_stride_bytes);
}

template <int TileId, int RowBlock, int ColBlock, class FragmentT>
inline void amx_store_accumulator_tile(FragmentT& fragment) {
  using Layout =
      AMXShapeLayout<typename FragmentT::value_type, FragmentT::mma_m,
                     FragmentT::mma_n, FragmentT::mma_k>;

  auto* packed = reinterpret_cast<uint8_t*>(fragment.storage);
  void* tile_base =
      packed +
      RowBlock * Layout::physical_m * Layout::accumulator_storage_stride_bytes +
      ColBlock * Layout::physical_n * 4;
  amx_tile_stored<TileId>(tile_base, Layout::accumulator_storage_stride_bytes);
}

template <int TileId, int RowBlock, class FragmentT>
inline void amx_load_matrix_a_tile(const FragmentT& fragment) {
  using Layout =
      AMXShapeLayout<typename FragmentT::value_type, FragmentT::mma_m,
                     FragmentT::mma_n, FragmentT::mma_k>;

  const auto* packed    = reinterpret_cast<const uint8_t*>(fragment.storage);
  const void* tile_base = packed + RowBlock * Layout::physical_m *
                                       Layout::matrix_a_storage_stride_bytes;
  amx_tile_loadd<TileId>(tile_base, Layout::matrix_a_storage_stride_bytes);
}

template <int TileId, int ColBlock, class FragmentT>
inline void amx_load_matrix_b_tile(const FragmentT& fragment) {
  using Layout =
      AMXShapeLayout<typename FragmentT::value_type, FragmentT::mma_m,
                     FragmentT::mma_n, FragmentT::mma_k>;

  const auto* packed    = reinterpret_cast<const uint8_t*>(fragment.storage);
  const void* tile_base = packed + ColBlock * Layout::physical_n *
                                       Layout::vnni_pack * Layout::input_bytes;
  amx_tile_loadd<TileId>(tile_base, Layout::matrix_b_storage_stride_bytes);
}

template <class FragmentT, class ValueT>
inline void store_matrix_sync(
    ExecutionSpaceTag<Kokkos::DefaultHostExecutionSpace>, ValueT* destination,
    FragmentT& source, const int stride, const bool row_major) {
  static_assert(FragmentT::use == FragmentUse::Accumulator,
                "AMX store_matrix_sync expects an accumulator fragment");

  using acc_t        = typename FragmentT::value_type;
  const auto* packed = reinterpret_cast<const acc_t*>(source.storage);

  for (int i = 0; i < FragmentT::mma_m; ++i) {
    for (int j = 0; j < FragmentT::mma_n; ++j) {
      const int offset = row_major ? i * stride + j : i + j * stride;
      destination[offset] =
          static_cast<ValueT>(packed[i * FragmentT::mma_n + j]);
    }
  }
}

template <class DFragT, class AFragT, class BFragT, class CFragT>
inline void mma_sync(ExecutionSpaceTag<Kokkos::DefaultHostExecutionSpace>,
                     DFragT& d_frag, AFragT& a_frag, BFragT& b_frag,
                     CFragT& c_frag) {
  static_assert(DFragT::use == FragmentUse::Accumulator,
                "AMX d fragment must be an accumulator");
  static_assert(AFragT::use == FragmentUse::MatrixA,
                "AMX a fragment must be matrix A");
  static_assert(BFragT::use == FragmentUse::MatrixB,
                "AMX b fragment must be matrix B");
  static_assert(CFragT::use == FragmentUse::Accumulator,
                "AMX c fragment must be an accumulator");

  static_assert(
      DFragT::mma_m == AFragT::mma_m && DFragT::mma_m == CFragT::mma_m,
      "AMX MMA fragments must agree on M");
  static_assert(
      DFragT::mma_n == BFragT::mma_n && DFragT::mma_n == CFragT::mma_n,
      "AMX MMA fragments must agree on N");
  static_assert(
      AFragT::mma_k == BFragT::mma_k && AFragT::mma_k == DFragT::mma_k,
      "AMX MMA fragments must agree on K");

  static_assert(
      std::is_same_v<typename DFragT::value_type, typename CFragT::value_type>,
      "AMX d and c accumulator fragments must use the same type");
  static_assert((std::is_same_v<typename AFragT::value_type,
                                Kokkos::Experimental::bhalf_t> &&
                 std::is_same_v<typename BFragT::value_type,
                                Kokkos::Experimental::bhalf_t> &&
                 std::is_same_v<typename DFragT::value_type, float>) ||
                    (std::is_same_v<typename AFragT::value_type, std::int8_t> &&
                     std::is_same_v<typename BFragT::value_type, std::int8_t> &&
                     std::is_same_v<typename DFragT::value_type, std::int32_t>),
                "AMX supports BF16/BF16->FP32 and INT8/INT8->INT32 MMA");

  amx_ensure_tile_config<DFragT>();

  if constexpr (DFragT::mma_m == 16 && DFragT::mma_n == 16 &&
                DFragT::mma_k == 16) {
    amx_load_accumulator_tile<0, 0, 0>(c_frag);
    amx_load_matrix_a_tile<1, 0>(a_frag);
    amx_load_matrix_b_tile<2, 0>(b_frag);
    _tile_dpbssd(0, 1, 2);
    amx_store_accumulator_tile<0, 0, 0>(d_frag);
  } else if constexpr (DFragT::mma_m == 16 && DFragT::mma_n == 16 &&
                       DFragT::mma_k == 32) {
    amx_load_accumulator_tile<0, 0, 0>(c_frag);
    amx_load_matrix_a_tile<1, 0>(a_frag);
    amx_load_matrix_b_tile<2, 0>(b_frag);
    _tile_dpbf16ps(0, 1, 2);
    amx_store_accumulator_tile<0, 0, 0>(d_frag);
  } else if constexpr (DFragT::mma_m == 32 && DFragT::mma_n == 32 &&
                       DFragT::mma_k == 32) {
    amx_load_accumulator_tile<0, 0, 0>(c_frag);
    amx_load_accumulator_tile<1, 0, 1>(c_frag);
    amx_load_accumulator_tile<2, 1, 0>(c_frag);
    amx_load_accumulator_tile<3, 1, 1>(c_frag);

    amx_load_matrix_a_tile<4, 0>(a_frag);
    amx_load_matrix_a_tile<5, 1>(a_frag);
    amx_load_matrix_b_tile<6, 0>(b_frag);
    amx_load_matrix_b_tile<7, 1>(b_frag);

    _tile_dpbf16ps(0, 4, 6);
    _tile_dpbf16ps(1, 4, 7);
    _tile_dpbf16ps(2, 5, 6);
    _tile_dpbf16ps(3, 5, 7);

    amx_store_accumulator_tile<0, 0, 0>(d_frag);
    amx_store_accumulator_tile<1, 0, 1>(d_frag);
    amx_store_accumulator_tile<2, 1, 0>(d_frag);
    amx_store_accumulator_tile<3, 1, 1>(d_frag);
  }
}

template <FragmentUse U, int MMA_M, int MMA_N, int MMA_K, class DType,
          class Layout>
struct NativeFragmentTImpl<Kokkos::DefaultHostExecutionSpace, U, MMA_M, MMA_N,
                           MMA_K, DType, Layout> {
  static_assert(amx_supported_fragment_v<U, MMA_M, MMA_N, MMA_K, DType>,
                "Kokkos SIMD MMA AMX supports BF16 "
                "mma_shape<16,16,32>/mma_shape<32,32,32> and INT8 "
                "mma_shape<16,16,16>");

  using type = AMXFragment<U, MMA_M, MMA_N, MMA_K, DType, Layout>;
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif
