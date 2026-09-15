// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestCuda_Category.hpp>

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstring>

#include <cupti.h>

// Kokkos can hand a functor to the device three different ways, and which one
// it picks depends only on how big the functor is. This test watches the CUDA
// runtime calls with CUPTI to confirm we get the route we expect.
//
// The CUPTI usage follows the callback_timestamp sample that ships with the
// toolkit, in both CTK 12.9 and 13.2
// (extras/CUPTI/samples/callback_timestamp/callback_timestamp.cu): register a
// CUpti_CallbackFunc, enable CUPTI_CB_DOMAIN_RUNTIME_API, look only at
// CUPTI_API_ENTER, and read the API name out of
// CUpti_CallbackData::functionName.
//
// The doc links are pinned to 13.2.0 for stability. The toolkit installs those
// same pages offline under extras/CUPTI/doc/html/api/, and everything cited is
// declared in the 12.9 headers as well.

namespace {

constexpr int num_elements = 256;

// Each padding below targets one of the thresholds in Impl::CudaTraits
// (core/src/Cuda/Kokkos_Cuda_Instance.hpp): ConstantMemoryUseThreshold is
// 512 B, ConstantMemoryUsage is 32 KiB, and KernelArgumentLimit is 4 KiB, or
// 32 KiB once __grid_constant__ is in play. That last pair tracks CUDA itself,
// which raised the kernel-parameter ceiling from 4,096 to 32,764 bytes on Volta
// and newer in 12.1:
// https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/
//
// Kokkos routes on sizeof(DriverType), which is the functor plus the policy, so
// the sizes are a little larger than the padding alone. The values below are
// chosen to sit clear of the nearest threshold rather than right against it.

// Under the smallest threshold of 512 B, so this goes as a kernel argument
// regardless of how Kokkos was configured.
constexpr std::size_t kernel_argument_padding_bytes = 64;
// With HintHeavyWeight and constant memory compiled in, this fits under the
// 32 KiB constant-memory ceiling. Without constant memory it falls back to
// kernel arguments, which allow 4 KiB (32 KiB with grid constant), so 1 KiB is
// inside either limit.
constexpr std::size_t constant_memory_padding_bytes = 1024;
// Too big for kernel arguments or constant memory, both capped at 32 KiB, which
// leaves global memory as the only route.
constexpr std::size_t global_memory_padding_bytes = 33000;

struct CallbackFlags {
  int to_symbol    = 0;
  int memcpy_async = 0;
  int launch       = 0;

  void clear() {
    to_symbol    = 0;
    memcpy_async = 0;
    launch       = 0;
  }
};

CallbackFlags callback_flags;
CUpti_SubscriberHandle subscriber;

// Only CUPTI_CB_DOMAIN_RUNTIME_API is ever enabled, so cbdata is always a
// CUpti_CallbackData:
// https://docs.nvidia.com/cupti/13.2.0/api/structCUpti__CallbackData.html#structcupti__callbackdata
// Every call fires twice, once on the way in and once on the way out, so
// filtering on CUPTI_API_ENTER counts each one exactly once:
// https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv4N21CUpti_ApiCallbackSite15CUPTI_API_ENTERE
void CUPTIAPI callback(void*, CUpti_CallbackDomain, CUpti_CallbackId,
                       const void* callback_data) {
  const auto* data = static_cast<const CUpti_CallbackData*>(callback_data);
  if (data->callbackSite != CUPTI_API_ENTER || data->functionName == nullptr)
    return;

  // functionName is whichever runtime API function triggered the callback, and
  // CUPTI guarantees the string is a global constant, so comparing against it
  // directly is safe:
  // https://docs.nvidia.com/cupti/13.2.0/api/structCUpti__CallbackData.html#structcupti__callbackdata
  //
  // Note that the name is the plain API spelling, not the versioned one used by
  // the callback ID enums. The callback_timestamp sample relies on the same
  // behavior: it stores functionName (line 148) and later compares it to
  // "cudaMemcpy" and "cudaLaunchKernel" (lines 236 and 243) in CTK 13.2's
  // extras/CUPTI/samples/callback_timestamp/callback_timestamp.cu.
  //
  // The sample does not cover the two memcpy names we need, so those were
  // confirmed empirically: raw CUDA probes under CUPTI on both 12.9 and 13.2
  // print exactly these strings. Matching on names rather than on
  // CUpti_runtime_api_trace_cbid values is deliberate, since those enum values
  // carry no cross-version stability guarantee.
  if (std::strcmp(data->functionName, "cudaMemcpyToSymbolAsync") == 0)
    callback_flags.to_symbol = 1;
  else if (std::strcmp(data->functionName, "cudaMemcpyAsync") == 0)
    callback_flags.memcpy_async = 1;
  else if (std::strcmp(data->functionName, "cudaLaunchKernel") == 0)
    callback_flags.launch = 1;
}

void start_detector() {
  callback_flags.clear();
  // RUNTIME_API covers every CUDA runtime function, which saves enabling
  // callbacks one ID at a time:
  // https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv4N20CUpti_CallbackDomain27CUPTI_CB_DOMAIN_RUNTIME_APIE
  // The leading 1 is the enable flag, not a count or a handle; anything
  // non-zero switches the whole domain on:
  // https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv417cuptiEnableDomain8uint32_t22CUpti_SubscriberHandle20CUpti_CallbackDomain
  ASSERT_EQ(CUPTI_SUCCESS,
            cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
}

CallbackFlags stop_detector() {
  // Same call with 0, which turns the domain back off:
  // https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv417cuptiEnableDomain8uint32_t22CUpti_SubscriberHandle20CUpti_CallbackDomain
  EXPECT_EQ(CUPTI_SUCCESS,
            cuptiEnableDomain(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  return callback_flags;
}

// The body is empty on purpose. Only the size of this type steers the routing
// decision, so padding is all the functor needs to carry. Being a plain
// unsigned char array, sizeof(SizedFunctor) is exactly PaddingBytes.
template <std::size_t PaddingBytes>
struct SizedFunctor {
  unsigned char padding[PaddingBytes] = {};

  KOKKOS_FUNCTION void operator()(const int) const {}
};

template <std::size_t PaddingBytes, class Policy, class ExecutionSpace>
CallbackFlags measure_parallel_for(const char* label, const Policy& policy,
                                   const ExecutionSpace& execution_space) {
  SizedFunctor<PaddingBytes> functor;
  start_detector();
  Kokkos::parallel_for(label, policy, functor);
  // Fence inside the window so the launch has definitely been issued by the
  // time we stop listening. Kokkos checks for CUDA errors here, so a launch
  // that the driver rejects outright still fails the test.
  execution_space.fence();
  return stop_detector();
}

void expect_callbacks(const CallbackFlags& observed, const int to_symbol,
                      const int memcpy_async, const int launch) {
  EXPECT_EQ(to_symbol, observed.to_symbol);
  EXPECT_EQ(memcpy_async, observed.memcpy_async);
  EXPECT_EQ(launch, observed.launch);
}

class CudaCuptiLaunchRoute : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Subscribing only registers the callback; it deliberately enables nothing,
    // which is why start_detector has to do that separately:
    // https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv414cuptiSubscribeP22CUpti_SubscriberHandle18CUpti_CallbackFuncPv
    // The docs recommend cuptiSubscribe_v2, but that one is not in the 12.9
    // headers, and plain cuptiSubscribe remains fully supported.
    //
    // Note that CUPTI allows only one subscriber per process, so this returns
    // CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if the test runs under
    // Nsight Systems, Nsight Compute, cuda-gdb or compute-sanitizer.
    ASSERT_EQ(CUPTI_SUCCESS, cuptiSubscribe(&subscriber, callback, nullptr));
  }

  static void TearDownTestSuite() {
    // https://docs.nvidia.com/cupti/13.2.0/api/group__CUPTI__CALLBACK__API.html#_CPPv416cuptiUnsubscribe22CUpti_SubscriberHandle
    EXPECT_EQ(CUPTI_SUCCESS, cuptiUnsubscribe(subscriber));
  }
};

// A host-space parallel_for should not touch the CUDA runtime at all, whatever
// the functor size. This is the control case: if it ever reported a callback,
// the counts in the CUDA test below would be measuring something other than the
// launch route.
TEST_F(CudaCuptiLaunchRoute, host_parallel_for_has_no_cuda_callbacks) {
  Kokkos::DefaultHostExecutionSpace execution_space;
  const auto policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
      execution_space, 0, num_elements);

  auto observed = measure_parallel_for<kernel_argument_padding_bytes>(
      "host_small", policy, execution_space);
  expect_callbacks(observed, 0, 0, 0);

  observed = measure_parallel_for<constant_memory_padding_bytes>(
      "host_medium", policy, execution_space);
  expect_callbacks(observed, 0, 0, 0);

  observed = measure_parallel_for<global_memory_padding_bytes>(
      "host_large", policy, execution_space);
  expect_callbacks(observed, 0, 0, 0);
}

TEST_F(CudaCuptiLaunchRoute, cuda_parallel_for_uses_expected_route) {
  Kokkos::Cuda execution_space;
  const auto policy =
      Kokkos::RangePolicy<Kokkos::Cuda>(execution_space, 0, num_elements);
  const auto heavy_policy = Kokkos::Experimental::require(
      policy, Kokkos::Experimental::WorkItemProperty::HintHeavyWeight);

  // Small enough to travel as a kernel argument, so the launch is the only
  // runtime call we expect.
  auto observed = measure_parallel_for<kernel_argument_padding_bytes>(
      "device_small", policy, execution_space);
  expect_callbacks(observed, 0, 0, 1);

  // HintHeavyWeight opts into constant memory, which shows up as a copy to the
  // constant buffer symbol. With constant memory compiled out there is nothing
  // to copy and this size still fits in kernel arguments.
  observed = measure_parallel_for<constant_memory_padding_bytes>(
      "device_medium", heavy_policy, execution_space);
#ifdef KOKKOS_ENABLE_IMPL_CUDA_CONSTANT_MEMORY
  expect_callbacks(observed, 1, 0, 1);
#else
  expect_callbacks(observed, 0, 0, 1);
#endif

  // Past every limit, so the functor is staged through global memory with an
  // ordinary async copy.
  observed = measure_parallel_for<global_memory_padding_bytes>(
      "device_large", policy, execution_space);
  expect_callbacks(observed, 0, 1, 1);
}

}  // namespace
