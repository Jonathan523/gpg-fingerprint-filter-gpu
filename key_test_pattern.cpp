#include "key_test.hpp"

#include <map>
#include <memory>
#include <sstream>

// Define the pattern_check kernel statically
__global__ void pattern_check(u32 *result,
                              const u32 *h0,
                              const u32 *h1,
                              const u32 *h2,
                              const u32 *h3,
                              const u32 *h4) {
    // A simple example: check if the last nibble of h0 matches some condition.
    // Modify as needed for your pattern logic.
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char nibble = (h0[index] & 0xF);
    if (nibble == 0) *result = index;
}

// Remove compile_patterns, NVRTC calls, and other driver references.

void CudaManager::load_patterns(const std::string &input) {
    CU_CALL(cuMemAlloc, &cu_result, sizeof(uint32_t));
    CU_CALL(cuMemsetD32, cu_result, UINT32_MAX, 1);
}

void CudaManager::gpu_pattern_check() {
    void *args[] = {&cu_result, h + 0, h + 1, h + 2, h + 3, h + 4};
    CU_CALL(cuLaunchKernel,
            cu_kernel,
            n_block_, 1, 1,
            thread_per_block_, 1, 1,
            0, 0, args, 0);
}
