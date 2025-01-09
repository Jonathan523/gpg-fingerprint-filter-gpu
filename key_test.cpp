#include "key_test.hpp"

const char *cuGetErrorName_wrapper(CUresult err) {
    const char *msg;
    cuGetErrorName(err, &msg);
    return msg;
}

CudaManager::CudaManager(int n_block, int thread_per_block, unsigned long base_time):
        n_block_(n_block), thread_per_block_(thread_per_block), base_time_(base_time) {
    cudaSetDevice(0);
    int batch_size = n_block_ * thread_per_block_;

    // Allocate memory for h arrays
    for (auto &ptr: h)
        CUDA_CALL(cudaMalloc, (void**)&ptr, batch_size * sizeof(u32));

    // Allocate memory for pattern check result
    CUDA_CALL(cudaMalloc, (void**)&d_result, sizeof(u32));
    CUDA_CALL(cudaMemset, d_result, 0xFF, sizeof(u32));
}

CudaManager::~CudaManager() {
    if (cu_module != nullptr)
        cuModuleUnload(cu_module);

    if (cu_result)
        cuMemFree(cu_result);

    CUDA_CALL(cudaDeviceSynchronize);
    CUDA_CALL(cudaPeekAtLastError);

    // Free device arrays
    for (auto &ptr: h)
        cudaFree(ptr);

    if (d_result)
        cudaFree(d_result);

    if (cu_context != nullptr)
        CU_CALL(cuDevicePrimaryCtxRelease, cu_device);
}

void CudaManager::test_key(const std::vector<u8> &key) {
    auto n_chunk = load_key(key);
    key_time0 = base_time_;

    gpu_proc_chunk(n_chunk, key_time0);
    gpu_pattern_check();
}

u32 CudaManager::get_result_time() const {
    u32 offset;
    cudaMemcpy(&offset, d_result, sizeof(u32), cudaMemcpyDeviceToHost);
    if (offset != UINT32_MAX)
        cudaMemset(d_result, 0xFF, sizeof(u32));
    return offset == UINT32_MAX ? UINT32_MAX : key_time0 - offset;
}

void CudaManager::load_patterns(const std::string &input) {
    // Remove NVRTC / driver-based dynamic loading.
    // Instead, no-op or store the pattern if desired.
}

void CudaManager::gpu_pattern_check() {
    // Call the statically compiled kernel
    pattern_check<<<n_block_, thread_per_block_>>>(d_result, h[0], h[1], h[2], h[3], h[4]);
}
