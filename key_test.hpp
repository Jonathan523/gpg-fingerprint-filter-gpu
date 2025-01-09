#ifndef _KEY_TEST_HPP_
#define _KEY_TEST_HPP_

#include <vector>
#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "error_check.hpp"
#define CUDA_CALL(func, args...) error_wrapper<cudaError_t>(#func, (func)(args), cudaSuccess, cudaGetErrorString)

using u32 = std::uint32_t;
using u8 = std::uint8_t;

class CudaManager {
private:
    u32 *h[5] = {};
    u32 key_time0 = 0;

    int n_block_;
    int thread_per_block_;
    unsigned long base_time_;

    u32 load_key(const std::vector<u8> &pubkey) const;
    void gpu_proc_chunk(u32 n_chunk, u32 key_time0) const;
    void gpu_pattern_check();

    u32 *d_result = nullptr;  // device pointer for results

public:
    CudaManager(int n_block, int thread_per_block, unsigned long base_time);

    CudaManager(const CudaManager&) = delete;
    CudaManager& operator= (const CudaManager&) = delete;

    ~CudaManager();

    void load_patterns(const std::string &input);
    void test_key(const std::vector<u8> &key);
    u32 get_result_time() const;
};

#endif
