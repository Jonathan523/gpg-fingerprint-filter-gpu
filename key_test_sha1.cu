#include "key_test.hpp"

#ifndef LROT32
#define LROT32(x, n) (((x)<<(n))|((x)>>(32-(n))))
#endif

// DSA, no key should be longer
__constant__ static u32 chunk_buffer[208];

__forceinline__ __device__ static
void sha1_main_loop(u32 w[16], u32 &a, u32 &b, u32 &c, u32 &d, u32 &e) {
    // Rounds 0-15: use w[i] directly, CH function, k = 0x5A827999
#pragma unroll
    for (int i = 0; i < 16; i++) {
        u32 temp = LROT32(a, 5) + (d ^ (b & (c ^ d))) + e + 0x5A827999U + w[i];
        e = d; d = c; c = LROT32(b, 30); b = a; a = temp;
    }
    // Rounds 16-19: w expansion begins, still CH function, k = 0x5A827999
#pragma unroll
    for (int i = 16; i < 20; i++) {
        u32 wi = w[i%16] = LROT32(w[(i-3)%16] ^ w[(i-8)%16] ^ w[(i-14)%16] ^ w[i%16], 1);
        u32 temp = LROT32(a, 5) + (d ^ (b & (c ^ d))) + e + 0x5A827999U + wi;
        e = d; d = c; c = LROT32(b, 30); b = a; a = temp;
    }
    // Rounds 20-39: PARITY function, k = 0x6ED9EBA1
#pragma unroll
    for (int i = 20; i < 40; i++) {
        u32 wi = w[i%16] = LROT32(w[(i-3)%16] ^ w[(i-8)%16] ^ w[(i-14)%16] ^ w[i%16], 1);
        u32 temp = LROT32(a, 5) + (b ^ c ^ d) + e + 0x6ED9EBA1U + wi;
        e = d; d = c; c = LROT32(b, 30); b = a; a = temp;
    }
    // Rounds 40-59: MAJ function (simplified: saves one OR vs. original), k = 0x8F1BBCDC
#pragma unroll
    for (int i = 40; i < 60; i++) {
        u32 wi = w[i%16] = LROT32(w[(i-3)%16] ^ w[(i-8)%16] ^ w[(i-14)%16] ^ w[i%16], 1);
        u32 temp = LROT32(a, 5) + ((b & c) | ((b | c) & d)) + e + 0x8F1BBCDCU + wi;
        e = d; d = c; c = LROT32(b, 30); b = a; a = temp;
    }
    // Rounds 60-79: PARITY function, k = 0xCA62C1D6
#pragma unroll
    for (int i = 60; i < 80; i++) {
        u32 wi = w[i%16] = LROT32(w[(i-3)%16] ^ w[(i-8)%16] ^ w[(i-14)%16] ^ w[i%16], 1);
        u32 temp = LROT32(a, 5) + (b ^ c ^ d) + e + 0xCA62C1D6U + wi;
        e = d; d = c; c = LROT32(b, 30); b = a; a = temp;
    }
}

__global__ static
void proc_chunk0(u32 t0, u32* __restrict__ h0, u32* __restrict__ h1, u32* __restrict__ h2, u32* __restrict__ h3, u32* __restrict__ h4) {
    constexpr u32 a0 = 0x67452301U;
    constexpr u32 b0 = 0xEFCDAB89U;
    constexpr u32 c0 = 0x98BADCFEU;
    constexpr u32 d0 = 0x10325476U;
    constexpr u32 e0 = 0xC3D2E1F0U;

    // u32 index is safe: total threads = n_block * thread_per_block = time_offset,
    // which is bounded well below UINT32_MAX by the int batch_size in the constructor.
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    u32 a, b, c, d, e;

    u32 w[16];
    memcpy(w, chunk_buffer, sizeof(w));
    w[1] = t0 - index;

    a = a0;
    b = b0;
    c = c0;
    d = d0;
    e = e0;

    sha1_main_loop(w, a, b, c, d, e);

    h0[index] = a + a0;
    h1[index] = b + b0;
    h2[index] = c + c0;
    h3[index] = d + d0;
    h4[index] = e + e0;
}

__global__ static
void proc_chunk(u32 chunk_idx, u32 *h0, u32 *h1, u32 *h2, u32 *h3, u32 *h4) {
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    u32 a, b, c, d, e;

    u32 w[16];
    memcpy(w, chunk_buffer + chunk_idx * 16, sizeof(w));

    a = h0[index];
    b = h1[index];
    c = h2[index];
    d = h3[index];
    e = h4[index];

    sha1_main_loop(w, a, b, c, d, e);

    h0[index] += a;
    h1[index] += b;
    h2[index] += c;
    h3[index] += d;
    h4[index] += e;
}

void CudaManager::gpu_proc_chunk(u32 n_chunk, u32 key_time0) const {
    proc_chunk0<<<n_block_, thread_per_block_>>>(key_time0, h[0], h[1], h[2], h[3], h[4]);
    for (u32 i = 1; i < n_chunk; i++)
        proc_chunk<<<n_block_, thread_per_block_>>>(i, h[0], h[1], h[2], h[3], h[4]);
}

u32 CudaManager::load_key(const std::vector<u8> &pubkey) const {
    std::vector<u8> buf = pubkey;
    u32 buf_len = buf.size();

    // sha-1 padding
    int pad_zero = (56 - (buf_len + 1) % 64) % 64;
    u32 buf_len2 = buf_len + 1 + pad_zero + 8;

    buf.push_back(0x80);
    buf.resize(buf_len2, 0);

    buf_len *= 8;
    for (auto it = buf.rbegin(); buf_len && it != buf.rend(); it++) {
        *it = buf_len & 0xff;
        buf_len >>= 8;
    }

    // group buffer to 32-bit words
    for (u32 i = 0; i < buf_len2; i += 4) {
        std::swap(buf[i], buf[i + 3]);
        std::swap(buf[i + 1], buf[i + 2]);
    }

    DIE_ON_ERR(sizeof(chunk_buffer) >= buf_len2);

    u32 n_chunk = buf_len2 / 64;
    if (n_chunk == 1 && cu_key_chunk0 != 0) {
        // Single-chunk path: upload key block directly to NVRTC constant memory used
        // by the fused kernel; avoids the heavier cudaMemcpyToSymbol call.
        CU_CALL(cuMemcpyHtoD, cu_key_chunk0, buf.data(), 64);
    } else {
        CUDA_CALL(cudaMemcpyToSymbol, chunk_buffer, buf.data(), buf_len2);
    }

    return n_chunk;
}
