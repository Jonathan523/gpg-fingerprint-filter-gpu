#include "key_test.hpp"

// DSA, no key should be longer
__constant__ static u32 chunk_buffer[208];

// Rotate-left 32 bits. An inline function gives the compiler cleaner
// dependency information than a macro and makes future swaps to a PTX
// intrinsic trivial.
__device__ __forceinline__ static u32 rotl32(u32 x, int n) {
    return (x << n) | (x >> (32 - n));
}

__forceinline__ __device__ static
void sha1_main_loop(u32 w[16], u32 &a, u32 &b, u32 &c, u32 &d, u32 &e) {
    // Rounds 0-15: use w[i] directly, CH function, k = 0x5A827999
#pragma unroll
    for (int i = 0; i < 16; i++) {
        u32 temp = rotl32(a, 5) + (d ^ (b & (c ^ d))) + e + 0x5A827999U + w[i];
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }
    // Rounds 16-19: w expansion begins, still CH function, k = 0x5A827999
    // &15 instead of %16 avoids modulo when not unrolled; splitting the read
    // of w[j] from the write makes the data-flow unambiguous for the register
    // allocator.
#pragma unroll
    for (int i = 16; i < 20; i++) {
        int j = i & 15;
        u32 x = w[(i-3) & 15] ^ w[(i-8) & 15] ^ w[(i-14) & 15] ^ w[j];
        u32 wi = rotl32(x, 1);
        w[j] = wi;
        u32 temp = rotl32(a, 5) + (d ^ (b & (c ^ d))) + e + 0x5A827999U + wi;
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }
    // Rounds 20-39: PARITY function, k = 0x6ED9EBA1
#pragma unroll
    for (int i = 20; i < 40; i++) {
        int j = i & 15;
        u32 x = w[(i-3) & 15] ^ w[(i-8) & 15] ^ w[(i-14) & 15] ^ w[j];
        u32 wi = rotl32(x, 1);
        w[j] = wi;
        u32 temp = rotl32(a, 5) + (b ^ c ^ d) + e + 0x6ED9EBA1U + wi;
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }
    // Rounds 40-59: MAJ function (simplified: saves one OR vs. original), k = 0x8F1BBCDC
#pragma unroll
    for (int i = 40; i < 60; i++) {
        int j = i & 15;
        u32 x = w[(i-3) & 15] ^ w[(i-8) & 15] ^ w[(i-14) & 15] ^ w[j];
        u32 wi = rotl32(x, 1);
        w[j] = wi;
        u32 temp = rotl32(a, 5) + ((b & c) | ((b | c) & d)) + e + 0x8F1BBCDCU + wi;
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }
    // Rounds 60-79: PARITY function, k = 0xCA62C1D6
#pragma unroll
    for (int i = 60; i < 80; i++) {
        int j = i & 15;
        u32 x = w[(i-3) & 15] ^ w[(i-8) & 15] ^ w[(i-14) & 15] ^ w[j];
        u32 wi = rotl32(x, 1);
        w[j] = wi;
        u32 temp = rotl32(a, 5) + (b ^ c ^ d) + e + 0xCA62C1D6U + wi;
        e = d; d = c; c = rotl32(b, 30); b = a; a = temp;
    }
}

// Unified multi-chunk kernel: processes all SHA-1 message blocks in a single
// kernel launch, keeping the running hash state (ha..he) in registers throughout.
// This replaces the old proc_chunk0 + proc_chunk pair, eliminating:
//   - Multiple kernel launches (was one per chunk)
//   - Intermediate global-memory write + read of h[0-4] between chunks
//
// Chunk 0 is peeled out of the loop so the remaining iterations contain no
// conditional branches.
__global__ static
void proc_all_chunks(u32 n_chunk, u32 t0,
                     u32* __restrict__ h0, u32* __restrict__ h1,
                     u32* __restrict__ h2, u32* __restrict__ h3,
                     u32* __restrict__ h4) {
    constexpr u32 A0 = 0x67452301U;
    constexpr u32 B0 = 0xEFCDAB89U;
    constexpr u32 C0 = 0x98BADCFEU;
    constexpr u32 D0 = 0x10325476U;
    constexpr u32 E0 = 0xC3D2E1F0U;

    // u32 index is safe: total threads = n_block * thread_per_block (= time_offset),
    // capped by the `int batch_size = n_block * thread_per_block` allocation in
    // CudaManager::CudaManager(), which keeps it well below UINT32_MAX.
    u32 index = blockIdx.x * blockDim.x + threadIdx.x;

    u32 ha = A0, hb = B0, hc = C0, hd = D0, he = E0;

    // --- Chunk 0: explicit unrolled constant-memory load + timestamp patch ---
    // All threads in a warp read the same addresses → ideal broadcast behavior.
    {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[i];
        // w[1] carries the per-thread creation timestamp
        // (bytes 4-7 of the OpenPGP fingerprint hash packet as a little-endian u32).
        w[1] = t0 - index;
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_main_loop(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    // --- Remaining chunks (RSA keys only): no per-thread data ---
    for (u32 chunk = 1; chunk < n_chunk; chunk++) {
        u32 w[16];
#pragma unroll
        for (int i = 0; i < 16; i++)
            w[i] = chunk_buffer[chunk * 16 + i];
        u32 a = ha, b = hb, c = hc, d = hd, e = he;
        sha1_main_loop(w, a, b, c, d, e);
        ha += a; hb += b; hc += c; hd += d; he += e;
    }

    h0[index] = ha;
    h1[index] = hb;
    h2[index] = hc;
    h3[index] = hd;
    h4[index] = he;
}

void CudaManager::gpu_proc_chunk(u32 n_chunk, u32 key_time0) const {
    proc_all_chunks<<<n_block_, thread_per_block_>>>(n_chunk, key_time0,
                                                     h[0], h[1], h[2], h[3], h[4]);
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
